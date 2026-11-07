"""Validate structured LLM plans and execute confined file/database actions.

No model output is evaluated as Python or passed to a shell. All operational
files live in ``data_dir/workspace``; tasks and audit records live outside that
workspace in a SQLite database. File operations use directory descriptors and
``O_NOFOLLOW`` where supported to resist symlink substitution.
"""

from __future__ import annotations

import csv
import io
import json
import os
import secrets
import sqlite3
import stat
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class ValidationError(ValueError):
    """An LLM plan or argument violates the supported tool contract."""


class ExecutionError(RuntimeError):
    """A validated action cannot be safely completed."""


MAX_ACTIONS = 8
MAX_CONTENT_BYTES = 50_000
ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}
BLOCKED_NAMES = {"credentials.json", "secrets.json", "api_keys.json"}
STATUSES = {"all", "pending", "completed"}


def _object(properties: dict, required: list[str] | None = None) -> dict:
    result = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        result["required"] = required
    return result


_path = {"type": "string", "minLength": 1, "maxLength": 240}
_status = {"type": "string", "enum": sorted(STATUSES)}
_tool_args = {
    "list_files": _object({"directory": _path}),
    "read_file": _object({"path": _path}, ["path"]),
    "write_file": _object({
        "path": _path, "content": {"type": "string", "maxLength": MAX_CONTENT_BYTES},
        "overwrite": {"type": "boolean"},
    }, ["path", "content"]),
    "delete_file": _object({"path": _path}, ["path"]),
    "create_task": _object({
        "title": {"type": "string", "minLength": 1, "maxLength": 240},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "due_date": {"type": ["string", "null"], "description": "YYYY-MM-DD, or null"},
    }, ["title"]),
    "list_tasks": _object({"status": _status}),
    "complete_task": _object({"task_id": {"type": "integer", "minimum": 1}}, ["task_id"]),
    "delete_task": _object({"task_id": {"type": "integer", "minimum": 1}}, ["task_id"]),
    "export_tasks": _object({"path": _path, "status": _status}, ["path"]),
}

TOOL_SCHEMA = _object({
    "summary": {"type": "string", "minLength": 1, "maxLength": 1000},
    "actions": {
        "type": "array", "minItems": 0, "maxItems": MAX_ACTIONS,
        "items": {"oneOf": [
            _object({"tool": {"type": "string", "const": name}, "args": schema}, ["tool", "args"])
            for name, schema in _tool_args.items()
        ]},
    },
}, ["summary", "actions"])


def requires_confirmation(plan: dict) -> bool:
    """Return whether a plan deletes data or explicitly overwrites a file."""
    return any(
        action.get("tool") in {"delete_file", "delete_task"}
        or (action.get("tool") == "write_file" and action.get("args", {}).get("overwrite") is True)
        for action in plan.get("actions", [])
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_audit(value: Any, key: str = "", depth: int = 0) -> Any:
    """Keep useful action metadata while omitting prompts, contents, and keys."""
    key_lower = key.lower()
    if key_lower in {"key", "apikey", "access_key"} or any(word in key_lower for word in ("content", "prompt", "request", "password", "secret", "token", "api_key", "authorization")):
        return "[redacted]"
    if depth > 6:
        return "[depth limit]"
    if isinstance(value, dict):
        return {str(k)[:80]: _safe_audit(v, str(k), depth + 1) for k, v in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        return [_safe_audit(v, key, depth + 1) for v in value[:20]]
    if isinstance(value, str):
        return value[:500]
    if value is None or type(value) in (int, float, bool):
        return value
    return str(value)[:200]


class AgentExecutor:
    """A thread-safe execution boundary for a single StudyDesk workspace."""

    def __init__(self, data_dir: Path):
        if os.name != "posix":
            raise ExecutionError("StudyDesk requires Linux or macOS for protected file operations. On Windows, run it in WSL or Docker.")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace = self.data_dir / "workspace"
        if self.workspace.is_symlink():
            raise ExecutionError("The workspace must not be a symbolic link.")
        self.workspace.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "agent.sqlite3"
        if self.db_path.is_symlink():
            raise ExecutionError("The database must not be a symbolic link.")
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    due_date TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    details TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def log_event(self, event: str, details: dict) -> None:
        """Persist a sanitized audit entry; no API keys or file contents."""
        encoded = json.dumps(_safe_audit(details), ensure_ascii=False)
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO audit(timestamp,event,details) VALUES(?,?,?)",
                (_now(), str(event)[:100], encoded),
            )

    def recent_audit(self, limit: int = 50) -> list[dict]:
        """Return newest-first audit events, capped at 200 entries."""
        if type(limit) is not int:
            limit = 50
        limit = min(200, max(1, limit))
        with self._lock, self._connect() as connection:
            records = connection.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(record), "details": json.loads(record["details"])} for record in records]

    def snapshot(self) -> dict:
        """Return files, tasks, and recent audit history without adding events.

        Only supported nonhidden regular files are listed, recursively, up to
        200 entries and 16 directory levels. Symlinks are never followed.
        """
        with self._lock:
            files = []

            def walk(parts: tuple[str, ...], depth: int = 0) -> None:
                if len(files) >= 200 or depth > 16:
                    return
                descriptor = self._directory_fd(parts)
                try:
                    for name in sorted(os.listdir(descriptor)):
                        if len(files) >= 200:
                            break
                        if name.startswith(".") or name.lower() in BLOCKED_NAMES:
                            continue
                        path = str(PurePosixPath(*parts, name))
                        try:
                            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                        except OSError:
                            # Another process can remove a file during a refresh.
                            continue
                        if stat.S_ISLNK(info.st_mode):
                            continue
                        if stat.S_ISDIR(info.st_mode):
                            try:
                                self._parts(path, directory=True)
                                walk((*parts, name), depth + 1)
                            except (ValidationError, OSError):
                                continue
                        elif stat.S_ISREG(info.st_mode) and PurePosixPath(name).suffix.lower() in ALLOWED_EXTENSIONS:
                            try:
                                self._parts(path)
                            except ValidationError:
                                continue
                            files.append({"path": path, "size": info.st_size})
                finally:
                    os.close(descriptor)

            walk(())
            return {"files": files, "tasks": self._list_tasks("all"), "audit": self.recent_audit(50)}

    def _parts(self, raw: str, *, directory: bool = False) -> tuple[str, ...]:
        if not isinstance(raw, str) or not raw or len(raw) > 240 or "\x00" in raw:
            raise ValidationError("Paths must be nonempty strings of at most 240 characters.")
        if raw == "." and directory:
            return ()
        if "\\" in raw or ":" in raw or raw.startswith("/"):
            raise ValidationError("Use a relative path inside the workspace.")
        split = raw.split("/")
        if any(not part or part in {".", ".."} or part.startswith(".") for part in split):
            raise ValidationError("Hidden paths, empty path components, and traversal are blocked.")
        if any(part.lower() in BLOCKED_NAMES for part in split):
            raise ValidationError("Credential and secret files are not accessible.")
        if any(any(ord(char) < 32 for char in part) for part in split):
            raise ValidationError("Control characters are not allowed in paths.")
        current = self.workspace
        for part in split:
            current = current / part
            if current.is_symlink():
                raise ValidationError("Symbolic links are not accessible.")
        if not directory and PurePosixPath(raw).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValidationError("Only .txt, .md, .csv, and .json files are supported.")
        return tuple(split)

    def validate_plan(self, raw: object) -> dict:
        """Return a canonical plan without executing any operational action.

        Validation attempts and proposed actions are audited. Validation does
        not create files or tasks, delete data, or contact an external service.
        """
        try:
            result = self._validate(raw)
        except ValidationError as exc:
            self.log_event("validation_failed", {"error": str(exc)})
            raise
        for index, action in enumerate(result["actions"]):
            self.log_event("action_proposed", {"index": index, **action})
        return result

    def _validate(self, raw: object) -> dict:
        if not isinstance(raw, dict) or set(raw) != {"summary", "actions"}:
            raise ValidationError("A plan must contain only summary and actions.")
        summary = raw["summary"]
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 1000:
            raise ValidationError("The plan summary must contain 1-1000 characters.")
        actions = raw["actions"]
        if not isinstance(actions, list) or not 0 <= len(actions) <= MAX_ACTIONS:
            raise ValidationError(f"A plan must contain 0-{MAX_ACTIONS} actions.")
        canonical = []
        for index, action in enumerate(actions):
            if not isinstance(action, dict) or set(action) != {"tool", "args"}:
                raise ValidationError(f"Action {index + 1} must contain only tool and args.")
            tool, args = action["tool"], action["args"]
            if not isinstance(tool, str) or tool not in _tool_args:
                raise ValidationError(f"Action {index + 1} uses an unsupported tool.")
            schema = _tool_args[tool]
            if not isinstance(args, dict) or set(args) - set(schema["properties"]):
                raise ValidationError(f"Action {index + 1} has unknown arguments.")
            if set(schema.get("required", [])) - set(args):
                raise ValidationError(f"Action {index + 1} is missing a required argument.")
            args = dict(args)
            if "path" in args:
                self._parts(args["path"])
            if tool == "list_files":
                args.setdefault("directory", ".")
                self._parts(args["directory"], directory=True)
            if tool == "write_file":
                content = args["content"]
                try:
                    content_bytes = len(content.encode("utf-8")) if isinstance(content, str) else MAX_CONTENT_BYTES + 1
                except UnicodeError as exc:
                    raise ValidationError("File content must contain valid Unicode text.") from exc
                if not isinstance(content, str) or content_bytes > MAX_CONTENT_BYTES:
                    raise ValidationError(f"File contents must be text up to {MAX_CONTENT_BYTES} UTF-8 bytes.")
                args.setdefault("overwrite", False)
                if type(args["overwrite"]) is not bool:
                    raise ValidationError("overwrite must be true or false.")
            if tool == "create_task":
                title = args["title"]
                if not isinstance(title, str) or not title.strip() or len(title) > 240 or any(ord(c) < 32 for c in title):
                    raise ValidationError("Task titles must contain 1-240 printable characters.")
                args["title"] = title.strip()
                args.setdefault("priority", "medium")
                if not isinstance(args["priority"], str) or args["priority"] not in {"low", "medium", "high"}:
                    raise ValidationError("Task priority must be low, medium, or high.")
                args.setdefault("due_date", None)
                if args["due_date"] is not None:
                    due = args["due_date"]
                    if not isinstance(due, str) or len(due) != 10:
                        raise ValidationError("due_date must be YYYY-MM-DD or null.")
                    try:
                        if date.fromisoformat(due).isoformat() != due:
                            raise ValueError("Noncanonical date")
                    except ValueError as exc:
                        raise ValidationError("due_date must be a valid YYYY-MM-DD date.") from exc
            if tool in {"list_tasks", "export_tasks"}:
                args.setdefault("status", "all")
                if not isinstance(args["status"], str) or args["status"] not in STATUSES:
                    raise ValidationError("status must be all, pending, or completed.")
            if tool in {"complete_task", "delete_task"}:
                if type(args["task_id"]) is not int or not 1 <= args["task_id"] <= 2**63 - 1:
                    raise ValidationError("task_id must be a positive integer.")
            if tool == "export_tasks" and PurePosixPath(args["path"]).suffix.lower() != ".csv":
                raise ValidationError("Task exports require a .csv path.")
            canonical.append({"tool": tool, "args": args})
        return {"summary": summary.strip(), "actions": canonical}

    def execute_plan(self, plan: dict, confirm_destructive: bool = False) -> list[dict]:
        """Run a validated plan in order, stopping after the first failure.

        Multi-action plans are not transactions: successful earlier actions
        remain applied if a later action fails. Destructive plans are blocked
        as a whole until the caller supplies an explicit confirmation.
        """
        with self._lock:
            try:
                canonical = self._validate(plan)
            except ValidationError as exc:
                self.log_event("validation_failed", {"error": str(exc)})
                raise
            if requires_confirmation(canonical) and confirm_destructive is not True:
                self.log_event("confirmation_required", {"tools": [a["tool"] for a in canonical["actions"]]})
                raise ExecutionError("Confirm this plan before deleting data or overwriting a file.")
            results = []
            failed = False
            for index, action in enumerate(canonical["actions"]):
                tool, args = action["tool"], action["args"]
                if failed:
                    result = {"tool": tool, "status": "skipped", "message": "Skipped because a previous action failed."}
                else:
                    self.log_event("action_started", {"index": index, "tool": tool, "args": args})
                    try:
                        message, data = self._dispatch(tool, args)
                        result = {"tool": tool, "status": "success", "message": message, "data": data}
                    except (OSError, sqlite3.Error, ValidationError, ExecutionError, UnicodeError) as exc:
                        # Absolute host paths and SQL internals stay out of UI/logs.
                        if isinstance(exc, (ValidationError, ExecutionError)):
                            message = str(exc)
                        elif isinstance(exc, FileNotFoundError):
                            message = "The requested file or directory does not exist."
                        elif isinstance(exc, FileExistsError):
                            message = "The output file already exists; choose a new name or explicitly request overwrite."
                        elif isinstance(exc, UnicodeError):
                            message = "The file is not valid UTF-8 text."
                        elif isinstance(exc, PermissionError):
                            message = "Permission denied for this operation."
                        elif isinstance(exc, sqlite3.Error):
                            message = "The task database could not complete the operation."
                        else:
                            message = "The file operation could not be completed safely."
                        result = {"tool": tool, "status": "error", "message": message}
                        failed = True
                self.log_event("action_result", {"index": index, **{k: v for k, v in result.items() if k != "data"}})
                results.append(result)
            return results

    def _directory_fd(self, parts: tuple[str, ...], create: bool = False) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.workspace, flags)
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _write_text(self, path: str, content: str, overwrite: bool = False) -> None:
        parts = self._parts(path)
        descriptor = self._directory_fd(parts[:-1], create=True)
        temporary = ".write-" + secrets.token_hex(12)
        try:
            try:
                existing = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(existing.st_mode):
                    raise ExecutionError("The output must be a regular file, not a directory or link.")
                if not overwrite:
                    raise FileExistsError(path)
            except FileNotFoundError:
                pass
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            file_descriptor = os.open(temporary, flags, 0o600, dir_fd=descriptor)
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, parts[-1], src_dir_fd=descriptor, dst_dir_fd=descriptor)
            else:
                # Hard-link creation is atomic and cannot silently overwrite.
                os.link(temporary, parts[-1], src_dir_fd=descriptor, dst_dir_fd=descriptor, follow_symlinks=False)
        finally:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except FileNotFoundError:
                pass
            os.close(descriptor)

    def _read_text(self, path: str) -> str:
        parts = self._parts(path)
        parent = self._directory_fd(parts[:-1])
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), dir_fd=parent)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ExecutionError("Only regular text files may be read.")
                raw = stream.read(MAX_CONTENT_BYTES + 1)
                if len(raw) > MAX_CONTENT_BYTES:
                    raise ExecutionError(f"This file exceeds the {MAX_CONTENT_BYTES}-byte reading limit.")
                return raw.decode("utf-8")
        finally:
            os.close(parent)

    def _list_tasks(self, status: str) -> list[dict]:
        query = "SELECT * FROM tasks"
        parameters = ()
        if status != "all":
            query += " WHERE status=?"
            parameters = (status,)
        query += " ORDER BY id"
        with self._connect() as connection:
            return [dict(record) for record in connection.execute(query, parameters).fetchall()]

    def _dispatch(self, tool: str, args: dict) -> tuple[str, Any]:
        if tool == "list_files":
            parts = self._parts(args["directory"], directory=True)
            descriptor = self._directory_fd(parts)
            try:
                entries = []
                for name in sorted(os.listdir(descriptor)):
                    if name.startswith(".") or name.lower() in BLOCKED_NAMES:
                        continue
                    info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        continue
                    entries.append({
                        "path": str(PurePosixPath(*parts, name)),
                        "type": "directory" if stat.S_ISDIR(info.st_mode) else "file",
                        "size_bytes": info.st_size,
                    })
                    if len(entries) >= 500:
                        break
                return f"Listed {len(entries)} workspace entries (maximum 500).", {"entries": entries}
            finally:
                os.close(descriptor)
        if tool == "read_file":
            content = self._read_text(args["path"])
            return f"Read {args['path']}.", {"path": args["path"], "content": content}
        if tool == "write_file":
            self._write_text(args["path"], args["content"], args["overwrite"])
            return f"Saved {args['path']}.", {"path": args["path"], "bytes": len(args["content"].encode("utf-8"))}
        if tool == "delete_file":
            parts = self._parts(args["path"])
            descriptor = self._directory_fd(parts[:-1])
            try:
                info = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise ExecutionError("Only regular text files may be deleted.")
                os.unlink(parts[-1], dir_fd=descriptor)
            finally:
                os.close(descriptor)
            return f"Deleted {args['path']}.", {"path": args["path"]}
        if tool == "create_task":
            with self._connect() as connection:
                cursor = connection.execute(
                    "INSERT INTO tasks(title,priority,due_date,created_at) VALUES(?,?,?,?)",
                    (args["title"], args["priority"], args["due_date"], _now()),
                )
                task = dict(connection.execute("SELECT * FROM tasks WHERE id=?", (cursor.lastrowid,)).fetchone())
            return f"Created task #{task['id']}: {task['title']}.", task
        if tool == "list_tasks":
            tasks = self._list_tasks(args["status"])
            return f"Found {len(tasks)} {args['status']} tasks.", {"tasks": tasks}
        if tool in {"complete_task", "delete_task"}:
            task_id = args["task_id"]
            with self._connect() as connection:
                if connection.execute("SELECT id FROM tasks WHERE id=?", (task_id,)).fetchone() is None:
                    raise ExecutionError(f"Task #{task_id} does not exist.")
                if tool == "complete_task":
                    connection.execute("UPDATE tasks SET status='completed',completed_at=? WHERE id=?", (_now(), task_id))
                else:
                    connection.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            verb = "Completed" if tool == "complete_task" else "Deleted"
            return f"{verb} task #{task_id}.", {"task_id": task_id}
        if tool == "export_tasks":
            tasks = self._list_tasks(args["status"])
            output = io.StringIO(newline="")
            columns = ["id", "title", "priority", "due_date", "status", "created_at", "completed_at"]
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for task in tasks:
                # Avoid spreadsheet formula injection in CSV text cells.
                safe = {key: ("'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value) for key, value in task.items()}
                writer.writerow(safe)
            content = output.getvalue()
            if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
                raise ExecutionError("The export exceeds the 50,000-byte limit; export a smaller status group.")
            self._write_text(args["path"], content, overwrite=False)
            return f"Exported {len(tasks)} tasks to {args['path']}.", {"path": args["path"], "task_count": len(tasks)}
        raise ExecutionError("Unsupported tool.")
