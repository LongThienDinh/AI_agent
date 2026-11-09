from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, datetime
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, quote, urlsplit

from studydesk.executor import AgentExecutor, ValidationError, requires_confirmation
from studydesk.llm import LLMClient, LLMError

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "ollama_local": "llama3.2:3b",
    "ollama_cloud": "gemma4:31b",
    "gemini": "gemini-2.5-flash",
}


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


class APIError(Exception):
    def __init__(self, message: str, status: int = 400):
        self.message, self.status = message, status
        super().__init__(message)


class AgentApplication:
    def __init__(self, data_dir: Path, password: str = "", secure_cookie: bool = False):
        self.executor = AgentExecutor(data_dir)
        self.password = password
        self.secure_cookie = secure_cookie
        self.lock = threading.RLock()
        self.execution_lock = threading.Lock()
        self.planning_slots = threading.BoundedSemaphore(2)
        self.csrf = secrets.token_urlsafe(32)
        self.sessions: dict[str, float] = {}
        self.plans: dict[str, dict] = {}
        self.requests: dict[str, deque] = defaultdict(deque)
        self.keys = {
            "ollama_local": "",
            "ollama_cloud": os.getenv("OLLAMA_API_KEY", ""),
            "gemini": os.getenv("GEMINI_API_KEY", ""),
        }
        self.provider = os.getenv("LLM_PROVIDER", "ollama_local").strip()
        if self.provider not in DEFAULT_MODELS:
            raise ValueError("LLM_PROVIDER must be ollama_local, ollama_cloud, or gemini.")
        self.model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[self.provider]

    def client(self) -> LLMClient:
        with self.lock:
            return LLMClient(provider=self.provider, model=self.model,
                             api_key=self.keys[self.provider])

    def status(self) -> dict:
        with self.lock:
            return {"app": "StudyDesk", "provider": self.provider, "model": self.model,
                    "configured": self.provider == "ollama_local" or bool(self.keys[self.provider]),
                    "csrf_token": self.csrf, "password_required": bool(self.password)}

    def throttle(self, key: str, maximum: int = 20, window: int = 60) -> None:
        with self.lock:
            now = time.monotonic()
            for stale in [k for k, v in self.requests.items() if not v or v[-1] < now - 300]:
                del self.requests[stale]
            events = self.requests[key]
            while events and events[0] < now - window:
                events.popleft()
            if len(events) >= maximum:
                raise APIError("Too many requests. Wait a minute and try again.", 429)
            events.append(now)

    def owner(self, cookie_header: str) -> str | None:
        if not self.password:
            return "local"
        cookies = SimpleCookie()
        try:
            cookies.load(cookie_header)
            token = cookies.get("studydesk_session")
            value = token.value if token else ""
        except CookieError:
            return None
        with self.lock:
            now = time.monotonic()
            for expired in [key for key, expiry in self.sessions.items() if expiry < now]:
                del self.sessions[expired]
            return value if value in self.sessions else None

    def configure(self, body: dict) -> dict:
        if set(body) - {"provider", "model", "api_key"}:
            raise APIError("Unknown settings field.")
        provider = body.get("provider")
        model, key = body.get("model", ""), body.get("api_key", "")
        if not isinstance(provider, str) or provider not in DEFAULT_MODELS:
            raise APIError("Choose Ollama local, Ollama cloud, or Gemini.")
        if not isinstance(model, str) or len(model) > 120 or any(ord(c) < 32 for c in model):
            raise APIError("Model name must be plain text up to 120 characters.")
        if not isinstance(key, str) or len(key) > 1024 or not key.isascii() or any(c.isspace() for c in key):
            raise APIError("API key must not contain whitespace.")
        model = model.strip() or DEFAULT_MODELS[provider]
        LLMClient(provider=provider, model=model, api_key=key)
        with self.lock:
            if key:
                self.keys[provider] = key
            self.provider, self.model = provider, model
            self.plans.clear()
        self.executor.log_event("settings_updated", {"provider": provider, "model": model})
        return self.status()

    def plan(self, body: dict, owner: str) -> dict:
        request = body.get("request")
        if set(body) - {"request", "current_date", "timezone"} or not isinstance(request, str) or not 1 <= len(request.strip()) <= 6000:
            raise APIError("Enter a request between 1 and 6,000 characters.")
        current_date = body.get("current_date", datetime.now().astimezone().date().isoformat())
        timezone = body.get("timezone", datetime.now().astimezone().tzname() or "server local time")
        try:
            if not isinstance(current_date, str) or date.fromisoformat(current_date).isoformat() != current_date:
                raise ValueError
        except ValueError:
            raise APIError("The request date must use YYYY-MM-DD.") from None
        if not isinstance(timezone, str) or not 1 <= len(timezone) <= 80 or any(ord(c) < 32 for c in timezone):
            raise APIError("Invalid timezone label.")
        self.throttle("plan:" + owner)
        if not self.planning_slots.acquire(blocking=False):
            raise APIError("The agent is handling another request. Please try again shortly.", 429)
        try:
            state = self.executor.snapshot()
            context = {"current_date": current_date, "timezone": timezone,
                       "files": state["files"][:100], "tasks": state["tasks"][:100]}
            self.executor.log_event("planning_started", {"request_length": len(request)})
            raw = self.client().generate_plan(request.strip(), context)
            plan = self.executor.validate_plan(raw)
        except LLMError:
            self.executor.log_event("planning_failed", {"reason": "llm_api_error"})
            raise
        finally:
            self.planning_slots.release()
        plan_id = secrets.token_urlsafe(24)
        with self.lock:
            self._expire_plans()
            if len(self.plans) >= 100:
                self.plans.pop(next(iter(self.plans)))
            self.plans[plan_id] = {"plan": plan, "owner": owner,
                                   "expires": time.monotonic() + 600}
        return {"plan_id": plan_id, "plan": plan, "requires_confirmation": requires_confirmation(plan),
                "expires_in": 600}

    def _expire_plans(self) -> None:
        now = time.monotonic()
        for plan_id in [key for key, value in self.plans.items() if value["expires"] < now]:
            del self.plans[plan_id]

    def execute(self, body: dict, owner: str) -> dict:
        if set(body) - {"plan_id", "confirm_destructive"}:
            raise APIError("Execution accepts only the saved plan ID and confirmation.")
        plan_id, confirm = body.get("plan_id"), body.get("confirm_destructive", False)
        if not isinstance(plan_id, str) or type(confirm) is not bool:
            raise APIError("Invalid plan or confirmation.")
        with self.execution_lock:
            with self.lock:
                self._expire_plans()
                saved = self.plans.get(plan_id)
                if not saved or saved["owner"] != owner:
                    raise APIError("This plan expired or was already used. Generate a new plan.", 409)
                if requires_confirmation(saved["plan"]) and not confirm:
                    raise APIError("Confirm deletion or overwriting before executing this plan.", 409)
                self.plans.pop(plan_id)
            results = self.executor.execute_plan(saved["plan"], confirm_destructive=confirm)
        return {"results": results}

    def cancel(self, body: dict, owner: str) -> dict:
        plan_id = body.get("plan_id")
        if set(body) != {"plan_id"} or not isinstance(plan_id, str):
            raise APIError("A plan ID is required.")
        with self.lock:
            saved = self.plans.get(plan_id)
            if saved and saved["owner"] == owner:
                self.plans.pop(plan_id)
        self.executor.log_event("plan_cancelled", {})
        return {"ok": True}


class AgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 20

    def __init__(self, address, application: AgentApplication):
        self.application = application
        super().__init__(address, RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "StudyDesk/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    @property
    def app(self) -> AgentApplication:
        return self.server.application

    def log_message(self, fmt, *args):
        print(f"HTTP {self.command} {urlsplit(self.path).path}", flush=True)

    def respond(self, payload, status=200, headers=None):
        if isinstance(payload, bytes):
            body = payload
            mime = "text/html; charset=utf-8"
        else:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            mime = "application/json; charset=utf-8"
        self.send_response(status)
        headers = dict(headers or {})
        self.send_header("Content-Type", headers.pop("Content-Type", mime))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
                         "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def check_host(self):
        bound_host = self.server.server_address[0]
        host = urlsplit("//" + self.headers.get("Host", "")).hostname
        if bound_host in {"127.0.0.1", "::1"} and host not in {"localhost", "127.0.0.1", "::1"}:
            raise APIError("Invalid host.", 403)

    def require_owner(self) -> str:
        owner = self.app.owner(self.headers.get("Cookie", ""))
        if owner is None:
            raise APIError("Enter the app password to continue.", 401)
        return owner

    def body(self) -> dict:
        if self.headers.get_content_type() != "application/json":
            raise APIError("Use an application/json request.", 415)
        if self.headers.get("Transfer-Encoding"):
            raise APIError("Chunked request bodies are not supported.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError("Invalid content length.") from None
        if not 0 < length <= 65536:
            raise APIError("Request body is missing or too large.", 413)
        try:
            data = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeError, RecursionError):
            raise APIError("Request body must contain valid JSON.") from None
        if not isinstance(data, dict):
            raise APIError("Request body must be a JSON object.")
        return data

    def do_GET(self):
        try:
            self.check_host()
            route = urlsplit(self.path)
            if route.path == "/health":
                return self.respond({"status": "ok"})
            if route.path == "/":
                return self.respond((ROOT / "static" / "index.html").read_bytes())
            self.require_owner()
            if route.path == "/api/status":
                return self.respond(self.app.status())
            if route.path == "/api/state":
                return self.respond(self.app.executor.snapshot())
            if route.path in {"/api/file", "/api/download"}:
                file_path = parse_qs(route.query).get("path", [""])[0]
                plan = self.app.executor.validate_plan({"summary": "Read a workspace file",
                           "actions": [{"tool": "read_file", "args": {"path": file_path}}]})
                result = self.app.executor.execute_plan(plan)[0]
                if result["status"] != "success":
                    raise APIError(result["message"], 404)
                data = result.get("data", {})
                content = data.get("content", "") if isinstance(data, dict) else str(data)
                if route.path == "/api/file":
                    return self.respond({"path": file_path, "content": content})
                return self.respond(content.encode("utf-8"), headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": "attachment; filename*=UTF-8''" + quote(Path(file_path).name)})
            raise APIError("Page not found.", 404)
        except Exception as error:
            self.handle_error(error)

    def do_POST(self):
        try:
            self.check_host()
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise APIError("Cross-origin requests are not allowed.", 403)
            route = urlsplit(self.path).path
            body = self.body()
            if route == "/api/login":
                self.app.throttle("login:" + self.client_address[0], maximum=8, window=60)
                password = body.get("password", "")
                if not isinstance(password, str) or len(password) > 1024:
                    raise APIError("Invalid password.", 401)
                if not secrets.compare_digest(password.encode(), self.app.password.encode()):
                    raise APIError("Incorrect app password.", 401)
                token = secrets.token_urlsafe(32)
                with self.app.lock:
                    now = time.monotonic()
                    self.app.sessions = {k: v for k, v in self.app.sessions.items() if v > now}
                    if len(self.app.sessions) >= 1000:
                        raise APIError("Too many active sessions. Try again later.", 429)
                    self.app.sessions[token] = now + 43200
                cookie = f"studydesk_session={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=43200"
                if self.app.secure_cookie:
                    cookie += "; Secure"
                return self.respond({"ok": True}, headers={"Set-Cookie": cookie})
            owner = self.require_owner()
            csrf = self.headers.get("X-CSRF-Token", "")
            if not secrets.compare_digest(csrf.encode(), self.app.csrf.encode()):
                raise APIError("Session check failed. Refresh the page and try again.", 403)
            if route == "/api/config":
                return self.respond(self.app.configure(body))
            if route == "/api/plan":
                return self.respond(self.app.plan(body, owner))
            if route == "/api/execute":
                return self.respond(self.app.execute(body, owner))
            if route == "/api/cancel":
                return self.respond(self.app.cancel(body, owner))
            raise APIError("Endpoint not found.", 404)
        except Exception as error:
            self.handle_error(error)

    def handle_error(self, error: Exception):
        if isinstance(error, APIError):
            payload = {"error": error.message}
            if error.status == 401:
                payload["password_required"] = True
            return self.respond(payload, error.status)
        if isinstance(error, ValidationError):
            return self.respond({"error": "The proposed plan was rejected: " + str(error)}, 422)
        if isinstance(error, LLMError):
            return self.respond({"error": str(error)}, 502)
        if isinstance(error, (BrokenPipeError, ConnectionResetError, TimeoutError)):
            return
        self.app.executor.log_event("server_error", {"error_type": type(error).__name__})
        self.respond({"error": "The operation failed. Check the activity log and try again."}, 500)


def main():
    load_env(ROOT / ".env")
    host = os.getenv("HOST", "127.0.0.1")
    password = os.getenv("APP_PASSWORD", "")
    try:
        port = int(os.getenv("PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("PORT must be an integer from 1 to 65535.") from None
    data_dir = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
    try:
        app = AgentApplication(data_dir, password, os.getenv("COOKIE_SECURE", "false").lower() == "true")
        server = AgentHTTPServer((host, port), app)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not start StudyDesk: {error}") from None
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"StudyDesk is ready at http://{display_host}:{port}", flush=True)
    print("Press Ctrl+C to stop. Configure a model through Settings in the app.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStudyDesk stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
