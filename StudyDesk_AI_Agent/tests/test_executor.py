"""Functional and safety tests for the real file and SQLite executor.

These tests use temporary directories; they never call an LLM or touch user data.
Run from the project root: python -m unittest discover -s tests -v
"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from studydesk.executor import AgentExecutor, ValidationError, requires_confirmation


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name) / "agent-data"
        self.agent = AgentExecutor(self.data_dir)

    def plan(self, *actions):
        return self.agent.validate_plan({"summary": "Test request", "actions": list(actions)})

    def action(self, tool, **args):
        return {"tool": tool, "args": args}

    def run_actions(self, *actions, confirmed=False):
        return self.agent.execute_plan(self.plan(*actions), confirm_destructive=confirmed)

    def assert_all_success(self, results):
        self.assertTrue(results, "The executor returned no action results")
        self.assertTrue(all(result["status"] == "success" for result in results), results)

    def assert_refused(self, plan):
        """A denied action may be surfaced as a result or a documented exception."""
        try:
            results = self.agent.execute_plan(plan, confirm_destructive=False)
        except (ValueError, RuntimeError):
            return
        self.assertTrue(any(item["status"] != "success" for item in results), results)

    def test_creates_and_reads_a_real_file(self):
        results = self.run_actions(
            self.action("write_file", path="notes/homework.md", content="Review agents.\n"),
            self.action("read_file", path="notes/homework.md"),
        )
        self.assert_all_success(results)
        self.assertEqual((self.agent.workspace / "notes/homework.md").read_text(), "Review agents.\n")
        self.assertIn("Review agents.", json.dumps(results[-1]))
        self.assertIn("notes/homework.md", [item["path"] for item in self.agent.snapshot()["files"]])

    def test_delete_requires_confirmation_and_then_deletes(self):
        self.assert_all_success(self.run_actions(self.action("write_file", path="scratch.txt", content="temp")))
        plan = self.plan(self.action("delete_file", path="scratch.txt"))
        self.assertTrue(requires_confirmation(plan))
        self.assert_refused(plan)
        self.assertTrue((self.agent.workspace / "scratch.txt").exists())
        self.assert_all_success(self.agent.execute_plan(plan, confirm_destructive=True))
        self.assertFalse((self.agent.workspace / "scratch.txt").exists())

    def test_existing_content_cannot_be_silently_overwritten(self):
        self.run_actions(self.action("write_file", path="notes.txt", content="Original"))
        results = self.run_actions(self.action("write_file", path="notes.txt", content="Replacement"))
        self.assertNotEqual(results[0]["status"], "success")
        self.assertEqual((self.agent.workspace / "notes.txt").read_text(), "Original")
        plan = self.plan(self.action("write_file", path="notes.txt", content="Replacement", overwrite=True))
        self.assertTrue(requires_confirmation(plan))
        self.assert_refused(plan)
        self.assertEqual((self.agent.workspace / "notes.txt").read_text(), "Original")
        self.assert_all_success(self.agent.execute_plan(plan, confirm_destructive=True))
        self.assertEqual((self.agent.workspace / "notes.txt").read_text(), "Replacement")

    def test_rejects_paths_outside_workspace_and_hidden_files(self):
        for path in ("../outside.txt", "/tmp/outside.txt", "nested/../../outside.txt", ".env", "notes/.secret.txt"):
            with self.subTest(path=path), self.assertRaises(ValidationError):
                self.plan(self.action("write_file", path=path, content="Forbidden"))
        self.assertFalse((self.data_dir / "outside.txt").exists())

    def test_rejects_symlink_escape(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        link = self.agent.workspace / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks are not available on this platform")
        try:
            plan = self.plan(self.action("write_file", path="escape/secret.txt", content="Forbidden"))
        except ValidationError:
            pass
        else:
            results = self.agent.execute_plan(plan)
            self.assertNotEqual(results[0]["status"], "success")
        self.assertFalse((outside / "secret.txt").exists())

    def test_rejects_unknown_commands_fields_types_and_large_plans(self):
        invalid = [
            {"summary": "Bad", "actions": [{"tool": "run_shell", "args": {"command": "ls"}}]},
            {"summary": "Bad", "actions": [], "execute": True},
            {"summary": "Bad", "actions": [{"tool": "list_tasks", "args": {"unrecognized": True}}]},
            {"summary": "Bad", "actions": [{"tool": "complete_task", "args": {"task_id": True}}]},
            {"summary": "Bad", "actions": [{"tool": "write_file", "args": {"path": "a.txt", "content": 3}}]},
            {"summary": "Bad", "actions": [{"tool": "write_file", "args": {"path": "a.py", "content": "print(1)"}}]},
            {"summary": "Bad", "actions": [self.action("list_tasks") for _ in range(9)]},
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                self.agent.validate_plan(raw)

    def test_task_database_persists_and_sql_text_is_data(self):
        title = "Read SQL'); DROP TABLE tasks;--"
        self.assert_all_success(self.run_actions(self.action("create_task", title=title, priority="high", due_date="2027-01-15")))
        reopened = AgentExecutor(self.data_dir)
        tasks = reopened.snapshot()["tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["title"], title)
        self.assertEqual(tasks[0]["priority"], "high")
        self.assertEqual(tasks[0]["due_date"], "2027-01-15")
        task_id = tasks[0]["id"]
        self.assert_all_success(self.run_actions(self.action("complete_task", task_id=task_id)))
        self.assertEqual(reopened.snapshot()["tasks"][0]["status"], "completed")
        deletion = self.plan(self.action("delete_task", task_id=task_id))
        self.assertTrue(requires_confirmation(deletion))
        self.assert_refused(deletion)
        self.assertEqual(len(reopened.snapshot()["tasks"]), 1)
        self.assert_all_success(self.agent.execute_plan(deletion, confirm_destructive=True))
        self.assertEqual(reopened.snapshot()["tasks"], [])

    def test_invalid_due_date_and_priority_are_rejected(self):
        for args in ({"title": "Read", "due_date": "2027-02-30"}, {"title": "Read", "priority": "urgent"}):
            with self.subTest(args=args), self.assertRaises(ValidationError):
                self.plan({"tool": "create_task", "args": args})

    def test_failed_action_skips_remaining_writes(self):
        results = self.run_actions(
            self.action("read_file", path="missing.txt"),
            self.action("write_file", path="should-not-exist.txt", content="No"),
        )
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[1]["status"], "skipped")
        self.assertFalse((self.agent.workspace / "should-not-exist.txt").exists())

    def test_csv_export_contains_actual_tasks_and_never_overwrites(self):
        title = "Read notes, then revise"
        self.run_actions(self.action("create_task", title=title, priority="medium"))
        self.assert_all_success(self.run_actions(self.action("export_tasks", path="exports/tasks.csv", status="all")))
        csv_path = self.agent.workspace / "exports/tasks.csv"
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], title)
        original = csv_path.read_bytes()
        results = self.run_actions(self.action("export_tasks", path="exports/tasks.csv", status="all"))
        self.assertNotEqual(results[0]["status"], "success")
        self.assertEqual(csv_path.read_bytes(), original)

    def test_audit_records_actions_without_file_contents(self):
        private_content = "PRIVATE_NOTE_CONTENT_5ed41d9cb604"
        self.run_actions(
            self.action("write_file", path="private.txt", content=private_content),
            self.action("read_file", path="private.txt"),
        )
        audit = self.agent.recent_audit()
        self.assertTrue(audit)
        serialized = json.dumps(audit)
        self.assertIn("write_file", serialized)
        self.assertIn("read_file", serialized)
        self.assertNotIn(private_content, serialized)

    def test_empty_clarification_plan_does_not_mutate_files_or_tasks(self):
        self.run_actions(
            self.action("write_file", path="existing.txt", content="Keep this note"),
            self.action("create_task", title="Keep this task", priority="low"),
        )
        before = self.agent.snapshot()
        plan = self.agent.validate_plan({
            "summary": "Which task should I complete? Please provide its ID.",
            "actions": [],
        })
        self.assertFalse(requires_confirmation(plan))
        self.assertEqual(self.agent.execute_plan(plan), [])
        after = self.agent.snapshot()
        self.assertEqual(after["files"], before["files"])
        self.assertEqual(after["tasks"], before["tasks"])
        self.assertEqual((self.agent.workspace / "existing.txt").read_text(), "Keep this note")


if __name__ == "__main__":
    unittest.main()
