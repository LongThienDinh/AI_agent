"""Real HTTP + SQLite/files integration tests with a stubbed model boundary.

These tests do not call a live LLM or pretend to verify its language reasoning.
"""
import copy
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from app import AgentApplication, AgentHTTPServer, RequestHandler, APIError


class FixtureLLM:
    """Only a test fixture; never available as a provider in the application."""
    def __init__(self):
        self.plan = {"summary": "Save a note and create a task.", "actions": [
            {"tool": "write_file", "args": {"path": "homework.md", "content": "Review agents."}},
            {"tool": "create_task", "args": {"title": "Finish AI homework", "priority": "high"}},
        ]}
        self.context = None

    def generate_plan(self, request, context):
        self.context = context
        return copy.deepcopy(self.plan)


class HTTPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama_local", "LLM_MODEL": "llama3.2:3b"}):
            self.app = AgentApplication(Path(self.temp.name))
        self.fixture = FixtureLLM()
        self.app.client = lambda: self.fixture
        self.server = AgentHTTPServer(("127.0.0.1", 0), self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.csrf = self.app.csrf
        self.cookie = ""

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, method, path, body=None, csrf=True, extra_headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body)
        if csrf:
            headers["X-CSRF-Token"] = self.csrf
        if self.cookie:
            headers["Cookie"] = self.cookie
        headers.update(extra_headers or {})
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        data = response.read()
        status, response_headers = response.status, dict(response.getheaders())
        conn.close()
        if response_headers.get("Content-Type", "").startswith("application/json"):
            data = json.loads(data)
        return status, data, response_headers

    def new_plan(self):
        code, body, _ = self.request("POST", "/api/plan", {"request": "Save a note and task"})
        self.assertEqual(code, 200, body)
        return body

    def test_preview_execute_real_effects_download_and_replay(self):
        preview = self.new_plan()
        self.assertFalse((self.app.executor.workspace / "homework.md").exists())
        self.assertEqual(self.app.executor.snapshot()["tasks"], [])
        code, body, _ = self.request("POST", "/api/execute", {"plan_id": preview["plan_id"]})
        self.assertEqual(code, 200, body)
        self.assertEqual([r["status"] for r in body["results"]], ["success", "success"])
        self.assertEqual((self.app.executor.workspace / "homework.md").read_text(), "Review agents.")
        self.assertEqual(self.app.executor.snapshot()["tasks"][0]["title"], "Finish AI homework")
        code, content, headers = self.request("GET", "/api/download?path=homework.md")
        self.assertEqual((code, content), (200, b"Review agents."))
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        code, data, _ = self.request("GET", "/api/file?path=homework.md")
        self.assertEqual((code, data["content"]), (200, "Review agents."))
        self.assertEqual(self.request("POST", "/api/execute", {"plan_id": preview["plan_id"]})[0], 409)

    def test_missing_csrf_and_cross_origin_requests_are_blocked(self):
        self.assertEqual(self.request("POST", "/api/plan", {"request": "hi"}, csrf=False)[0], 403)
        self.assertEqual(self.request("POST", "/api/plan", {"request": "hi"},
                                    extra_headers={"Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.request("GET", "/api/status", extra_headers={"Host": "other.example"})[0], 403)

    def test_client_cannot_replace_the_saved_plan(self):
        preview = self.new_plan()
        code, _, _ = self.request("POST", "/api/execute", {
            "plan_id": preview["plan_id"], "plan": {"actions": []}})
        self.assertEqual(code, 400)
        self.assertFalse((self.app.executor.workspace / "homework.md").exists())

    def test_confirmation_gate_preserves_plan_until_explicit_approval(self):
        (self.app.executor.workspace / "note.txt").write_text("Keep me")
        self.fixture.plan = {"summary": "Delete your note.", "actions": [
            {"tool": "delete_file", "args": {"path": "note.txt"}}]}
        preview = self.new_plan()
        self.assertTrue(preview["requires_confirmation"])
        self.assertEqual(self.request("POST", "/api/execute", {"plan_id": preview["plan_id"]})[0], 409)
        self.assertTrue((self.app.executor.workspace / "note.txt").exists())
        code, result, _ = self.request("POST", "/api/execute", {
            "plan_id": preview["plan_id"], "confirm_destructive": True})
        self.assertEqual(code, 200, result)
        self.assertFalse((self.app.executor.workspace / "note.txt").exists())

    def test_rejected_llm_plan_executes_nothing(self):
        self.fixture.plan["actions"].append({"tool": "shell", "args": {"command": "whoami"}})
        code, body, _ = self.request("POST", "/api/plan", {"request": "Do work"})
        self.assertEqual(code, 422, body)
        self.assertEqual(self.app.executor.snapshot()["files"], [])
        self.assertEqual(self.app.executor.snapshot()["tasks"], [])

    def test_clarification_cancel_expiry_and_owner(self):
        self.fixture.plan = {"summary": "Which note should I read?", "actions": []}
        preview = self.new_plan()
        self.assertEqual(preview["plan"]["actions"], [])
        self.request("POST", "/api/cancel", {"plan_id": preview["plan_id"]})
        self.assertNotIn(preview["plan_id"], self.app.plans)
        preview = self.new_plan()
        self.app.plans[preview["plan_id"]]["expires"] = time.monotonic() - 1
        self.assertEqual(self.request("POST", "/api/execute", {"plan_id": preview["plan_id"]})[0], 409)
        preview = self.new_plan()
        self.app.plans[preview["plan_id"]]["owner"] = "different-session"
        self.assertEqual(self.request("POST", "/api/execute", {"plan_id": preview["plan_id"]})[0], 409)

    def test_api_key_is_not_returned_and_blank_keeps_matching_key(self):
        code, body, _ = self.request("POST", "/api/config", {
            "provider": "ollama_cloud", "model": "gemma4:31b", "api_key": "test-only-secret"})
        self.assertEqual(code, 200, body)
        self.assertNotIn("test-only-secret", json.dumps(body))
        self.request("POST", "/api/config", {"provider": "gemini", "api_key": "other-secret"})
        self.request("POST", "/api/config", {"provider": "ollama_cloud", "api_key": ""})
        self.assertEqual(self.app.keys["ollama_cloud"], "test-only-secret")
        self.assertNotIn("test-only-secret", json.dumps(self.request("GET", "/api/state")[1]))
        self.assertEqual(self.request("POST", "/api/config", {"provider": [], "api_key": ""})[0], 400)

    def test_login_session_and_private_api_protection(self):
        self.app.password = "classroom-test-password"
        self.assertEqual(self.request("GET", "/health")[0], 200)
        self.assertEqual(self.request("GET", "/api/state")[0], 401)
        self.assertEqual(self.request("POST", "/api/login", {"password": "wrong"}, csrf=False)[0], 401)
        code, _, headers = self.request("POST", "/api/login", {"password": self.app.password}, csrf=False)
        self.assertEqual(code, 200)
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        self.cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.assertEqual(self.request("GET", "/api/state")[0], 200)

    def test_plan_rate_limiting_is_bounded(self):
        for _ in range(20):
            self.app.throttle("test")
        with self.assertRaises(APIError) as result:
            self.app.throttle("test")
        self.assertEqual(result.exception.status, 429)

    def test_browser_date_and_timezone_ground_relative_dates(self):
        code, body, _ = self.request("POST", "/api/plan", {
            "request": "Create a task for tomorrow", "current_date": "2026-11-03",
            "timezone": "America/Los_Angeles"})
        self.assertEqual(code, 200, body)
        self.assertEqual(self.fixture.context["current_date"], "2026-11-03")
        self.assertEqual(self.fixture.context["timezone"], "America/Los_Angeles")
        self.assertEqual(self.request("POST", "/api/plan", {
            "request": "Task", "current_date": "2026-02-30"})[0], 400)


if __name__ == "__main__":
    unittest.main()
