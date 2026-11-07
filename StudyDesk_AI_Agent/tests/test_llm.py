"""Network-free tests exercise real provider request/response boundaries."""

import io
import json
import socket
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from studydesk.llm import LLMClient, LLMError, MAX_RESPONSE_BYTES, _RejectRedirects, _retry_delay


PLAN = {"summary": "List tasks", "actions": [{"tool": "list_tasks", "args": {}}]}


def response(value):
    result = Mock()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=None)
    result.read.return_value = json.dumps(value).encode()
    return result


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.opener = Mock()
        self.patcher = patch("studydesk.llm.urlrequest.build_opener", return_value=self.opener)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def ollama_response(self, text=None):
        return response({"message": {"content": text if text is not None else json.dumps(PLAN)}, "done": True})

    def test_local_request_contract_and_parsing(self):
        self.opener.open.return_value = self.ollama_response()
        client = LLMClient()
        self.assertEqual(client.generate_plan("List tasks", {}), PLAN)
        req = self.opener.open.call_args.args[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(body["format"], "json")
        self.assertFalse(body["stream"])
        self.assertIsNone(req.get_header("Authorization"))

    def test_cloud_does_not_send_unsupported_format_and_keeps_key_private(self):
        self.opener.open.return_value = self.ollama_response()
        client = LLMClient("ollama_cloud", api_key="secret-value")
        self.assertEqual(client.generate_plan("List tasks", {}), PLAN)
        req = self.opener.open.call_args.args[0]
        self.assertEqual(req.get_header("Authorization"), "Bearer secret-value")
        self.assertNotIn("format", json.loads(req.data))
        self.assertNotIn("secret-value", json.dumps(client.public_info()))

    def test_gemini_request_and_response(self):
        self.opener.open.return_value = response({"candidates": [{"finishReason": "STOP", "content": {
            "parts": [{"text": json.dumps(PLAN)}]}}]})
        self.assertEqual(LLMClient("gemini", api_key="key").generate_plan("List tasks", {}), PLAN)
        req = self.opener.open.call_args.args[0]
        body = json.loads(req.data)
        self.assertTrue(req.full_url.endswith("gemini-2.5-flash:generateContent"))
        self.assertEqual(req.get_header("X-goog-api-key"), "key")
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")

    def test_whole_fenced_json_is_accepted(self):
        self.opener.open.return_value = self.ollama_response("```json\n" + json.dumps(PLAN) + "\n```")
        self.assertEqual(LLMClient().generate_plan("List tasks", {}), PLAN)

    def test_prose_arrays_duplicate_keys_and_malformed_json_are_rejected(self):
        for text in ["Here is a plan: " + json.dumps(PLAN), "[]", "{broken}",
                     '{"summary":"a","summary":"b","actions":[]}', "```json\n{}\n```\nmore", "\ud800"]:
            with self.subTest(text=text):
                self.opener.open.return_value = self.ollama_response(text)
                with self.assertRaises(LLMError):
                    LLMClient().generate_plan("List tasks", {})

    @patch("studydesk.llm.time.sleep")
    def test_transient_rate_limit_retries_then_succeeds(self, sleep):
        err = HTTPError("https://ollama.com/api/chat", 429, "Too many", {"Retry-After": "100"}, io.BytesIO(b"secret"))
        self.opener.open.side_effect = [err, self.ollama_response()]
        self.assertEqual(LLMClient().generate_plan("List tasks", {}), PLAN)
        self.assertEqual(self.opener.open.call_count, 2)
        sleep.assert_called_once_with(5.0)

    @patch("studydesk.llm.time.sleep")
    def test_three_total_attempts_and_secret_error_body_not_returned(self, sleep):
        self.opener.open.side_effect = [HTTPError("https://example", 503, "secret", {}, io.BytesIO(b"key-secret")) for _ in range(3)]
        with self.assertRaises(LLMError) as caught:
            LLMClient().generate_plan("List tasks", {})
        self.assertEqual(self.opener.open.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn("secret", str(caught.exception))

    def test_authentication_failure_does_not_retry(self):
        self.opener.open.side_effect = HTTPError("https://example", 401, "secret", {}, io.BytesIO(b"key-secret"))
        with self.assertRaisesRegex(LLMError, "access was denied"):
            LLMClient("gemini", api_key="key-secret").generate_plan("List tasks", {})
        self.assertEqual(self.opener.open.call_count, 1)

    def test_missing_key_stops_before_network(self):
        with self.assertRaisesRegex(LLMError, "API key is required"):
            LLMClient("gemini").generate_plan("List tasks", {})
        self.opener.open.assert_not_called()

    def test_oversized_api_response_is_rejected(self):
        raw = self.ollama_response()
        raw.read.return_value = b"x" * (MAX_RESPONSE_BYTES + 1)
        self.opener.open.return_value = raw
        with self.assertRaisesRegex(LLMError, "too large"):
            LLMClient().generate_plan("List tasks", {})
        raw.read.assert_called_once_with(MAX_RESPONSE_BYTES + 1)

    def test_connection_timeout_and_missing_model_are_actionable(self):
        for err, expected in [(URLError("private-details"), "Start Ollama"),
                              (socket.timeout(), "timed out"),
                              (HTTPError("https://example", 404, "private", {}, io.BytesIO()), "not found")]:
            with self.subTest(error=type(err)):
                self.opener.open.side_effect = err
                with self.assertRaisesRegex(LLMError, expected):
                    LLMClient().generate_plan("List tasks", {})

    def test_redirect_handler_never_follows_location(self):
        self.assertIsNone(_RejectRedirects().redirect_request(None, None, 302, "redirect", {}, "https://evil.invalid"))

    def test_retry_after_values_are_bounded(self):
        self.assertEqual(_retry_delay("999", 0), 5.0)
        self.assertEqual(_retry_delay("-9", 0), 0.0)
        self.assertEqual(_retry_delay("not a date", 1), 2.0)
        self.assertEqual(_retry_delay("nan", 1), 2.0)


if __name__ == "__main__":
    unittest.main()
