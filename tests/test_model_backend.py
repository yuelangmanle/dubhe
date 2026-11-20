"""Protocol and subprocess lifecycle regression for the optional model worker."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "tools" / "model_backend.py"

# A minimal counterpart exposes the same contract without dependencies or weights.
FAKE_SERVER = """
    import argparse
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json
    import os
    from pathlib import Path
    import signal
    import time

    p = argparse.ArgumentParser()
    for arg in ("host", "port", "model", "alias", "api-key-file", "ctx-size", "threads", "parallel"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--mmproj")
    a, _ = p.parse_known_args()
    assert a.host == "127.0.0.1" and a.parallel == "1"
    assert not os.environ.get("LLAMA_ARG_AGENT")
    key = Path(a.api_key_file).read_text().strip()
    assert key and not Path(a.api_key_file).stat().st_mode & 0o077
    shutdown_log = os.environ.get("DUBHE_TEST_CHILD_SHUTDOWN")
    if shutdown_log:
        def stop(_signum, _frame):
            with Path(shutdown_log).open("a") as log:
                log.write("SIGTERM\\n")
            time.sleep(0.15)
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
    Path(os.environ["DUBHE_TEST_CHILD_PID"]).write_text(str(os.getpid()))
    print("Test llama-server startup diagnostic", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, code=200):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def authorized(self):
            if self.headers.get("Authorization") == "Bearer " + key:
                return True
            self.reply({"error": {"message": "unauthorized"}}, 401)
            return False

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/v1/models":
                self.reply({"data": [{"id": a.alias, "meta": {"n_params": 123}}]})
            elif self.path == "/props":
                self.reply({"default_generation_settings": {"n_ctx": int(a.ctx_size)}})
            else:
                self.reply({}, 404)

        def do_POST(self):
            if not self.authorized():
                return
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["stream"] is True and body["stream_options"]["include_usage"] is True
            assert body["model"] == a.alias and body["temperature"] == 0
            text, reason = "Hi", "length"
            content = body["messages"][-1]["content"]
            if a.mmproj:
                assert Path(a.mmproj).name == "Xiaomi-OCR-0.mmproj-bf16.gguf"
                assert a.ctx_size == "16384"
                image, instruction = content
                assert image["type"] == "image_url" and image["image_url"]["url"].startswith("data:image/png;base64,")
                assert instruction["type"] == "text"
                assert body["chat_template_kwargs"] == {"enable_thinking": False}
                prompt = instruction["text"]
                if "slow-test" in prompt:
                    Path(os.environ["DUBHE_TEST_SCAN_STARTED"]).touch()
                    time.sleep(30)
                if "Markdown" in prompt:
                    text = "<md>\\n# Report\\n\\n<fcel>Header<lcel><nl>\\n<fcel>01<fcel>02<nl>\\n\\nNotes\\n</md>"
                elif "OTSL" in prompt:
                    text = "<fcel><script>alert(1)</script><lcel><nl><ucel><xcel><nl>"
                elif "LaTeX" in prompt:
                    text = "x^2+y^2"
                elif "JSON" in prompt:
                    assert 'according to the following schema:' in prompt
                    assert 'leave it as an empty string' in prompt
                    fmt = body["response_format"]
                    assert fmt["type"] == "json_schema"
                    fields = fmt["json_schema"]["schema"]
                    assert fields == {"type": "object", "additionalProperties": False,
                        "required": ["日期", "地点"],
                        "properties": {key: {"type": "string"} for key in ("日期", "地点")}}
                    text = '{"日期":"11/20","地点":""}'
                else:
                    text = "Water level 10.2 m"
                reason = "stop"
            else:
                assert body["seed"] == 42 and content == "Hello"
                assert "response_format" not in body
                assert body["max_tokens"] == 2 and body["top_k"] == 8
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in (
                {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": reason}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            ):
                self.wfile.write(("data: " + json.dumps(event) + "\\r\\n\\r\\n").encode())
            self.wfile.write(b"data: [DONE]\\r\\n\\r\\n")

    HTTPServer((a.host, int(a.port)), Handler).serve_forever()
"""


class TestModelBackend(unittest.TestCase):
    def test_worker_authenticates_streams_and_reaps(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            server = directory / "llama-server"
            server.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_SERVER))
            server.chmod(0o700)
            model = directory / "fixture with whitespace.gguf"
            key, arch = b"general.architecture", b"gemma3"
            model.write_bytes(
                struct.pack("<4sIQQQ", b"GGUF", 3, 0, 1, len(key)) + key
                + struct.pack("<IQ", 8, len(arch)) + arch
            )
            child_pid = directory / "pid"
            env = dict(os.environ, DUBHE_LLAMA_SERVER=str(server), DUBHE_TEST_CHILD_PID=str(child_pid),
                       LLAMA_ARG_AGENT="true")
            request = {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
                "max_tokens": 2, "temperature": 0, "top_p": 0.9, "top_k": 8, "seed": 42,
                "ocr_fields": ["ignored for text"],
            }
            media = {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}
            proc = subprocess.run(
                [sys.executable, str(ADAPTER), "--backend", "llama", "--model", str(model),
                 "--ctx", "128", "--threads", "2"],
                input="\n".join(map(json.dumps, (media, request, request))) + "\n",
                env=env, text=True, capture_output=True, timeout=15,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("startup diagnostic", proc.stderr)
            events = list(map(json.loads, proc.stdout.splitlines()))
            self.assertEqual(events[0], {"ready": True, "backend": "llama.cpp", "arch": "gemma3",
                                        "context": 128, "params": 123})
            self.assertIn("error", events[1])
            answer = [{"delta": "Hi"}, {"done": True, "finish_reason": "length",
                                        "prompt_tokens": 5, "completion_tokens": 2}]
            self.assertEqual(events[2:], answer * 2)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(child_pid.read_text()), 0)


if __name__ == "__main__":
    unittest.main()
