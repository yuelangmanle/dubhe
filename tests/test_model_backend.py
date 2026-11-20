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

    p = argparse.ArgumentParser()
    for arg in ("host", "port", "model", "alias", "api-key-file", "ctx-size", "threads", "parallel"):
        p.add_argument("--" + arg, required=True)
    a, _ = p.parse_known_args()
    assert a.host == "127.0.0.1" and a.parallel == "1"
    assert not os.environ.get("LLAMA_ARG_AGENT")
    key = Path(a.api_key_file).read_text().strip()
    assert key and not Path(a.api_key_file).stat().st_mode & 0o077
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
            assert body["model"] == a.alias and body["seed"] == 42
            assert body["messages"][-1]["content"] == "Hello"
            assert body["max_tokens"] == 2 and body["temperature"] == 0 and body["top_k"] == 8
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in (
                {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
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
