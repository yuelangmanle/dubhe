#!/usr/bin/env python3
"""Optional local model worker. Standard input and output use JSON lines."""

import argparse
from contextlib import closing
import http.client
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time


def chat_request(data, *, allow_image=False):
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list) or not data["messages"]:
        raise ValueError("messages 必须是非空列表。")
    messages = []
    for message in data["messages"]:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("此后端仅支持文本对话。")
        content = message.get("content")
        if isinstance(content, list):
            def is_text(part):
                return (isinstance(part, dict) and part.get("type") == "text"
                        and isinstance(part.get("text"), str))

            if all(is_text(part) for part in content):
                content = "".join(part["text"] for part in content)
            elif allow_image and message["role"] == "user" and len(content) == 2:
                image, text = content
                url = image.get("image_url") if isinstance(image, dict) else None
                url = url.get("url") if isinstance(url, dict) else None
                if (not isinstance(url, str) or image.get("type") != "image_url" or not is_text(text)
                        or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,"))):
                    raise ValueError("扫描只接受直接上传的一张 JPEG 或 PNG 图片。")
                content = [{"type": "image_url", "image_url": {"url": url}},
                           {"type": "text", "text": text["text"]}]
            else:
                raise ValueError("此后端仅支持文本消息。")
        if not isinstance(content, (str, list)) or message.get("tool_calls"):
            raise ValueError("消息格式不正确。")
        messages.append({"role": message["role"], "content": content})

    request = {"messages": messages}
    for name, default, floor, ceiling in (
        ("max_tokens", 256, 1, 2**31 - 1), ("top_k", 40, 0, 2**31 - 1),
        ("seed", -1, -1, 2**32 - 1), ("temperature", 0.7, 0, None), ("top_p", 0.95, 0, 1),
        ("repeat_penalty", 1.0, 0.01, 10),
    ):
        value = data.get(name, default)
        types = (int,) if type(default) is int else (int, float)
        if type(value) not in types or value < floor or (ceiling is not None and value > ceiling):
            raise ValueError(name + " 超出有效范围。")
        if not math.isfinite(value):
            raise ValueError(name + " 必须是有限数值。")
        request[name] = value
    if data.get("no_think") is True:
        request["chat_template_kwargs"] = {"enable_thinking": False}
    if allow_image and "ocr_fields" in data:
        fields = data["ocr_fields"]
        if (not isinstance(fields, list) or not 1 <= len(fields) <= 20
                or any(not isinstance(field, str) or not field.strip() or len(field.encode("utf-8")) > 96
                       for field in fields) or len(set(fields)) != len(fields)):
            raise ValueError("扫描字段列表格式不正确。")
        request["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "ocr_fields", "strict": True, "schema": {
                "type": "object", "required": fields, "additionalProperties": False,
                "properties": {field: {"type": "string"} for field in fields},
            },
        }}
    return request


def completion(reason, prompt_tokens, completion_tokens):
    if reason not in ("stop", "length") or any(
        type(n) is not int or n < 0 for n in (prompt_tokens, completion_tokens)
    ):
        raise RuntimeError("后端未返回完整的生成结果。")
    return dict(done=True, finish_reason=reason, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens)


class MLXModel:
    def __init__(self, path, context):
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("MLX 模型需要包含 config.json 的本地目录。")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import get_total_parameters
        except (ImportError, OSError) as e:
            raise RuntimeError("无法加载 MLX 运行环境。请用安装了 mlx-lm 的 Python 启动。") from e

        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") == "lfm3" and find_spec("mlx_lm.models.lfm3") is None:
            from mlx_lfm3 import register
            register()

        self.model, self.tokenizer, config = load(
            str(path), trust_remote_code=False,
            tokenizer_config={"trust_remote_code": False}, return_config=True,
        )
        trained_context = config.get("max_position_embeddings")
        if type(trained_context) is int and trained_context > 0:
            context = min(context, trained_context)
        self.info = dict(backend="MLX", arch=config.get("model_type") or "mlx",
                         context=context, params=int(get_total_parameters(self.model)))
        self.mx = mx
        self.stream_generate = stream_generate
        self.make_sampler = make_sampler

    def generate(self, request):
        ids = self.tokenizer.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True,
            return_dict=False, enable_thinking=False,
        )
        room = self.info["context"] - len(ids)
        if room <= 0:
            raise ValueError("输入已用完模型的上下文，请缩短对话。")
        if request["seed"] >= 0:
            self.mx.random.seed(request["seed"])
        sampler = self.make_sampler(temp=request["temperature"], top_p=request["top_p"],
                                    top_k=request["top_k"])
        for part in self.stream_generate(self.model, self.tokenizer, prompt=ids,
                                         max_tokens=min(request["max_tokens"], room), sampler=sampler):
            if part.text:
                yield {"delta": part.text}
            if part.finish_reason:
                yield completion(part.finish_reason, part.prompt_tokens, part.generation_tokens)
                return
        raise RuntimeError("MLX 生成提前结束。")

    def close(self):
        pass


def gguf_architecture(path):
    """Read the architecture tag without reading tensor data or importing NumPy."""
    try:
        with path.open("rb") as model:
            magic, version, _, fields = struct.unpack("<4sIQQ", model.read(24))
            if magic != b"GGUF" or version not in (2, 3):
                return "gguf"

            def number(code):
                return struct.unpack("<" + code, model.read(struct.calcsize(code)))[0]

            def string():
                size = number("Q")
                if size > 65535:
                    raise ValueError("invalid GGUF metadata")
                return model.read(size).decode()

            def skip(kind, count=1):
                if kind == 8:
                    for _ in range(count):
                        model.seek(number("Q"), os.SEEK_CUR)
                elif kind == 9:
                    for _ in range(count):
                        skip(number("I"), number("Q"))
                else:
                    sizes = (1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8)
                    model.seek(sizes[kind] * count, os.SEEK_CUR)

            for _ in range(fields):
                key, kind = string(), number("I")
                if key == "general.architecture" and kind == 8:
                    return string() or "gguf"
                skip(kind)
    except (ValueError, OSError, IndexError, struct.error, RecursionError):
        pass
    return "gguf"


class LlamaModel:
    @staticmethod
    def ocr_options(executable, env):
        try:
            help_text = subprocess.run([str(executable), "--help"], env=env, timeout=4,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        # Consecutive OCR pages rarely reuse their image tokens. Keep the loaded model
        # available during a job without keeping a cross-page cache or idle GPU allocation.
        options = []
        if b"--cache-ram" in help_text:
            options.extend(["--cache-ram", "0"])
            if b"--no-cache-idle-slots" in help_text:
                options.append("--no-cache-idle-slots")
        if b"--sleep-idle-seconds" in help_text:
            options.extend(["--sleep-idle-seconds", "60"])
        return options

    def __init__(self, path, context, threads, llama_server=None, projector=None):
        if not path.is_file():
            raise ValueError("llama.cpp 模型需要本地 GGUF 文件。")
        executable = shutil.which(llama_server or os.environ.get("DUBHE_LLAMA_SERVER") or "llama-server")
        if not executable:
            raise RuntimeError("找不到 llama-server。请安装它，或设置 DUBHE_LLAMA_SERVER。")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.key = secrets.token_urlsafe(32)
        self.alias = "dubhe-" + secrets.token_hex(12)
        self.work = tempfile.TemporaryDirectory(prefix="dubhe-llama-")
        self.process = None
        try:
            key_file = Path(self.work.name) / "key"
            key_file.touch(mode=0o600)
            key_file.write_text(self.key + "\n", encoding="ascii")
            # Use just the flags for this private worker and ignore inherited server options.
            env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_")}
            command = [str(Path(executable).resolve()), "--model", str(path), "--alias", self.alias,
                       "--host", "127.0.0.1", "--port", str(self.port), "--api-key-file", str(key_file),
                       "--ctx-size", str(context), "--threads", str(threads), "--parallel", "1",
                       "--no-webui", "--no-context-shift", "--jinja"]
            if projector:
                command.extend(["--mmproj", str(projector), "--image-max-tokens", "7168", "--reasoning", "off"])
                command.extend(self.ocr_options(executable, env))
            self.process = subprocess.Popen(
                command,
                cwd=self.work.name, env=env, stdin=subprocess.DEVNULL, stdout=sys.stderr,
            )
            info = self.wait_until_ready()
            status, props = self.get("/props")
            if status != 200:
                raise RuntimeError(f"读取 llama-server 上下文失败（HTTP {status}）。")
            actual_context = props.get("default_generation_settings", {}).get("n_ctx")
            self.info = dict(backend="llama.cpp", arch=gguf_architecture(path),
                             context=actual_context or context)
            params = (info.get("meta") or {}).get("n_params")
            if isinstance(params, int) and params > 0:
                self.info["params"] = params
        except BaseException:
            self.close()
            raise

    def connection(self, timeout):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def get(self, path):
        with closing(self.connection(2)) as conn:
            conn.request("GET", path, headers={"Authorization": "Bearer " + self.key})
            with conn.getresponse() as response:
                return response.status, json.load(response)

    def wait_until_ready(self):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server 启动失败（退出码 {self.process.returncode}）。请查看日志。")
            try:
                status, data = self.get("/v1/models")
            except (OSError, http.client.HTTPException):
                time.sleep(0.1)
                continue
            if status == 200:
                loaded = next((m for m in data.get("data", []) if m.get("id") == self.alias), None)
                if loaded is None:
                    raise RuntimeError("无法确认私有 llama-server 的模型。")
                if loaded.get("meta") is not None:
                    return loaded
            elif status != 503:
                raise RuntimeError(f"llama-server 返回了 HTTP {status}。")
            time.sleep(0.1)
        raise RuntimeError("等待 llama-server 加载模型超时。")

    def generate(self, request):
        body = {**request, "model": self.alias, "stream": True,
                "stream_options": {"include_usage": True}}
        headers = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}
        with closing(self.connection(600)) as conn:
            conn.request("POST", "/v1/chat/completions", json.dumps(body).encode("utf-8"), headers)
            with conn.getresponse() as response:
                if response.status != 200:
                    try:
                        error = json.load(response).get("error")
                    except ValueError:
                        error = None
                    message = error.get("message") if isinstance(error, dict) else error
                    raise RuntimeError(message or f"llama-server 返回了 HTTP {response.status}。")
                reason = None
                usage = {}
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        yield completion(reason, usage.get("prompt_tokens"), usage.get("completion_tokens"))
                        return
                    event = json.loads(data)
                    if "error" in event:
                        error = event["error"]
                        raise RuntimeError(error.get("message") if isinstance(error, dict) else str(error))
                    for choice in event.get("choices", []):
                        text = choice.get("delta", {}).get("content")
                        if text:
                            yield {"delta": text}
                        reason = choice.get("finish_reason") or reason
                    if event.get("usage"):
                        usage = event["usage"]
        raise RuntimeError("llama-server 生成提前结束。")

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.work.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "llama"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--llama-server")
    parser.add_argument("--mmproj", type=Path)
    args = parser.parse_args()

    # Library and subprocess diagnostics belong on stderr, including native stdout.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def emit(event):
        print(json.dumps(event, ensure_ascii=False), file=protocol, flush=True)

    def shutdown(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, shutdown)
    model = None
    try:
        if args.ctx <= 0 or args.threads <= 0:
            raise ValueError("上下文和线程数必须大于零。")
        path = args.model.resolve(strict=True)
        projector = args.mmproj.resolve(strict=True) if args.mmproj else None
        if projector and (args.backend != "llama" or not projector.is_file()):
            raise ValueError("扫描需要本机 llama.cpp 及视觉组件文件。")
        model = MLXModel(path, args.ctx) if args.backend == "mlx" else LlamaModel(
            path, args.ctx, args.threads, args.llama_server, projector)
        emit({"ready": True, **model.info})
        for line in sys.stdin:
            try:
                for event in model.generate(chat_request(json.loads(line), allow_image=projector is not None)):
                    emit(event)
            except BrokenPipeError:
                return 0
            except Exception as e:
                emit({"error": str(e) or type(e).__name__})
        return 0
    except BrokenPipeError:
        return 0
    except Exception as e:
        emit({"error": str(e) or type(e).__name__})
        return 1
    finally:
        if model is not None:
            model.close()


if __name__ == "__main__":
    sys.exit(main())
