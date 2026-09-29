"""Loopback-only ComfyUI transport with resumable, non-duplicating submissions."""

import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from .config import validate_endpoint


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ComfyClient:
    def __init__(self, endpoint: str):
        self.endpoint = validate_endpoint(endpoint)
        self.opener = build_opener(ProxyHandler({}))

    def request(self, path, data=None, *, content_type="application/json", raw=False):
        if data is not None and not isinstance(data, bytes):
            data = json.dumps(data).encode()
        request = Request(self.endpoint + path, data=data, headers={"Content-Type": content_type})
        try:
            with self.opener.open(request, timeout=60) as response:
                result = response.read()
        except HTTPError as exc:
            detail = exc.read(16384).decode(errors="replace")
            raise RuntimeError(f"ComfyUI HTTP {exc.code}: {detail}") from exc
        return result if raw else json.loads(result)

    def upload_audio(self, path: Path):
        content = path.read_bytes()
        name = hashlib.sha256(content).hexdigest() + path.suffix.lower()
        boundary = "lic" + uuid.uuid4().hex
        # ComfyUI's generic input-file upload route is historically named image.
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                "Content-Type: application/octet-stream\r\n\r\n").encode() + content
        body += f"\r\n--{boundary}--\r\n".encode()
        result = self.request("/upload/image", body, content_type=f"multipart/form-data; boundary={boundary}")
        if result.get("type") != "input" or result.get("subfolder") not in {"", None} or result.get("name") != name:
            raise ValueError(f"Unexpected ComfyUI upload location: {result}")
        return name

    def execute(self, workflow, cache: Path, *, identity, timeout_seconds):
        cache.mkdir(parents=True, exist_ok=True)
        request_path = cache / "request.json"
        fingerprint = hashlib.sha256(json.dumps({"workflow": workflow, "identity": identity},
                                                sort_keys=True).encode()).hexdigest()
        if request_path.exists():
            state = json.loads(request_path.read_text(encoding="utf-8"))
            if state["fingerprint"] != fingerprint:
                raise ValueError("H3 cached request inputs changed; use a new build/run directory")
            if state.get("endpoint") != self.endpoint:
                raise ValueError("H3 cached request belongs to another worker endpoint")
            if (cache / "submission-error.json").exists():
                raise RuntimeError(f"H3 previously rejected this submission; inspect {cache / 'submission-error.json'}")
        else:
            state = {"fingerprint": fingerprint, "prompt_id": str(uuid.uuid4()),
                     "endpoint": self.endpoint, "workflow": workflow, "identity": identity}
            # Persist BEFORE the POST. A lost response must never repeat generation.
            save_json(request_path, state)
            try:
                response = self.request("/prompt", {"prompt": workflow, "prompt_id": state["prompt_id"]})
                save_json(cache / "submission.json", response)
                if response.get("prompt_id") != state["prompt_id"]:
                    raise ValueError("ComfyUI returned an unexpected prompt ID")
            except (URLError, TimeoutError):
                print(f"H3 submission response lost; checking persisted prompt {state['prompt_id']}", flush=True)
            except RuntimeError as exc:
                save_json(cache / "submission-error.json", {"error": str(exc)})
                raise
        native = cache / "native.flac"
        receipt = cache / "native.json"
        if native.exists() and receipt.exists():
            if hashlib.sha256(native.read_bytes()).hexdigest() != json.loads(receipt.read_text(encoding="utf-8"))["sha256"]:
                raise ValueError("Cached H3 native audio was modified")
            return native
        prompt_id = state["prompt_id"]
        print(f"H3 queued {prompt_id}; native audio cache: {cache}", flush=True)
        started = time.monotonic()
        deadline = started + timeout_seconds
        last_report = 0
        while time.monotonic() < deadline:
            try:
                history = self.request(f"/history/{prompt_id}").get(prompt_id)
            except (URLError, TimeoutError):
                history = None
            if history:
                save_json(cache / "history.json", history)
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"H3 execution failed; inspect {cache / 'history.json'}")
                if status.get("completed"):
                    audio = history.get("outputs", {}).get("92", {}).get("audio", [])
                    if len(audio) != 1 or audio[0].get("type") != "output":
                        raise ValueError("H3 completed without exactly one native audio artifact")
                    content = self.request("/view?" + urlencode(audio[0]), raw=True)
                    if not content.startswith(b"fLaC"):
                        raise ValueError("H3 output is not a FLAC audio file")
                    temporary = cache / "native.partial.flac"
                    temporary.write_bytes(content)
                    temporary.replace(native)
                    save_json(receipt, {"sha256": hashlib.sha256(content).hexdigest(), "file": audio[0]})
                    return native
            if time.monotonic() - last_report >= 30:
                if not history and time.monotonic() - started >= 30:
                    try:
                        queue = self.request("/queue")
                    except (URLError, TimeoutError):
                        queue = None
                    if queue is not None and not any(
                            len(entry) > 1 and entry[1] == prompt_id
                            for group in ("queue_running", "queue_pending") for entry in queue.get(group, [])):
                        raise RuntimeError(f"Persisted H3 prompt {prompt_id} is absent from queue and history. "
                                           "The worker may have restarted. Inspect the saved request before using a new run.")
                print(f"H3 waiting for {prompt_id}", flush=True)
                last_report = time.monotonic()
            time.sleep(2)
        raise TimeoutError(f"H3 prompt {prompt_id} did not finish. Request is cached in {cache}; "
                           "rerun to resume polling. This does not cancel or resubmit the GPU job.")
