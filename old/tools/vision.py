"""HTTP + image-prep helpers for the vision calls grade_flats.py makes.

Lifted from match_bra.py, which lived outside this project. Only the pieces
grade_flats.py imports are here; the bra-matching pipeline itself is not.

Two things differ from the original and both matter:

  * `Client` talks to the same llama-server the harness talks to. The harness
    sets QWEN_BASE_URL without a `/v1` suffix and match_bra set it with one, so
    the suffix is normalised here rather than depending on which of the two
    exported the variable last.

  * The API key falls back to the harness's own default. llama.cpp leaves
    /v1/models open but 401s completions, so a missing key looks like a working
    server that refuses every actual call.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
SMALL = CACHE / "small"


def _base_url() -> str:
    url = os.environ.get("QWEN_BASE_URL", "http://10.11.245.41:8091").rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


DEFAULT_BASE_URL = _base_url()
DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "")   # empty = first the server lists


def _load_api_key() -> str:
    if os.environ.get("QWEN_API_KEY"):
        return os.environ["QWEN_API_KEY"]
    kf = ROOT / ".qwen_key"
    if kf.exists():
        return kf.read_text().strip()
    return "pick-a-long-secret-string"      # harness.py's default


API_KEY = _load_api_key()

IS_TTY = sys.stdout.isatty()


def transient(msg: str) -> None:
    """In-flight status. Only drawn on a terminal, where settled() overwrites it;
    piping or redirecting drops it so logs are not littered with half-lines."""
    if IS_TTY:
        print(msg, end="", flush=True)


def settled(msg: str) -> None:
    """Final line, replacing any transient status that preceded it."""
    print(("\r" + msg + " " * 25) if IS_TTY else msg)


def http_json(url: str, payload: dict | None = None, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class Client:
    def __init__(self, base_url: str, model: str, timeout: int = 300,
                 think: bool = False):
        base_url = base_url.rstrip("/")
        self.base_url = base_url if base_url.endswith("/v1") else base_url + "/v1"
        self.model = model
        self.timeout = timeout
        # Qwen3 is a reasoning model: with thinking on, llama.cpp puts the chain
        # in `reasoning_content` and leaves `content` empty until it finishes.
        # Off by default - measured ~8x slower for no gain on judging tasks.
        self.think = think

    def models(self) -> list[str]:
        return [m["id"] for m in http_json(f"{self.base_url}/models",
                                           timeout=20).get("data", [])]

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        ids = self.models()
        if not ids:
            raise RuntimeError("server returned no models")
        self.model = ids[0]
        return self.model

    def chat(self, content: list, max_tokens: int = 900,
             temperature: float = 0.0, system: str | None = None) -> str:
        # describe.py's standing instruction ("you see ONE side, never infer the
        # other") has to hold for the whole call, not read as one more paragraph
        # of the question. Optional, so every existing caller is unaffected.
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": msgs,
            # Thinking needs headroom for the chain *plus* the answer; running
            # out mid-chain returns empty content and finish_reason "length".
            "max_tokens": max_tokens * 4 if self.think else max_tokens,
            "temperature": temperature,
        }
        if not self.think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        out = http_json(f"{self.base_url}/chat/completions", payload,
                        timeout=self.timeout)
        choice = out["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if not text and choice.get("finish_reason") == "length":
            raise RuntimeError(
                "model hit the token limit while thinking and returned no "
                "answer; raise max_tokens or drop think")
        return text


def ensure_small(src: Path, max_dim: int = 1024) -> Path:
    """Downscale into .cache/small. A 4K generation is ~20MB; sending that raw
    blows up both the request and the model's vision budget."""
    SMALL.mkdir(parents=True, exist_ok=True)
    # The parent folder is part of the cache name. Every run writes the SAME
    # stems - offset_upload.jpg, cand_01.jpg - so keying on the stem alone made
    # the cache collide across runs, and the mtime guard could not catch it: the
    # entry cached by today's run is newer than an August source, so the check
    # passes and the wrong garment is returned as a hit. Caught when the
    # description step read one bra and produced a description of another.
    # match_reference.py's copy has always done this; this one had not.
    tag = hashlib.sha1(str(src.resolve().parent).encode()).hexdigest()[:8]
    dst = SMALL / f"{src.stem}__{tag}__{max_dim}.jpg"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    subprocess.run(["sips", "-Z", str(max_dim), str(src), "--out", str(dst)],
                   check=True, capture_output=True)
    return dst


def data_url(path: Path) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode()}"


def image_part(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(path)}}


def text_part(s: str) -> dict:
    return {"type": "text", "text": s}


def parse_json_blob(text: str) -> dict:
    """Models wrap JSON in fences or prose often enough that this is worth it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])
