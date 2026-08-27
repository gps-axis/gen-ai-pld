#!/usr/bin/env python3
"""
harness.py - a very small Claude-Code-style agent loop driven by a local model.

Points a tool-using agent loop at an OpenAI-compatible endpoint (llama.cpp or
vLLM), loads a SKILL.md as its operating manual, and lets it work in a
workspace until it calls finish() or hits the iteration cap.

    ./run.sh --skill laydown-match
    ./run.sh --skill laydown-match --task "Run stage 1 only and report the numbers"
    ./run.sh --task "Summarise what is in this folder" --max-iters 5

Design notes worth knowing before you edit this file:

  * The model is a *reasoning* model. Replies arrive split across a
    chain-of-thought field and `content` (the actual answer). llama.cpp calls
    that field `reasoning_content` and vLLM calls it `reasoning`; both are read.
    A stingy max_tokens returns content="" and finish_reason="length", which
    looks like a crash but is just truncation mid-thought. See call_model().

  * The context window is read off the server at startup, not assumed - see
    preflight(). It has been both 32768 (llama.cpp) and 262144 (vLLM) on this
    project, and every budget here scales off it. Tool results are still
    truncated on the way in (full text always spooled to disk), and history is
    compacted past COMPACT_FRACTION of the window. See Session.compact().

  * Prior-turn reasoning is dropped from history. Keeping it is what makes
    naive local-model agent loops fall over at turn ~6.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# tools/ is not a package - the pipeline scripts are run from inside it - so the
# trace module is reached the same way they reach common.py.
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from runlog import trace as TR, stream_subprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

def _base_url() -> str:
    """The server root, without the /v1 suffix this file appends itself.

    The endpoint gets pasted around in both forms - vision.py wants it WITH
    /v1, this file builds /v1/chat/completions itself - so accept either and
    normalise, rather than producing /v1/v1/chat/completions from a perfectly
    reasonable QWEN_BASE_URL.
    """
    url = os.environ.get("QWEN_BASE_URL", "http://10.11.245.41:8091").rstrip("/")
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


BASE_URL = _base_url()
API_KEY = os.environ.get("QWEN_API_KEY", "pick-a-long-secret-string")

# Fallback only. The real window is read off the server at startup - see
# preflight(). Hardcoding it was wrong the moment the endpoint moved: the old
# llama.cpp box served 32768 and the current vLLM one serves 262144, so a stale
# constant meant compacting away tool output at 21k with 240k still free, and
# printing a context percentage that was off by 8x.
N_CTX_FALLBACK = 32768
MAX_TOKENS = 3000        # generation budget per turn; reasoning eats most of it
COMPACT_FRACTION = 0.64  # compact history once the prompt passes this much of N_CTX
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 900    # seconds; a long reasoning turn on a 35B model is slow
RECONNECT_BUDGET = 600   # keep a run alive across a sleeping laptop / Wi-Fi blip
RECONNECT_POLL = 10

MAX_TOOL_CHARS = 4000    # per tool result fed back to the model
BASH_TIMEOUT = 900

# "The library has nothing close enough to this garment" is an expected answer,
# not a fault, and it has to be distinguishable from one. It used to leave here
# as 1, the same code a crash leaves as, which made a scheduler show a run that
# worked perfectly as a failure and put it in the same list as a broken model
# server. 20 is far enough from the small codes to collide with nothing: 1 is
# breakage, 2 is the entrypoint's own usage errors, 3 is a short delivery.
EXIT_NO_REFERENCE = 20

# "The pre-clean gate rejected the source" is the same kind of answer: the
# pipeline worked, and what it produced is a refusal to spend money on a garment
# it can prove was damaged before generation. Distinct from 20 because the fix
# is different - one needs a hero uploaded, the other needs the input photo or
# the eraser looked at.
EXIT_UNCLEAN_SOURCE = 21

# Tools that only observe. These never prompt for approval.
READONLY_TOOLS = {"read_file", "view_image", "compare_images", "finish"}

# Refused outright, in any mode. Not a security boundary - a guardrail against
# an agent that has decided rm -rf is the shortest path to a clean workspace.
BASH_DENY = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]", "recursive/forced rm"),
    (r"\bmkfs\b|\bdd\s+if=.*\bof=/dev/", "raw disk write"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"\bshutdown\b|\breboot\b|\bhalt\b", "power control"),
    (r"\bgit\s+push\b", "git push"),
    (r"\bcurl\b[^|]*\|\s*(ba)?sh", "curl pipe to shell"),
    # Reading the pipeline's own source is never the job, and it has cost three
    # runs: 44%, 54% and 51% of the context window spent before any real step,
    # one of them running out of turns after two images. The SKILL says how to
    # call these; --help covers the rest.
    (r"\b(cat|head|tail|less|more|bat)\b[^|;]*\b(tools/\S+\.py|harness\.py)",
     "dumping tool source - use --help, the skill documents these"),
]


def c(code: str, s: str) -> str:
    """Wrap s in an ANSI colour unless output is redirected."""
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


DIM, BOLD, RED, GRN, YEL, BLU, CYA = "2", "1", "31", "32", "33", "34", "36"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> dict:
    """Tolerant .env reader - handles `KEY = value`, quotes, comments."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def child_env(workspace: Path) -> dict:
    """Environment handed to bash: inherited, plus .env, plus the CA fix.

    Credentials come from the harness's own .env first, then the workspace's,
    so a workspace elsewhere on disk still gets FAL_KEY without a copy of it
    sitting next to the images. A workspace .env wins where both define a key.

    The skill calls out that certifi's Mozilla-only bundle rejects the corporate
    proxy certificate, so point requests at the Homebrew OpenSSL bundle when it
    is present.
    """
    env = dict(os.environ)
    env.update(load_dotenv(HERE / ".env"))
    if workspace.resolve() != HERE:
        env.update(load_dotenv(workspace / ".env"))
    ca = Path("/opt/homebrew/etc/openssl@3/cert.pem")
    if ca.exists():
        env.setdefault("SSL_CERT_FILE", str(ca))
        env.setdefault("REQUESTS_CA_BUNDLE", str(ca))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def find_python() -> str:
    """The interpreter the agent is told to run pipeline scripts with.

    The project .venv first, then the old sibling env3.10, then whatever is
    running us. Since this project left the Axis tree, a bare `python3` on this
    machine is Homebrew 3.14 with none of numpy/PIL/scipy installed, so handing
    that to the agent means it discovers the problem as an ImportError mid-run.
    """
    for cand in (HERE / ".venv" / "bin" / "python",
                 HERE.parent / "env3.10" / "bin" / "python3"):
        if cand.exists():
            return str(cand)
    return sys.executable


# ---------------------------------------------------------------------------
# Model client
# ---------------------------------------------------------------------------

class ModelError(RuntimeError):
    pass


class ConnectionLost(ModelError):
    """Server unreachable - as opposed to a bad request it will never accept."""


class MalformedToolCall(ModelError):
    """The model emitted tool-call JSON the server could not parse."""


_MSG_SEEN: dict[str, list[str]] = {}


def _fp(msg) -> str:
    return hashlib.sha1(json.dumps(msg, sort_keys=True, default=str).encode()).hexdigest()


def _trace_request_messages(payload: dict) -> None:
    """Log the messages this request added or rewrote, not the whole history.

    Dumping every message on every call is quadratic - by turn 30 the trace is
    thirty copies of the same conversation and unreadable. Messages are
    fingerprinted per conversation instead, so each one is written once, when it
    first appears, and again if it changes: compaction rewrites tool results in
    place, and that rewrite is a real event worth seeing at the position it
    happened. Vision calls are their own conversation - keyed off the first
    message - so they never collide with the agent's history.
    """
    msgs = payload.get("messages") or []
    if not msgs:
        return
    conv = _fp(msgs[0])[:12]
    seen = _MSG_SEEN.setdefault(conv, [])
    parts = []
    for i, m in enumerate(msgs):
        f = _fp(m)
        if i < len(seen):
            if seen[i] == f:
                continue
            seen[i] = f
            parts.append((i, "rewritten", m))
        else:
            seen.append(f)
            parts.append((i, "new", m))
    if not parts:
        TR.debug("http", "messages unchanged since the previous request",
                 conv=conv, count=len(msgs))
        return
    body = "\n".join(f"[{i}] {kind}: {json.dumps(m, indent=2, default=str)}"
                     for i, kind, m in parts)
    TR.debug("http", "request messages", body=body, conv=conv,
             total=len(msgs), logged=len(parts))


def post(payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    body_bytes = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body_bytes,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    TR.debug("http", "POST /v1/chat/completions",
             bytes=len(body_bytes), messages=len(payload.get("messages", [])),
             max_tokens=payload.get("max_tokens"),
             temperature=payload.get("temperature"),
             tools=len(payload.get("tools") or []))
    _trace_request_messages(payload)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        data = json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        TR.error("http", f"HTTP {e.code} after {time.time() - t0:.1f}s", body=body)
        if e.code == 401:
            raise ModelError(
                "401 from the model server. Set QWEN_API_KEY to the key the "
                "server was started with (--api-key on llama.cpp, "
                "--api-key on vLLM)."
            ) from e
        # llama.cpp 500s when the model's own tool-call JSON will not parse -
        # almost always a long string argument truncated by max_tokens. That is
        # a generation accident, not a broken request, so it is worth retrying
        # with more room. Observed writing a ~6KB markdown file in one call.
        if e.code == 500 and "parse" in body.lower() and "tool" in body.lower():
            raise MalformedToolCall(body) from e
        raise ModelError(f"HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", e)
        TR.error("http", f"unreachable after {time.time() - t0:.1f}s",
                 error=f"{type(e).__name__}: {reason}")
        raise ConnectionLost(
            f"Cannot reach {BASE_URL} ({reason}). Is the model server up, and are "
            f"you on the same network?"
        ) from e

    usage = data.get("usage", {}) or {}
    choice = (data.get("choices") or [{}])[0]
    TR.debug("http", f"200 in {time.time() - t0:.1f}s",
             bytes=len(raw), finish=choice.get("finish_reason"),
             prompt_tokens=usage.get("prompt_tokens"),
             completion_tokens=usage.get("completion_tokens"))
    TR.debug("http", "response payload",
             body=json.dumps(data, indent=2, default=str))
    return data


def server_up(timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_server() -> bool:
    """Block until the server answers again, or RECONNECT_BUDGET expires.

    Returns True if it came back. In-memory conversation state is what makes
    this worth doing: the run can pick up exactly where it left off.
    """
    deadline = time.time() + RECONNECT_BUDGET
    budget = (f"{RECONNECT_BUDGET // 60} min" if RECONNECT_BUDGET >= 60
              else f"{RECONNECT_BUDGET}s")
    TR.warn("server", "waiting for the server to come back", budget_s=RECONNECT_BUDGET)
    print()
    print(c(YEL, f"  server unreachable - waiting up to {budget} for it to come back"))
    print(c(DIM, "  (wake the machine / restart the server; Ctrl-C to give up)"))
    while time.time() < deadline:
        time.sleep(RECONNECT_POLL)
        if server_up():
            print("\r" + " " * 48 + "\r", end="")
            print(c(GRN, "  server is back - resuming\n"))
            return True
        left = max(0, int(deadline - time.time()))   # a slow poll can overshoot
        print(c(DIM, f"    still down, {left//60}m{left%60:02d}s left   "),
              end="\r", flush=True)
    print("\r" + " " * 48 + "\r", end="")
    print(c(RED, "  gave up waiting."))
    return False


def call_model(messages, tools=None, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
               retries=2) -> dict:
    """One chat completion, with the reasoning-model truncation trap handled.

    A reasoning model spends tokens on `reasoning_content` before writing any
    `content`. If the budget runs out mid-thought the reply is
    content="" / finish_reason="length" - not an error, just a starved turn.
    Retry once with a bigger budget rather than surfacing an empty answer.
    """
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    last_err = None
    attempt = 0
    while True:
        try:
            data = post(payload)
        except MalformedToolCall as e:
            last_err = e
            attempt += 1
            if attempt > retries:
                raise ModelError(
                    "The model kept emitting tool-call JSON the server could not "
                    "parse. This is usually one very large string argument being "
                    "truncated - have the agent write long files via bash heredoc "
                    "instead of write_file, or raise MAX_TOKENS."
                ) from e
            payload["max_tokens"] = min(int(payload["max_tokens"] * 2), 8000)
            TR.warn("model", "malformed tool-call JSON; retrying",
                    body=str(e), attempt=attempt, max_tokens=payload["max_tokens"])
            print(c(YEL, f"  ! malformed tool-call JSON (likely a truncated "
                         f"argument); retrying at max_tokens="
                         f"{payload['max_tokens']}"))
            continue
        except ConnectionLost as e:
            # A local server on a laptop sleeps, drops off Wi-Fi, or takes a new
            # DHCP lease. Losing a 20-turn run to a 30-second blip is worse than
            # waiting, so keep trying for RECONNECT_BUDGET seconds first.
            last_err = e
            TR.warn("model", "connection lost; waiting for the server", body=str(e))
            if wait_for_server():
                TR.info("model", "server returned; retrying the same turn")
                continue
            raise
        except ModelError as e:
            last_err = e
            attempt += 1
            TR.warn("model", f"model error (attempt {attempt}/{retries})", body=str(e))
            if attempt > retries:
                raise
            time.sleep(2 * attempt)
            continue

        choice = data["choices"][0]
        msg = choice["message"]
        starved = (
            choice.get("finish_reason") == "length"
            and not (msg.get("content") or "").strip()
            and not msg.get("tool_calls")
        )
        if starved and attempt < retries:
            attempt += 1        # must advance, or a persistently starved turn spins
            payload["max_tokens"] = min(int(payload["max_tokens"] * 2), 8000)
            TR.warn("model", "turn starved by the token budget; retrying",
                    attempt=attempt, max_tokens=payload["max_tokens"])
            print(c(DIM, f"    (turn starved by token budget; retrying at "
                        f"max_tokens={payload['max_tokens']})"))
            continue
        return data

    raise last_err or ModelError("model call failed")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the workspace and return stdout+stderr. "
                "Use this to run python, inspect files, and do the actual work. "
                "FAL_KEY and the CA-bundle variables are already set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file. Use offset/limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "First line, 1-based."},
                    "limit": {"type": "integer", "description": "How many lines."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace the first exact occurrence of old_text with new_text. "
                "old_text must match the file byte for byte."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": (
                "LOOK at an image and get a description back. Optionally crop "
                "first with box='x,y,w,h' in source pixels - use this to inspect "
                "a seam or a silhouette edge at 1:1 with no resampling. Ask a "
                "specific question; vague questions get vague answers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "question": {"type": "string", "description": "What to look for."},
                    "box": {"type": "string", "description": "Optional crop 'x,y,w,h'."},
                },
                "required": ["path", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_images",
            "description": (
                "LOOK at two images side by side in ONE vision call and get "
                "the differences back. Use this - not two view_image calls - "
                "whenever the question is 'how does this differ from the "
                "reference'. Optionally crop each with box_a/box_b='x,y,w,h' "
                "in that image's own source pixels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path_a": {"type": "string", "description": "First image."},
                    "path_b": {"type": "string", "description": "Second image."},
                    "question": {"type": "string", "description": "What to compare."},
                    "box_a": {"type": "string", "description": "Optional crop of A."},
                    "box_b": {"type": "string", "description": "Optional crop of B."},
                },
                "required": ["path_a", "path_b", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call when the goal is met or you are blocked. Ends the run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What you did, with numbers."},
                    "status": {
                        "type": "string",
                        "enum": ["done", "blocked"],
                        "description": "done if the goal is met, blocked otherwise.",
                    },
                },
                "required": ["summary", "status"],
            },
        },
    },
]


class Tools:
    def __init__(self, workspace: Path, run_dir: Path, allow_outside: bool):
        self.ws = workspace
        self.run_dir = run_dir
        self.allow_outside = allow_outside
        self.env = child_env(workspace)
        self.python = find_python()
        self._spool = 0

    # -- path handling ----------------------------------------------------
    def resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.ws / p
        p = p.resolve()
        if not self.allow_outside:
            # Reads outside the workspace are fine (the reference implementation
            # lives next door); writes are confined by the caller.
            pass
        return p

    def _guard_write(self, p: Path):
        if self.allow_outside:
            return
        try:
            p.relative_to(self.ws)
        except ValueError:
            raise PermissionError(
                f"Refusing to write outside the workspace: {p}\n"
                f"Workspace is {self.ws}. Re-run with --allow-outside to permit this."
            )

    def spool(self, name: str, text: str) -> Path:
        """Write full output to disk so truncation never loses anything."""
        self._spool += 1
        p = self.run_dir / "tool_output" / f"{self._spool:03d}_{name}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        TR.debug("tool", "spooled oversized output", path=str(p), chars=len(text))
        return p

    # -- individual tools -------------------------------------------------
    def bash(self, command: str) -> str:
        for pat, why in BASH_DENY:
            if re.search(pat, command):
                TR.warn("bash", "REFUSED by guardrail", body=command, rule=why)
                return f"REFUSED: command blocked by harness guardrail ({why})."
        TR.info("bash", "run", body=command, cwd=str(self.ws))
        t0 = time.time()
        try:
            r = subprocess.run(
                command, shell=True, cwd=self.ws, env=self.env,
                capture_output=True, text=True, timeout=BASH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            TR.error("bash", f"TIMEOUT after {BASH_TIMEOUT}s; killed", body=command)
            return f"TIMEOUT after {BASH_TIMEOUT}s. Command killed."
        dt = time.time() - t0
        # stdout and stderr are logged apart even though the model gets them
        # concatenated: when a pipeline script fails, which stream carried the
        # traceback is the first thing you want to know.
        TR.info("bash", f"exit={r.returncode}", secs=round(dt, 2),
                out_chars=len(r.stdout or ""), err_chars=len(r.stderr or ""))
        if r.stdout:
            TR.debug("bash", "stdout", body=r.stdout)
        if r.stderr:
            TR.debug("bash", "stderr", body=r.stderr)
        out = (r.stdout or "") + (r.stderr or "")
        if not out.strip():
            out = "(no output)"
        return f"exit={r.returncode}\n{out}"

    SOURCE_GUARD = re.compile(r"(^|/)(harness\.py|tools/[^/]+\.py)$")

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        if self.SOURCE_GUARD.search(str(self.resolve(path))):
            return ("REFUSED: that is pipeline source, not an input. The skill "
                    "documents how to call it and `<script> --help` covers the "
                    "rest. Three runs have burned ~half their context window "
                    "reading these before doing any work.")
        p = self.resolve(path)
        if not p.exists():
            return f"ERROR: no such file: {p}"
        if p.is_dir():
            return f"ERROR: {p} is a directory. Use bash ls."
        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception as e:
            return f"ERROR reading {p}: {e}"
        offset = max(1, int(offset or 1))
        chunk = lines[offset - 1: offset - 1 + int(limit or 2000)]
        body = "\n".join(f"{offset + i:6d}\t{l}" for i, l in enumerate(chunk))
        tail = ""
        if offset - 1 + len(chunk) < len(lines):
            tail = (f"\n... {len(lines) - (offset - 1 + len(chunk))} more lines. "
                    f"Re-read with offset={offset + len(chunk)}.")
        return body + tail

    def write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        self._guard_write(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content)
        n = len(content.splitlines())
        return f"{'Overwrote' if existed else 'Wrote'} {p} ({n} lines)."

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        p = self.resolve(path)
        self._guard_write(p)
        if not p.exists():
            return f"ERROR: no such file: {p}"
        src = p.read_text(errors="replace")
        if old_text not in src:
            return ("ERROR: old_text not found. It must match byte for byte, "
                    "including indentation. Read the file again and copy exactly.")
        if src.count(old_text) > 1:
            return (f"ERROR: old_text appears {src.count(old_text)} times and is "
                    f"ambiguous. Include more surrounding context.")
        p.write_text(src.replace(old_text, new_text, 1))
        return f"Edited {p}."

    def _prep_image(self, path: str, question: str, box: str | None):
        """Crop, downscale and base64 one image for a vision call.

        Returns (b64, note) on success or (None, error_string) - the caller
        returns that string to the model verbatim.
        """
        try:
            from PIL import Image
        except ImportError:
            return None, (f"ERROR: Pillow not available to the harness "
                          f"interpreter ({sys.executable}). Run the harness "
                          f"via ./run.sh.")
        p = self.resolve(path)
        if not p.exists():
            return None, f"ERROR: no such image: {p}"
        try:
            im = Image.open(p)
            im.load()
        except Exception as e:
            return None, f"ERROR opening {p}: {e}"
        src_size = im.size
        note = f"{p.name} source {src_size[0]}x{src_size[1]}"

        # Asking to "zoom in" without a box gets you the whole frame squeezed to
        # 1024px - the opposite of a close look. This happened on a real run:
        # the model called downsampled fabric "real and consistent" while a 1:1
        # crop showed the knit structure had been destroyed. Refuse instead.
        zoomish = re.search(r"\b(zoom|close[- ]?up|1:1|pixel|crop|seam|stitch|"
                            r"fringe|texture|weave|knit)\b", question, re.I)
        if zoomish and not box and max(src_size) > 1400:
            return None, (f"REFUSED: '{zoomish.group(0)}' asks for a close look, "
                          f"but no box was given for {p.name}, so this would "
                          f"downscale the whole {src_size[0]}x{src_size[1]} "
                          f"frame to 1024px and show you nothing of the kind. "
                          f"Pass a box 'x,y,w,h' in source pixels (a few "
                          f"hundred px wide) to inspect at 1:1.")

        if box:
            try:
                x, y, w, h = (int(float(v)) for v in re.split(r"[,\s]+", box.strip()))
                im = im.crop((x, y, x + w, y + h))
                note += f", crop @({x},{y}) {w}x{h}"
            except Exception:
                return None, "ERROR: box must be 'x,y,w,h' in source pixels."

        # Only downscale when we must. A 1:1 crop is the point of this tool.
        if max(im.size) > 1024:
            im.thumbnail((1024, 1024), Image.LANCZOS)
            note += f", downscaled to {im.size[0]}x{im.size[1]}"
        else:
            note += f", sent 1:1 at {im.size[0]}x{im.size[1]}"

        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=92)
        TR.debug("vision", "prepared image", note=note, jpeg_bytes=buf.tell(),
                 path=str(p), box=box)
        return base64.b64encode(buf.getvalue()).decode(), note

    @staticmethod
    def _ask_vision(text: str, b64s: list[str], max_tokens: int = 1200) -> str:
        """Ask the multimodal model about one or more images.

        max_tokens has to cover the model's REASONING, not just its answer.
        Measured on this server (Qwen3.6-35B-A3B) on 2026-08-13: a two-image
        comparison spent 2891 completion tokens, of which ~2750 were reasoning,
        to emit a 138-token answer. At 1200 the reasoning never terminates,
        finish_reason comes back "length", and content is empty - the starvation
        retry doubles once to 2400 and still falls short, so the call returns
        nothing at all. One image is far cheaper; two need the bigger budget.
        """
        content = [{"type": "text", "text": text}]
        content += [{"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                    for b in b64s]
        try:
            data = call_model([{"role": "user", "content": content}],
                              tools=None, max_tokens=max_tokens,
                              temperature=0.1, retries=1)
        except ModelError as e:
            return f"ERROR from vision call: {e}"
        choice = data["choices"][0]
        answer = (choice["message"].get("content") or "").strip()
        if answer:
            return answer
        if choice.get("finish_reason") == "length":
            return ("(the vision model used its whole token budget on reasoning "
                    "and emitted no answer - ask a narrower question, or crop "
                    "to the detail in question with a box)")
        return "(model returned nothing)"

    def view_image(self, path: str, question: str, box: str | None = None) -> str:
        """Send an image to the multimodal model and return what it sees."""
        b64, note = self._prep_image(path, question, box)
        if b64 is None:
            return note
        answer = self._ask_vision(
            f"{question}\n\nAnswer concretely in a few sentences. "
            f"If you cannot tell, say so rather than guessing.", [b64])
        return f"[{note}]\n{answer}"

    def compare_images(self, path_a: str, path_b: str, question: str,
                       box_a: str | None = None,
                       box_b: str | None = None) -> str:
        """Put two images in ONE vision call and ask how they differ.

        Two separate view_image calls cannot do this. Each describes its own
        frame in its own words, and the difference between two descriptions is
        not a comparison - "centred" from one call and "roughly centred" from
        another says nothing about whether the two garments overlay. Judging
        placement, scale, symmetry or proportion against a reference needs both
        frames in front of the model at once.

        Both images get their own 1024px budget, so this is also sharper than
        viewing a side-by-side contact sheet, where each half arrives at ~512px.
        """
        a = self._prep_image(path_a, question, box_a)
        if a[0] is None:
            return a[1]
        b = self._prep_image(path_b, question, box_b)
        if b[0] is None:
            return b[1]
        name_a, name_b = self.resolve(path_a).name, self.resolve(path_b).name
        answer = self._ask_vision(
            f"You are given two images. The FIRST is {name_a}. The SECOND is "
            f"{name_b}.\n\n{question}\n\nAnswer concretely in a few sentences, "
            f"naming which image each observation is about. Describe only "
            f"differences you can actually see; if you cannot tell, say so "
            f"rather than guessing.", [a[0], b[0]], max_tokens=4000)
        return f"[1st: {a[1]}]\n[2nd: {b[1]}]\n{answer}"

    def finish(self, summary: str, status: str = "done") -> str:
        return json.dumps({"status": status, "summary": summary})

    def dispatch(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if fn is None or name not in {s["function"]["name"] for s in TOOL_SPECS}:
            TR.error("tool", f"no such tool '{name}'", args=json.dumps(args, default=str))
            return f"ERROR: no such tool '{name}'."
        TR.info("tool", f"call {name}", body=json.dumps(args, indent=2, default=str))
        t0 = time.time()
        try:
            out = fn(**args)
        except TypeError as e:
            out = f"ERROR: bad arguments for {name}: {e}"
        except PermissionError as e:
            out = f"REFUSED: {e}"
        except Exception as e:
            TR.exception("tool", f"{name} raised {type(e).__name__}")
            out = f"ERROR in {name}: {type(e).__name__}: {e}"
        # The result verbatim, before the model's 4000-char view of it. bash
        # already logged its own streams; this is the text the model will read.
        TR.info("tool", f"result {name}", body=out,
                secs=round(time.time() - t0, 2), chars=len(out))
        return out


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------

class Approver:
    def __init__(self, yolo: bool):
        self.yolo = yolo
        self.always = set()

    def ok(self, name: str, args: dict) -> tuple[bool, str]:
        allowed, why = self._decide(name, args)
        TR.event("INFO" if allowed else "WARN", "approval",
                 f"{name} {'allowed' if allowed else 'DENIED'}",
                 reason=why or None)
        return allowed, why

    def _decide(self, name: str, args: dict) -> tuple[bool, str]:
        if self.yolo:
            return True, "--yolo"
        if name in READONLY_TOOLS:
            return True, "read-only tool"
        if name in self.always:
            return True, "user said always"
        if not sys.stdin.isatty():
            return False, ("Denied: harness is not interactive and --yolo was not "
                           "passed, so mutating tools cannot be approved.")
        preview = args.get("command") or args.get("path") or ""
        print(c(YEL, f"\n  approve {name}?") + f"  {preview[:200]}")
        try:
            ans = input(c(DIM, "    [y]es / [n]o / [a]lways this tool / [q]uit > ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False, "Denied by user."
        TR.debug("approval", "prompted at the terminal", answer=ans or "(empty=yes)")
        if ans in ("a", "always"):
            self.always.add(name)
            return True, ""
        if ans in ("y", "yes", ""):
            return True, ""
        if ans in ("q", "quit"):
            TR.warn("approval", "user quit at the approval prompt")
            raise KeyboardInterrupt
        return False, "Denied by user. Try a different approach or ask for guidance."


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

SYSTEM = """\
You are a focused engineering agent working in a real workspace on a real \
machine. You have tools; use them. Do not describe what you would do - do it, \
then report what actually happened.

Workspace: {ws}
Python with numpy/PIL/scipy: {python}
Today: {today}

# Workspace inventory

These are the real files, fingerprinted at startup. They are the inputs you are
being asked about - not any similarly-named file elsewhere on this machine.

{inventory}

{skill_block}

# The goal

{task}

# How to work

- Take ONE step at a time. Call a tool, read the result, then decide the next step.
- Ground every claim in output you actually saw. Never report a number you did
  not measure or a file you did not create.
- NEVER rely on another program's default arguments. Pass every input and output
  path explicitly, as an absolute path. A script that defaults its inputs will
  happily process files from its own directory instead of this workspace, and
  produce real, precise, entirely wrong numbers. Matching file names and even
  matching image dimensions do NOT prove you processed the right file - check
  the fingerprints in the inventory above.
- Before you report a result, confirm it derives from the workspace inputs. If
  you produced an image, view_image it and check the subject is what the
  inventory says it should be.
- Prefer small, checkable steps over one large script. When something fails,
  read the error before changing anything.
- Numbers are not proof on their own. When the work is visual, call view_image
  and LOOK before declaring success. When the question is how one image differs
  from another, use compare_images - it puts both in a single vision call. Two
  separate view_image calls give you two independent descriptions, and the gap
  between two descriptions is not a measured difference.
- Keep tool output small. Print the few numbers you need, not whole arrays;
  pipe long output through head, grep or wc.
- write_file is for short files. Anything long - a report, a README - must be
  written with bash and a quoted heredoc, appending section by section. A big
  string argument gets truncated mid-JSON and the whole call is rejected.
- When the goal is met - or you are genuinely blocked - call finish() with a
  summary containing the concrete numbers you measured.

You have a limited context window. Be economical: it is the scarcest resource
you have, and a run that fills it ends before the work does.
"""


SKIP_DIRS = {"runs", ".git", "__pycache__", "node_modules", ".venv",
             "output", "notes",
             # 45 reference photos the agent must not choose between. The
             # reference is picked before the first turn by
             # tools/select_reference.py and installed into inputs/; listing the
             # library as well invites a second, hand-picked opinion and would
             # eat the whole inventory at 8 files per folder.
             "library_reference"}

# No single folder may occupy more than this much of the listing. A bulk asset
# directory otherwise crowds out the files the run is actually about.
PER_DIR = 8


def build_inventory(ws: Path, limit: int = 60) -> str:
    """Fingerprint the workspace so the agent can tell its inputs apart.

    This exists because of a real failure: an agent ran a neighbouring script
    with default arguments, processed that script's own same-named, same-sized
    images, and reported precise measurements of the wrong photograph. Names and
    dimensions were identical; only the content differed. Hence the md5.

    Two things make the listing useless if they are not handled, and both have
    happened:

      * Hidden DIRECTORIES were not pruned - only files whose own name began
        with a dot. `.cache/` sorts ahead of everything and its scratch files
        are not themselves hidden, so a populated cache filled all 40 slots and
        the agent's inventory contained no inputs at all. It then spent two
        turns running `ls` to find files the inventory was supposed to name.

      * One folder can still crowd out the rest. Garment_Library holds 26 PSDs
        that are not inputs to a laydown run; unbounded they take nearly half
        the listing. Hence PER_DIR, which truncates per folder and says so, so
        the agent knows more exist without being shown all of them.
    """
    import hashlib
    from collections import defaultdict

    def describe(p: Path, rel: Path) -> str:
        extra = ""
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
            try:
                from PIL import Image
                with Image.open(p) as im:
                    dims = f"{im.size[0]}x{im.size[1]}"
                digest = hashlib.md5(p.read_bytes()).hexdigest()[:8]
                extra = f"  {dims}  md5:{digest}"
            except Exception:
                extra = "  (unreadable image)"
        return f"  {str(rel):46s} {p.stat().st_size/1e6:7.2f} MB{extra}"

    # Grouped by folder so the "N more" note sits with the folder it describes
    # rather than at the end of the listing, where it reads as global.
    groups: dict[str, list[str]] = defaultdict(list)
    over: dict[str, int] = defaultdict(int)
    order: list[str] = []
    n, truncated = 0, False
    for p in sorted(ws.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(ws)
        # Any hidden component, at any depth - not just a hidden basename.
        if any(part.startswith(".") for part in rel.parts):
            continue
        if set(rel.parts) & SKIP_DIRS:
            continue
        if n >= limit:
            truncated = True
            break
        folder = str(rel.parent)
        if folder not in groups:
            order.append(folder)
        if len(groups[folder]) >= PER_DIR:
            over[folder] += 1
            continue
        groups[folder].append(describe(p, rel))
        n += 1

    rows = []
    for folder in order:
        rows += groups[folder]
        if over[folder]:
            where = "" if folder == "." else f" in {folder}/"
            rows.append(f"  … and {over[folder]} more file(s){where}")
    if truncated:
        rows.append("  … (listing truncated)")
    return "\n".join(rows) or "  (empty workspace)"


class Session:
    def __init__(self, task: str, skill_text: str, workspace: Path, run_dir: Path,
                 tools: Tools, approver: Approver, max_iters: int, verbose: bool,
                 n_ctx: int = N_CTX_FALLBACK):
        self.ws = workspace
        self.run_dir = run_dir
        self.tools = tools
        self.approver = approver
        self.max_iters = max_iters
        self.verbose = verbose
        self.n_ctx = n_ctx
        self.compact_at = int(n_ctx * COMPACT_FRACTION)
        self.prompt_tokens = 0
        self.total_completion = 0
        self.compactions = 0

        skill_block = ""
        if skill_text:
            skill_block = (
                "# Your operating manual\n\n"
                "The following skill was written from measured experience on this "
                "exact problem. Its warnings are not theoretical - each one records "
                "something that already went wrong. Follow it.\n\n"
                "<skill>\n" + skill_text + "\n</skill>"
            )

        system = SYSTEM.format(
            ws=workspace, python=tools.python,
            today=datetime.now().strftime("%Y-%m-%d"),
            inventory=build_inventory(workspace),
            skill_block=skill_block, task=task,
        )
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        self.pinned = 2  # never compact the system prompt or the goal
        self.log_path = run_dir / "transcript.jsonl"

        # The system prompt is assembled here from the skill and a fingerprint
        # of the workspace, so it differs run to run - and half of "why did it
        # do that" is answered by what it was told at the start.
        TR.info("session", "system prompt", body=system, chars=len(system))
        TR.info("session", "task", body=task)
        TR.info("session", "skill loaded" if skill_text else "no skill",
                chars=len(skill_text or ""))

    def log(self, kind: str, payload):
        with self.log_path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, "data": payload},
                               default=str) + "\n")

    # -- context management ------------------------------------------------
    def compact(self):
        """Reclaim context by eliding the oldest tool results.

        Tool output is where the context actually goes, and the oldest results
        are the ones the model has already acted on. Elide from the front until
        we are back under budget; the full text stays on disk either way.
        """
        target = self.compact_at * 0.6
        est = self.prompt_tokens
        freed = 0
        for m in self.messages[self.pinned:]:
            if est - freed < target:
                break
            if m.get("role") == "tool" and not m.get("_elided"):
                freed += len(m.get("content", "")) // 3.5
                m["content"] = "[earlier tool output elided to reclaim context - " \
                               "re-run the command if you need it again]"
                m["_elided"] = True
        self.compactions += 1
        self.log("compact", {"freed_est": int(freed), "prompt_tokens": self.prompt_tokens})
        TR.warn("context", "compacted history",
                freed_est=int(freed), prompt_tokens=self.prompt_tokens,
                compact_at=self.compact_at, compactions=self.compactions,
                elided_msgs=sum(1 for m in self.messages if m.get("_elided")))
        print(c(DIM, f"  · compacted context (~{int(freed)} tokens reclaimed)"))

    def history(self):
        """History as the API wants it - private bookkeeping keys stripped."""
        return [{k: v for k, v in m.items() if not k.startswith("_")}
                for m in self.messages]

    # -- main loop ---------------------------------------------------------
    def delivered(self) -> bool:
        """Is there anything in output/ yet."""
        return any((self.run_dir / "output").glob("pick*.png"))

    def run(self) -> dict:
        result = {"status": "max_iters", "summary": "Hit the iteration cap."}
        warned_late = False

        for i in range(1, self.max_iters + 1):
            if self.prompt_tokens > self.compact_at:
                self.compact()

            # Three turns out, with nothing delivered, the run is told plainly
            # that it is nearly over. A real run spent its last thirty turns
            # appealing construction flags image by image and reached the cap
            # with an empty output/ and ten paid-for images in archive/. It did
            # not run out of information; it ran out of turns while deciding.
            left = self.max_iters - i + 1
            if left <= 3 and not warned_late and not self.delivered():
                warned_late = True
                TR.warn("session", "delivery warning injected", turns_left=left)
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"{left} turn(s) left before the run is cut off, and "
                        f"output/ is still empty. Stop investigating and "
                        f"deliver now:\n"
                        f"  cd {self.ws}/tools && {self.tools.python} "
                        f"grade_flats.py --run {self.run_dir} --ship-faithful 4\n"
                        f"then write LOG.md and call finish(). It ships the "
                        f"most faithful candidates first and prints exactly "
                        f"what each one carries, so an imperfect batch is "
                        f"still a delivery with its costs named. Shipping four "
                        f"flagged images with the flags written down beats "
                        f"shipping nothing."),
                })

            TR.rule(f"turn {i}/{self.max_iters}")
            TR.info("turn", "calling the model", messages=len(self.messages),
                    prompt_tokens=self.prompt_tokens, n_ctx=self.n_ctx)
            print(c(BOLD, f"\n[{i}/{self.max_iters}]") + c(DIM, "  thinking..."), end="", flush=True)
            t0 = time.time()
            try:
                data = call_model(self.history(), TOOL_SPECS)
            except ModelError as e:
                TR.error("turn", "model call failed; ending the run", body=str(e))
                print()
                print(c(RED, f"  model error: {e}"))
                return {"status": "error", "summary": str(e)}
            dt = time.time() - t0

            choice = data["choices"][0]
            msg = choice["message"]
            usage = data.get("usage", {})
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            self.total_completion += usage.get("completion_tokens", 0)

            pct = 100 * self.prompt_tokens / self.n_ctx
            print("\r" + " " * 40 + "\r", end="")
            print(c(BOLD, f"[{i}/{self.max_iters}]") +
                  c(DIM, f"  {dt:.0f}s · ctx {self.prompt_tokens}/{self.n_ctx} ({pct:.0f}%)"))

            self.log("assistant", {"message": msg, "usage": usage})

            think = msg.get("reasoning_content") or msg.get("reasoning")
            TR.info("turn", f"reply in {dt:.1f}s",
                    finish=choice.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    ctx_pct=f"{pct:.0f}%")
            # Reasoning is dropped from the conversation, so unless it is written
            # here it is gone - and it is where the model's actual decision was
            # made. --verbose shows 600 characters of it; the trace keeps it all.
            if think:
                TR.debug("reasoning", f"turn {i}", body=think)
            if msg.get("content"):
                TR.info("assistant", f"turn {i} content", body=msg["content"])
            if self.verbose and think:
                print(c(DIM, textwrap.indent(
                    textwrap.shorten(think, 600), "    · ")))

            text = (msg.get("content") or "").strip()
            if text:
                print(textwrap.indent(text, "  "))

            calls = msg.get("tool_calls") or []

            # Reasoning is dropped from history on purpose - see module docstring.
            self.messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                **({"tool_calls": calls} if calls else {}),
            })

            if calls:
                TR.info("turn", f"{len(calls)} tool call(s)",
                        names=",".join(c["function"]["name"] for c in calls))

            if not calls:
                TR.warn("turn", "no tool call" + ("" if text else " and no text"))
                # No tool, no text: a starved turn. Nudge rather than spin.
                if not text:
                    self.messages.append({
                        "role": "user",
                        "content": "You returned an empty reply. Take the next "
                                   "concrete step with a tool, or call finish().",
                    })
                    continue
                self.messages.append({
                    "role": "user",
                    "content": "Continue with the next tool call, or call finish() "
                               "if the goal is met.",
                })
                continue

            for call in calls:
                fn = call["function"]
                name = fn["name"]
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except Exception as e:
                    out = (f"ERROR: could not parse your tool arguments as JSON "
                           f"({e}). Send valid JSON.")
                    TR.error("tool", f"unparseable arguments for {name}",
                             body=str(raw), error=str(e))
                    self._tool_result(call, out)
                    print(c(RED, f"  ! bad tool arguments for {name}"))
                    continue

                self._show_call(name, args)

                if name == "finish":
                    result = {"status": args.get("status", "done"),
                              "summary": args.get("summary", "")}
                    self.log("finish", result)
                    TR.info("session", f"finish({result['status']}) on turn {i}",
                            body=result["summary"])
                    return result

                allowed, why = self.approver.ok(name, args)
                out = why if not allowed else self.tools.dispatch(name, args)

                full = out
                if len(out) > MAX_TOOL_CHARS:
                    sp = self.tools.spool(name, full)
                    head = out[: MAX_TOOL_CHARS // 2]
                    tail = out[-MAX_TOOL_CHARS // 2:]
                    out = (f"{head}\n\n... [{len(full) - MAX_TOOL_CHARS} chars elided; "
                           f"full output at {sp}] ...\n\n{tail}")

                self.log("tool", {"name": name, "args": args, "result": full[:20000]})
                if len(out) != len(full):
                    TR.debug("tool", f"{name} result truncated for the model",
                             full_chars=len(full), sent_chars=len(out))
                self._show_result(out)
                self._tool_result(call, out)

        TR.warn("session", "hit the iteration cap without finish()",
                max_iters=self.max_iters)
        return result

    def _tool_result(self, call, content: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "content": content,
        })

    def _show_call(self, name: str, args: dict):
        if name == "bash":
            detail = args.get("command", "")
        elif name == "finish":
            detail = args.get("status", "")
        elif name == "view_image":
            detail = f"{args.get('path','')} {args.get('box') or ''} - {args.get('question','')}"
        else:
            detail = args.get("path", "")
        detail = " ".join(str(detail).split())
        print(c(CYA, f"  → {name}") + f"  {detail[:160]}")

    def _show_result(self, out: str):
        lines = out.splitlines()
        shown = lines[:8]
        for l in shown:
            print(c(DIM, "    " + l[:160]))
        if len(lines) > len(shown):
            print(c(DIM, f"    … {len(lines) - len(shown)} more lines"))


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def resolve_skill(workspace: Path, name: str | None, skill_file: str | None) -> Path | None:
    """Find a SKILL.md.

    Order: an explicit --skill-file; skills/<name>/SKILL.md; <name> as a path;
    and finally a SKILL.md sitting in the workspace root, which is how a skill
    that lives beside the work it describes gets picked up automatically.
    """
    if skill_file:
        p = Path(skill_file).expanduser()
        if not p.is_absolute():
            p = workspace / p
        if not p.exists():
            raise SystemExit(f"No such skill file: {p}")
        return p
    if name:
        for cand in (workspace / "skills" / name / "SKILL.md",
                     Path(name).expanduser(),
                     workspace / name):
            if cand.exists() and cand.is_file():
                return cand
        avail = sorted(d.name for d in (workspace / "skills").glob("*") if d.is_dir())
        raise SystemExit(f"No skill '{name}'. Available: {', '.join(avail) or '(none)'}")
    root = workspace / "SKILL.md"
    return root if root.exists() else None


def load_skill(p: Path) -> tuple[str, str]:
    """Return (skill_text, description) from a SKILL.md."""
    text = p.read_text()

    # Pull `description:` out of the frontmatter to seed a default task.
    desc = ""
    m = re.search(r"^---\s*\n(.*?)\n---", text, re.S)
    if m:
        d = re.search(r'^description:\s*"?(.*?)"?\s*$', m.group(1), re.M | re.S)
        if d:
            desc = " ".join(d.group(1).split())
    return text, desc


DEFAULT_TASK = (
    "Accomplish this skill's goal in the workspace, end to end.\n\n"
    "Start by orienting yourself: list the workspace, identify the input images, "
    "and check whether a working implementation already exists that the skill "
    "refers to. Read before you write.\n\n"
    "Work incrementally and verify each stage with measurements before moving on. "
    "Anything that costs money must be justified by a measurement first - say what "
    "you are about to spend and why before you spend it.\n\n"
    "Write your outputs to a new timestamped folder under runs/ and finish by "
    "reporting the numbers you measured."
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def preflight() -> tuple[str, int]:
    """Confirm the server is reachable; return (model id, context window).

    vLLM reports `max_model_len` per model on /v1/models; llama.cpp does not,
    hence the fallback. Reading it beats a constant: this project has already
    run a 262144-token server while every budget in the file said 32768.
    """
    TR.info("preflight", "GET /v1/models", url=f"{BASE_URL}/v1/models")
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/models",
                                     headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        TR.debug("preflight", "models response",
                 body=json.dumps(data, indent=2, default=str))
        m = data["data"][0]
        n_ctx = int(m.get("max_model_len") or N_CTX_FALLBACK)
        TR.info("preflight", "server ready", model=m["id"], n_ctx=n_ctx,
                from_server=bool(m.get("max_model_len")))
        return m["id"], n_ctx
    except SystemExit:
        raise
    except Exception as e:
        TR.error("preflight", "cannot reach the model server",
                 error=f"{type(e).__name__}: {e}")
        raise SystemExit(
            f"Cannot reach the model server at {BASE_URL}: {e}\n"
            f"Start it, or set QWEN_BASE_URL (with or without a /v1 suffix)."
        )


def pre_clean(workspace: Path, run_dir: Path, python: str) -> Path | None:
    """Step 0a: erase the tag and drop the background BEFORE the reference is
    picked.

    The matcher scores the off-set photo against library flats that are all
    clean studio plates. Handed the raw phone photo it is scoring a garment plus
    a room plus a hang tag against a garment on white, and every point that
    costs comes off the one comparison the whole run is anchored to.

    Cleaning used to live inside prepare.py, which the agent runs on its first
    turn - after step 0 had already matched. Same work, wrong order. It runs
    here now and prepare.py verifies the result instead of redoing it, so there
    is one cleaning path and the total spend is unchanged.

    Returns (cleaned image or None, gate failures). None is survivable: the
    caller falls back to the raw input and says so. A non-empty failure list is
    NOT survivable and parks the run - see the caller.
    """
    # segment.py, not clean.py. The fal object-removal endpoint this step used
    # to call has been restricted since 2026-08-27 and every run crashed here.
    # The self-hosted segmentation service does the half that is available -
    # background dropped, garment on a white plate - and does not remove a tag,
    # ticket or pin. Those are named by describe.py under TO REMOVE and asked
    # for in the prompt instead.
    #
    # No outline gate runs against a segmentation, so this returns no failures.
    # The gate existed to catch a generative eraser silently redrawing the
    # garment; a segmenter either finds the garment or does not, and segment.py
    # checks that itself before writing anything.
    script = workspace / "tools" / "segment.py"
    out = run_dir / "archive" / "offset_upload.jpg"
    if not script.exists():
        TR.warn("step0", f"no {script}; matching against the raw input")
        return None, []
    print(c(BOLD, "\nstep 0a · segment") +
          c(DIM, "  (background dropped, so the reference is matched against "
                 "the clean image)"), flush=True)
    TR.rule("step 0a - segment")
    with TR.console_component("step0"):
        rc = stream_subprocess([python, str(script), "--run", str(run_dir)],
                               cwd=script.parent, comp="step0")
    if rc == 0 and out.exists():
        TR.info("step0", "segmentation ok", out=str(out))
        return out, []

    # Which kind of failure this is decides whether the run may continue, so
    # read the audit rather than the exit code alone. A gate failure means the
    # cleaned image is not the same garment as the photograph; anything else
    # (no key, no network, a crash) leaves the raw input usable.
    fails = []
    audit = run_dir / "archive" / "clean_audit.json"
    if audit.exists():
        try:
            attempts = json.loads(audit.read_text()).get("attempts") or []
            fails = list((attempts[-1] if attempts else {}).get("outline_fails") or [])
        except (json.JSONDecodeError, OSError):
            fails = []
    if fails:
        TR.error("step0", "pre-clean gate failed", exit_code=rc, fails=fails)
        return (out if out.exists() else None), fails
    TR.warn("step0", "pre-clean failed; matching against the raw input",
            exit_code=rc, out_exists=out.exists())
    print(c(YEL, f"  pre-clean failed (exit {rc}); the reference will be "
                 f"matched against the raw input."))
    return None, []


def force_ship(workspace: Path, run_dir: Path, python: str, n: int) -> bool:
    """Deliver from what is already generated, when the agent did not.

    The backstop for the failure this harness has actually produced: a run that
    reached its iteration cap with ten paid-for images in archive/ and nothing
    in output/. Every one of those images was billed, graded and then
    abandoned, because the last turn arrived while the agent was still
    arbitrating between a grade and a flag.

    Nothing here is a judgement call. grade_flats.py --ship-faithful picks the
    candidates stage 3 found intact, backfills by grade only if there are too
    few, and prints what each pick carries. If the agent already shipped, this
    does nothing at all.

    The expected-changes declaration is carried over from the agent's own last
    grading pass, so the forced delivery reproduces the judgement the run
    already made rather than taking a fresh and different one.
    """
    outd, arch = run_dir / "output", run_dir / "archive"
    if any(outd.glob("pick*.png")):
        return False
    if not list(arch.glob("cand_*.png")):
        return False
    script = workspace / "tools" / "grade_flats.py"
    if not script.exists():
        return False

    expected = ""
    try:
        expected = str(json.loads((arch / "metrics.json").read_text())
                       .get("expected_changes") or "")
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        expected = ""

    cmd = [python, str(script), "--run", str(run_dir), "--ship-faithful", str(n)]
    if expected:
        cmd += ["--expected-changes", expected]
    print(c(YEL, f"\nthe run ended without delivering, and "
                 f"{len(list(arch.glob('cand_*.png')))} generated image(s) are "
                 f"sitting in archive/."))
    print(c(DIM, f"  shipping the {n} most faithful of them, deterministically. "
                 f"This is the harness, not the agent."))
    TR.warn("harness", "forcing delivery after an undelivered run",
            n=n, expected_changes=expected or None)
    with TR.console_component("ship"):
        rc = stream_subprocess(cmd, cwd=script.parent, comp="ship")
    shipped = any(outd.glob("pick*.png"))
    TR.info("harness", "forced delivery finished", exit_code=rc, shipped=shipped)
    if not shipped:
        print(c(RED, "  nothing shipped - read the output above; output/ is "
                     "still empty."))
    return shipped


def select_reference(workspace: Path, run_dir: Path, python: str,
                     category: str | None, threshold: float,
                     query: Path | None = None,
                     silhouette: bool = False) -> int:
    """Step 0: install the lay reference, before the agent gets a turn.

    This is deliberately not the agent's job. Choosing the reference is a
    judgement with one right answer per garment, it is worth several turns and
    a lot of context if done conversationally, and it is the input everything
    downstream is measured against - so it happens once, deterministically,
    and lands in the workspace inventory as a plain fact by the time the model
    reads it.

    Returns the child's exit code: 0 installed, 2 no match, 1 broke.
    """
    script = workspace / "tools" / "select_reference.py"
    if not script.exists():
        TR.warn("step0", f"no {script}; skipping reference selection")
        print(c(YEL, f"  no {script.name}; skipping reference selection"))
        return 0
    cmd = [python, str(script), "--run", str(run_dir),
           "--threshold", str(threshold)]
    if query:
        cmd += ["--query", str(query), "--query-cleaned"]
    if category:
        cmd += ["--category", category]
    if silhouette:
        # Installs a line drawing of the winner instead of its photograph.
        # Without this passed through, step 0 rewrites inputs/ on every run and
        # an outline installed by hand survives exactly until the next one.
        cmd += ["--silhouette"]
    # Flushed: the child writes straight to this terminal, so an unflushed
    # header appears after everything it was supposed to introduce.
    print(c(BOLD, "\nstep 0 · reference selection") +
          c(DIM, "  (deterministic, before the agent starts)"), flush=True)
    TR.rule("step 0 - reference selection")
    # Read through us rather than letting the child inherit fd 1, or its output
    # is the one part of a run the trace cannot see.
    with TR.console_component("step0"):
        rc = stream_subprocess(cmd, cwd=script.parent, comp="step0")
    prov = run_dir / "reference_selection.json"
    if prov.exists():
        TR.info("step0", "reference_selection.json", body=prov.read_text())
    return rc


def main():
    ap = argparse.ArgumentParser(
        description="A very small Claude-Code-style agent loop on a local model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              ./run.sh --skill laydown-match
              ./run.sh --skill laydown-match --task "Run stage 1 only, no API calls"
              ./run.sh --task "Summarise this folder" --max-iters 5 --yolo
              ./run.sh --reference-only          # step 0 only: can the library
                                                 # serve this garment? 0 yes,
                                                 # 20 upload a hero
        """),
    )
    ap.add_argument("--skill", help="Skill name under skills/<name>/SKILL.md, or a path.")
    ap.add_argument("--skill-file", help="Explicit path to a SKILL.md.")
    ap.add_argument("--task", help="What to do. Defaults to the skill's own goal.")
    ap.add_argument("--workspace", type=Path, default=HERE)
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--yolo", action="store_true",
                    help="Run tools without asking. You are trusting the model with a shell.")
    ap.add_argument("--allow-outside", action="store_true",
                    help="Permit writes outside the workspace.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show a slice of the model's reasoning each turn.")
    ap.add_argument("--no-reference-select", action="store_true",
                    help="Skip step 0 and run against whatever reference is "
                         "already sitting in inputs/.")
    # Back ON by default. It was switched off on 2026-08-27 when the fal
    # object-removal endpoint started answering 403 and every run crashed here;
    # step 0a now calls the self-hosted segmentation service instead, which is
    # reachable and costs nothing.
    ap.add_argument("--no-pre-clean", action="store_true",
                    help="Skip step 0a. The reference is then matched against "
                         "the raw input, background and all, and the generator "
                         "is handed a photo that still has the room in it.")
    ap.add_argument("--reference-outline", action="store_true",
                    help="Install a line drawing of the reference instead of "
                         "its photograph. The lay reference is the second image "
                         "sent to the generator, and while it is a photograph "
                         "of a garment the model sometimes draws that garment "
                         "too - roughly 2 to 3 candidates in 10 come back "
                         "holding two. An outline carries the pose and contains "
                         "no garment to copy.")
    ap.add_argument("--ship-on-cap", type=int, default=4, metavar="N",
                    help="If the run ends with output/ empty and generated "
                         "images in archive/, the harness ships the N most "
                         "faithful of them itself (grade_flats.py "
                         "--ship-faithful N) rather than abandoning paid-for "
                         "images. 0 turns it off. The agent is also warned "
                         "three turns before the cap.")
    ap.add_argument("--allow-dirty-source", action="store_true",
                    help=f"Start the agent even when the pre-clean OUTLINE GATE "
                         f"failed - that is, when the cleaned image is missing "
                         f"part of the garment. Without this the run stops "
                         f"before the agent and exits {EXIT_UNCLEAN_SOURCE}, "
                         f"because every image generated from that source "
                         f"inherits the loss.")
    ap.add_argument("--reference-category",
                    help="Force the library subfolder step 0 searches "
                         "(e.g. bras, leggings) instead of letting the "
                         "off-set photo's own garment type pick it.")
    # 95: an operator decision to take the near-miss out of the reference slot
    # entirely. A wrong reference is not a cheap error - it is the image every
    # candidate is laid against, and on this project it has bled a V-neckline
    # and seam piping into four of ten generations at 15c each. Parking the run
    # and asking for a hero costs nothing by comparison.
    #
    # What it gives up, so the trade is on the record rather than rediscovered.
    # It was 87 for a measured reason: a periwinkle sports bra scored 89.2
    # against a library piece that agreed on garment type, neckline, strap style
    # and finish and differed only in COLOUR - the heaviest term in the score
    # (weight 2.0) - while select_reference.py desaturates the winner before
    # installing it. At 95 that match is refused on the one attribute the
    # pipeline throws away.
    #
    # And the score is not as stable as a 95 gate assumes. The same library
    # image scored 99.2 against one cleaned copy of a garment and 81.6 against
    # another copy of the SAME garment, because the query's attributes are
    # re-extracted per image. One real batch clears 95 with 4.2 points to
    # spare; another clears it by 0.2. Expect no-match parks, and read
    # result_top_matches.jpg before assuming the library has nothing.
    #
    # Only reference SELECTION moves. match_reference.py keeps 90 for a direct
    # call, where the caller is asking "is this the same garment?" rather than
    # "is this close enough in shape to lay against?".
    ap.add_argument("--reference-threshold", type=float, default=95.0,
                    help="Score a library image must reach to be installed as "
                         "the reference (default 95). Lower it to 87 to accept "
                         "a construction-identical, wrong-colour match.")
    ap.add_argument("--allow-no-reference", action="store_true",
                    help="Start the agent even when step 0 found no matching "
                         "reference. Off by default: the run would be measured "
                         "against a reference for a different garment.")
    ap.add_argument("--reference-only", action="store_true",
                    help="Run step 0 and stop: pre-clean, match, install the "
                         "reference, write reference_selection.json, never "
                         "start the agent. This is the cheap gate a caller runs "
                         "first to find out whether the library can serve this "
                         "garment at all - exit 0 if it can, "
                         f"{EXIT_NO_REFERENCE} if a human has to upload a hero. "
                         "No fal spend and no text model needed either way.")
    ap.add_argument("--trace-level", default="DEBUG",
                    choices=["DEBUG", "INFO", "WARN", "ERROR"],
                    help="Detail written to runs/<session>/run.log. DEBUG (the "
                         "default) keeps the model's reasoning and every HTTP "
                         "payload; INFO keeps the console, tools and results.")
    ap.add_argument("--no-trace", action="store_true",
                    help="Do not write runs/<session>/run.log.")
    args = ap.parse_args()

    # Before anything else that prints or calls out: the trace has nowhere to
    # live until the run folder is stamped, so early events are buffered and
    # flushed by attach() below.
    if args.no_trace:
        TR.enabled = False
    else:
        TR.tee()
    TR.info("harness", "invoked", argv=" ".join(sys.argv[1:]), pid=os.getpid(),
            cwd=os.getcwd(), python=sys.executable)

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise SystemExit(f"No such workspace: {workspace}")

    skill_path = resolve_skill(workspace, args.skill, args.skill_file)
    skill_text, skill_desc = load_skill(skill_path) if skill_path else ("", "")

    if args.reference_only and args.no_reference_select:
        raise SystemExit("--reference-only and --no-reference-select cancel "
                         "each other out: the first runs nothing but step 0 "
                         "and the second is what skips step 0.")

    task = args.task or (DEFAULT_TASK if skill_path else None)
    # A gate run never reaches the agent, so it does not need to be told what
    # the agent would have done.
    if not task and not args.reference_only:
        raise SystemExit("Nothing to do: pass --task and/or --skill.")
    if skill_path and not args.task and skill_desc:
        task = f"{skill_desc}\n\n{task}"

    # One run means one folder, and run.sh stamps it. Deriving a second stamp
    # here put the transcript and the pipeline's own steps.log in folders that
    # only agreed because they were a second apart; set it either way so that
    # calling harness.py directly still gives the tools one folder to share.
    stamp = os.environ.get("LAYDOWN_SESSION") or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["LAYDOWN_SESSION"] = stamp
    run_dir = workspace / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Attached before preflight, not after: a run that dies because the server
    # is unreachable is exactly the one someone wants a trace of, and until
    # there is a file everything is only buffered in memory.
    if TR.enabled:
        TR.attach(run_dir / "run.log", level=args.trace_level, header={
            "session": stamp,
            "argv": " ".join(sys.argv[1:]),
            "workspace": workspace,
            "skill": skill_path or "(none)",
            "server": BASE_URL,
            "python": sys.executable,
            "pid": os.getpid(),
        })
        TR.info("harness", "options", **{
            k: v for k, v in vars(args).items() if k != "task"})

    # Skipped for a gate run, and deliberately: step 0 talks to the vision
    # server (REFMATCH_BASE_URL), not this one, so a gate that preflighted the
    # text model would report "no hero" as unreachable-server breakage on a day
    # the agent was never going to run anyway.
    model, n_ctx = (None, N_CTX_FALLBACK) if args.reference_only else preflight()

    tools = Tools(workspace, run_dir, args.allow_outside)
    TR.info("harness", "child interpreter for pipeline scripts", python=tools.python)

    # Step 0, before the Session is built: the inventory in the system prompt is
    # fingerprinted at construction time, so the reference has to be on disk by
    # then or the agent is told about the previous garment's one.
    if not args.no_reference_select:
        # Clean first, match second. The order is the point: the matcher has to
        # see the same image the rest of the pipeline works from.
        clean, clean_fails = ((None, []) if args.no_pre_clean else
                              pre_clean(workspace, run_dir, tools.python))

        # A failed outline gate parks the run HERE, before the agent gets a
        # turn and before anything is billed. It used to be a printed warning
        # that nothing acted on: one run's gate reported the bottom edge of the
        # garment pulled in 16.7%, the pipeline carried on, and 150 cents of
        # images were generated from a source the pipeline had already
        # rejected. Everything downstream then inherited it - the description,
        # the prompt, and a construction check that flagged all ten candidates
        # for correctly dropping the pins the clean had failed to remove.
        if clean_fails and not args.allow_dirty_source:
            print(c(RED, "\npre-clean gate FAILED - the cleaned image is not "
                         "the same garment."))
            for f in clean_fails:
                print(c(DIM, f"    {f}"))
            print(c(DIM, f"  audit     {run_dir / 'archive' / 'clean_audit.json'}"))
            print(c(DIM, f"  cleaned   {run_dir / 'archive' / 'offset_upload.jpg'}"
                         f"  (written so it can be looked at)"))
            print(c(DIM, "  The run stops before the agent starts, because "
                         "every image generated from this source would inherit "
                         "the loss and every check downstream would agree the "
                         "garment always looked like this."))
            print(c(DIM, "  --allow-dirty-source runs anyway; "
                         "--no-pre-clean skips the clean entirely."))
            return EXIT_UNCLEAN_SOURCE
        if clean_fails:
            print(c(YEL, "\n  --allow-dirty-source: continuing from a source "
                         "the outline gate rejected."))
            for f in clean_fails:
                print(c(DIM, f"    {f}"))
        rc = select_reference(workspace, run_dir, tools.python,
                              args.reference_category, args.reference_threshold,
                              query=clean, silhouette=args.reference_outline)
        # 2 and 1 are different answers and must not leave here as the same
        # code. 2 is "the library has nothing close enough, a human has to
        # upload a hero" - the pipeline worked and gave its verdict. 1 is the
        # matcher itself breaking. Collapsing them meant a scheduler could only
        # see "the run stopped", so a correct verdict sat in the failure list
        # next to an unreachable model server, and the list stopped being read.
        if rc != 0 and not args.allow_no_reference:
            business = rc == 2
            (TR.warn if business else TR.error)(
                "step0", "no reference installed; stopping before the agent",
                exit_code=rc, outcome="no_reference" if business else "error")
            if business:
                print(c(YEL, "\nno reference: the library has nothing close "
                             "enough to this garment."))
                print(c(DIM, "  This is an answer, not a failure. Someone has "
                             "to upload a hero for this style; the run stops "
                             f"here and exits {EXIT_NO_REFERENCE}."))
                print(c(DIM, "  reference_selection.json records the closest "
                             "the library came, and result_top_matches.jpg "
                             "shows it."))
            else:
                print(c(RED, "\nstopping before the agent starts."))
                print(c(DIM, "  Reference selection broke; nothing was "
                             "installed, so every measurement downstream would "
                             "be against the wrong garment."))
            print(c(DIM, "  --allow-no-reference runs anyway; "
                         "--no-reference-select uses what is already in inputs/;"))
            print(c(DIM, "  --reference-category / --reference-threshold widen "
                         "the search."))
            return EXIT_NO_REFERENCE if business else 1
        if rc != 0:
            print(c(YEL, "\n  --allow-no-reference: starting with whatever "
                         "reference inputs/ already holds."))

    # The gate stops here. Everything above is deterministic and cheap; the
    # agent below is neither, and a caller that only wants to know whether the
    # library can serve this garment should not have to pay for it to find out.
    if args.reference_only:
        prov = run_dir / "reference_selection.json"
        TR.info("step0", "--reference-only: stopping after step 0",
                receipt=str(prov), exists=prov.exists())
        print(c(BOLD, "\n--reference-only") +
              c(DIM, "  the agent was not started."))
        if prov.exists():
            r = json.loads(prov.read_text())
            if r.get("match_found"):
                print(c(GRN, f"  reference {Path(r['installed']).name}  <- "
                             f"{Path(r['source']).name}  ({r['score']}/100)"))
            else:
                closest = r.get("closest") or {}
                print(c(YEL, f"  no reference: closest was "
                             f"{closest.get('file', '?')} at "
                             f"{(closest.get('score') or 0):.1f}, needed "
                             f"{(r.get('threshold') or 0):.0f}"))
        print(c(DIM, f"  receipt   {prov}"))
        return 0

    approver = Approver(args.yolo)
    sess = Session(task, skill_text, workspace, run_dir, tools, approver,
                   args.max_iters, args.verbose, n_ctx)

    print(c(BOLD, "qwen harness"))
    print(c(DIM, f"  model     {model}"))
    print(c(DIM, f"  context   {n_ctx} tokens (compacting past {sess.compact_at})"))
    print(c(DIM, f"  server    {BASE_URL}"))
    print(c(DIM, f"  workspace {workspace}"))
    print(c(DIM, f"  skill     {skill_path or '(none)'}"))
    print(c(DIM, f"  run       {run_dir}"))
    prov = run_dir / "reference_selection.json"
    if prov.exists():
        r = json.loads(prov.read_text())
        # The receipt exists on a miss too, with nulls where the reference would
        # be. Only --allow-no-reference gets this far with one, and that run is
        # exactly the one that has to say out loud what it is laying against.
        if r.get("installed"):
            print(c(DIM, f"  reference {Path(r['installed']).name}  <- "
                         f"{Path(r['source']).name}  ({r['score']}/100)"))
        else:
            print(c(YEL, "  reference NONE - step 0 found no match and the run "
                         "was allowed to start anyway"))
    print(c(DIM, f"  approval  {'OFF (--yolo)' if args.yolo else 'on'}"))
    sess.log("start", {"task": task, "skill": args.skill, "model": model})

    t0 = time.time()
    try:
        result = sess.run()
    except KeyboardInterrupt:
        TR.warn("harness", "interrupted by the user (Ctrl-C)")
        print(c(YEL, "\n\ninterrupted."))
        result = {"status": "interrupted", "summary": "User interrupted the run."}
    except BaseException as e:
        TR.exception("harness", f"run aborted: {type(e).__name__}: {e}")
        raise

    dt = time.time() - t0

    # Whatever the agent decided, paid-for images do not get abandoned.
    forced = False
    if args.ship_on_cap:
        forced = force_ship(workspace, run_dir, tools.python, args.ship_on_cap)

    colour = {"done": GRN, "blocked": YEL}.get(result["status"], RED)
    print("\n" + c(BOLD, "─" * 60))
    print(c(colour, f"{result['status'].upper()}") +
          c(DIM, f"  ·  {dt/60:.1f} min  ·  {sess.total_completion} tokens generated"
                 f"  ·  {sess.compactions} compactions"))
    if result.get("summary"):
        print("\n" + textwrap.indent(textwrap.fill(result["summary"], 76), "  "))
    if forced:
        print(c(YEL, "\n  output/ was written by the harness, not by the agent. "
                     "Nothing in the run's own LOG.md describes these picks - "
                     "read grade_flats.py's output above for what each carries."))
    print(c(DIM, f"\n  transcript: {sess.log_path}"))
    if TR.path:
        print(c(DIM, f"  trace:      {TR.path}"))

    sess.log("end", {"result": result, "seconds": dt})
    TR.info("harness", f"run finished: {result['status']}",
            body=result.get("summary"), seconds=round(dt, 1),
            completion_tokens=sess.total_completion,
            compactions=sess.compactions)
    return 0 if result["status"] == "done" else 1


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        # argparse and the preflight both leave this way; the message is the
        # whole story of a run that never started, so it belongs in the trace.
        TR.error("harness", f"exit: {e}")
        TR.close("exited before completing")
        raise
    except BaseException as e:
        TR.exception("harness", f"unhandled {type(e).__name__}: {e}")
        TR.close("crashed")
        raise
    TR.close(f"exit code {rc}")
    sys.exit(rc)
