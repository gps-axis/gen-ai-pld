#!/usr/bin/env python3
"""
harness.py - a small agent loop that re-lays a garment towards a reference.

One job. The model is handed two images - the segmented off-set photo and the
reference laydown - and iterates towards the second using three real capabilities:
a segmentation service, a fal.ai image editor, and its own eyes. It writes and
edits the prompt itself, chooses what each generation starts from, and stops when
it judges the result good enough or when the image budget runs out.

    ./run.sh --yolo
    ./run.sh --source inputs/off_set_image.jpg --reference inputs/ref.jpg --max-images 6
    ./run.sh --max-images 0 --yolo      # free: exercises the loop, buys nothing

The loop is deliberately not a pipeline. Every tool is available on every turn,
in any order and any number of times: segmenting a GENERATED candidate and then
generating again from it is a legitimate move, and so is throwing the prompt away
and starting it over on turn nine.

Design notes worth knowing before you edit this file:

  * Images are conversation, not tool output. Turn 1 pins the source and the
    reference into the first user message, and every new candidate is appended as
    an image with its measurements underneath - so the model sees what it bought
    without spending a turn asking. See Session.attach_image().

  * That would eat the window, so candidate images are elided every turn, not on
    a token threshold - see Session.elide_images() and the comment there for why
    unconditional matters.

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
# The reference filenames come from common.py rather than being spelled again
# here. They are read by five tools and written by two, and a name mirrored into
# each of them is a name that drifts - the harness would pin one file into turn 1
# while generate.py sent fal a different one.
from common import (REFERENCE_GREY, REFERENCE_ORIGINAL,  # noqa: E402
                    SOURCE_ORIGINAL, heif_to_jpeg, is_heif,
                    prepare_reference_image, reference_path)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent


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


_DOTENV = load_dotenv(HERE / ".env")


def _conf(key: str, default: str = "") -> str:
    """A setting from the environment, else .env, else the default.

    The endpoint settings used to be environment-only, so pointing the harness
    at a different server meant exporting three variables in every shell that
    ran it - while FAL_KEY, sitting in the same .env, was picked up for free.
    An explicit export still wins, so a one-off override is unchanged.
    """
    v = os.environ.get(key)
    return v if v not in (None, "") else _DOTENV.get(key, default)


def _base_url() -> str:
    """The server root, without the /v1 suffix this file appends itself.

    The endpoint gets pasted around in both forms - vision.py wants it WITH
    /v1, this file builds /v1/chat/completions itself - so accept either and
    normalise, rather than producing /v1/v1/chat/completions from a perfectly
    reasonable QWEN_BASE_URL.
    """
    url = _conf("QWEN_BASE_URL", "http://10.11.245.41:8091").rstrip("/")
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


BASE_URL = _base_url()
API_KEY = _conf("QWEN_API_KEY", "pick-a-long-secret-string")

# The name to route on. A single-model llama.cpp/vLLM server ignores the field
# and serves whatever it loaded, so this file used to omit it. A proxy cannot:
# LiteLLM fronts many models and 400s "Invalid model name passed in model=None"
# on every request without it. preflight() overwrites this with the id the
# server actually reports, so the default only has to survive until then.
MODEL = _conf("QWEN_MODEL")

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

# fal.ai images for a whole run. The ceiling, not a target: the model spends what
# it needs and stops. Read from the environment so the container and run.sh can
# set it in one place; --max-images lowers it per run.
DEFAULT_BUDGET = int(os.environ.get("LAYDOWN_MAX_IMAGES", "10"))

# "The library has nothing close enough to this garment." An expected answer,
# not a crash: it means step 0 searched inputs/reference_library, found nothing
# that could serve as this garment's lay reference, and stopped before a single
# billed image. The fix is a human supplying a reference or adding to the
# library, which is why it is not 1.
#
# It was unreachable for a while - the reference became operator-supplied and
# there was no search left to fail - and both DOCKER.md and docker-entrypoint.sh
# said so. tools/select_reference.py brings the search back, so this can occur
# again and those notes have been corrected.
EXIT_NO_REFERENCE = 20

# "The segmentation gate rejected the source": the run worked, and what it
# produced is a refusal to spend money on a source it can prove is damaged.
# Distinct from 1 because the fix is different - this one needs the input photo
# or the segmenter looked at, not the model server.
EXIT_UNCLEAN_SOURCE = 21

# Tools that only observe or cost nothing. These never prompt for approval.
READONLY_TOOLS = {"read_file", "view_image", "compare_images",
                  "measure", "prompt_show", "note", "finish"}

# What the model may hand to `source`, and what a run can be told to do with it.
FINISH_STATUSES = ("done", "budget_exhausted", "gave_up", "no_candidates")

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
    """Is the endpoint answering? Used only by the reconnect loop.

    /health is llama.cpp's and vLLM's; LiteLLM keeps that path for an
    admin-only upstream check that 401s/403s a normal key, and puts the
    liveness probe on /health/liveliness. Try both, and treat an auth refusal
    as up - something answered, which is the whole question here. Without this
    the reconnect loop against a proxy waits out its full budget and gives up
    on a server that is running fine.
    """
    for path in ("/health/liveliness", "/health"):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}{path}",
                headers={"Authorization": f"Bearer {API_KEY}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 200:
                    return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return True
        except Exception:
            continue
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
               retries=2, think=True) -> dict:
    """One chat completion, with the reasoning-model truncation trap handled.

    A reasoning model spends tokens on `reasoning_content` before writing any
    `content`. If the budget runs out mid-thought the reply is
    content="" / finish_reason="length" - not an error, just a starved turn.
    Retry once with a bigger budget rather than surfacing an empty answer.

    `think=False` switches the chain off for this one call. vLLM's Qwen chat
    template honours `enable_thinking`; a server that does not know the key
    ignores it and reasons as usual. For a form-filling question with a fixed
    answer shape the chain is ~8x the latency for the same answer (measured in
    stage C of tools/select_reference.py); for an open judgement it is what
    makes the answer worth having, so the agent's own turns keep it on.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
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

def _tool(name: str, description: str, properties: dict,
          required: list[str] | None = None) -> dict:
    """One OpenAI function spec. Four parallel lists used to drift; this is one."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required or []}}}


_PATH = {"type": "string", "description": "Path, absolute or workspace-relative."}
_CANDIDATE = {
    "type": "string",
    "description": ("A candidate name: 'cand_03', or 'cand_03s' for its "
                    "segmented form. 'source' means the original segmented "
                    "photo."),
}

TOOL_SPECS = [
    _tool("bash",
          "Run a shell command in the workspace. Returns exit code, stdout and "
          "stderr.",
          {"command": {"type": "string"}}, ["command"]),

    _tool("read_file",
          "Read a text file with line numbers.",
          {"path": _PATH,
           "offset": {"type": "integer", "description": "First line, 1-based."},
           "limit": {"type": "integer", "description": "How many lines."}},
          ["path"]),

    _tool("write_file",
          "Write a text file, overwriting it. Short files only - use bash with a "
          "quoted heredoc for anything long, because a big string argument gets "
          "truncated mid-JSON and the call is rejected.",
          {"path": _PATH, "content": {"type": "string"}}, ["path", "content"]),

    _tool("edit_file",
          "Replace the first exact occurrence of old_text with new_text. Refuses "
          "if old_text is missing or ambiguous.",
          {"path": _PATH,
           "old_text": {"type": "string"},
           "new_text": {"type": "string"}},
          ["path", "old_text", "new_text"]),

    _tool("view_image",
          "Look at one image and ask a question about it. For a close look at a "
          "detail, pass a box in source pixels - without one the whole frame is "
          "downscaled to 1024px and you see nothing of the kind.",
          {"path": _PATH,
           "question": {"type": "string"},
           "box": {"type": "string",
                   "description": "'x,y,w,h' in source pixels. Optional."}},
          ["path", "question"]),

    _tool("compare_images",
          "Put TWO images in one vision call and ask how they differ. Use this "
          "rather than two view_image calls whenever the question is about a "
          "difference: two independent descriptions are not a comparison.",
          {"path_a": _PATH, "path_b": _PATH,
           "question": {"type": "string"},
           "box_a": {"type": "string", "description": "'x,y,w,h'. Optional."},
           "box_b": {"type": "string", "description": "'x,y,w,h'. Optional."}},
          ["path_a", "path_b", "question"]),

    _tool("segment",
          "Drop the background and stand the garment on flat white. Works on "
          "the source AND on any candidate you have generated. Free. The result "
          "is a CUTOUT - no plate, no shadow - so it is a thing to measure or "
          "to generate from, never a thing to deliver. It chooses where the "
          "result goes: cand_03 becomes cand_03s, which is a name every other "
          "tool accepts. It does NOT remove tags, pins or clips - only the "
          "background.",
          {"image": _CANDIDATE}, ["image"]),

    _tool("prompt_show",
          "Show the prompt: every section, which ones are still empty, the word "
          "count, and what the guardrails make of it. Free, and the way to check "
          "a prompt before spending on it.",
          {}),

    _tool("prompt_set",
          "Add a prompt section or replace one, leaving every other section "
          "byte-for-byte alone. The seeded sections are garment, fidelity, "
          "flatten, pose, background, reference - but any name works, so a "
          "detail none of them covers gets its own.",
          {"section": {"type": "string"},
           "text": {"type": "string"}},
          ["section", "text"]),

    _tool("prompt_remove",
          "Delete a prompt section.",
          {"section": {"type": "string"}}, ["section"]),

    _tool("prompt_replace",
          "Throw away every section and write the whole prompt as one block. "
          "Destructive - use prompt_set when you only mean to change one thing.",
          {"text": {"type": "string"}}, ["text"]),

    _tool("generate",
          "THE ONLY THING THAT COSTS MONEY. Send the prompt and two images to "
          "fal.ai and get candidates back. `source` is image 1: 'source' for the "
          "original segmented photo, or a candidate name to edit an earlier "
          "attempt instead of starting over. The reference is always image 2. "
          "Each image counts against the run's budget and the budget never "
          "refills. Generate one or two at a time and look at what comes back.",
          {"source": dict(_CANDIDATE, default="source"),
           "num": {"type": "integer",
                   "description": "Images to buy in this call. Default 1."},
           "resolution": {"type": "string", "enum": ["1K", "2K", "4K"],
                          "description": "Default 2K. 4K costs double."},
           "seed": {"type": "integer",
                    "description": "The draw to sample. Same prompt + same seed "
                                   "= same picture, so CHANGE IT to escape a bad "
                                   "draw the prompt cannot fix - a second garment "
                                   "in frame, a stray shadow, sleeves that came "
                                   "out wrong on this sample but not the last. "
                                   "Sets the base for the wave: seed 500 with "
                                   "num 2 gives you 500 and 501. Omit it and "
                                   "numbering continues automatically."},
           "use_reference": {"type": "boolean",
                             "description": "Default true. False sends image 1 "
                                            "alone."},
           "match_pose": {"type": "boolean",
                          "description": "Take the sleeve angle, cuff spacing "
                                         "and size in frame off image 2 instead "
                                         "of from your words. "
                                         "Use it when you have already described "
                                         "the pose correctly and the generator "
                                         "ignored you - a sleeve angle is a "
                                         "geometric fact and words are lossy. It "
                                         "raises the odds of a SECOND garment in "
                                         "frame, so buy one or two and change the "
                                         "seed if a duplicate appears rather than "
                                         "editing the prompt."},
           "force": {"type": "boolean",
                     "description": "Send a prompt the guardrails refused. "
                                    "Recorded."}},
          []),

    _tool("measure",
          "Re-read the free numbers for a candidate: LAY against the reference "
          "first, then size vs the reference, flatness, colour dE with its hue "
          "part, and same-garment IoU against the original source. They "
          "already arrive automatically with each new candidate, so this is for "
          "after a segmentation pass, or when a reading has scrolled out of "
          "context. Free.",
          {"candidate": _CANDIDATE}, ["candidate"]),

    _tool("note",
          "Record what you concluded about a candidate. Free. This is also what "
          "survives when its image is elided from the conversation - a candidate "
          "with no note collapses to a filename and three numbers, so note "
          "anything you want to be able to compare against later.",
          {"candidate": _CANDIDATE, "text": {"type": "string"}},
          ["candidate", "text"]),

    _tool("pick_best",
          "Name the candidates you would ship, BEST FIRST, up to four, and why. "
          "The delivery is four ranked files, so keep the list full. Call it as "
          "often as you like and re-issue the whole list each time - the last "
          "call wins, and whatever is named when the run ends is what gets "
          "delivered, with any empty slot filled by the harness from its lay "
          "ranking. UNTOUCHED generations only - cand_NN as fal.ai returned "
          "it; a segmented or polished form is refused, so generate from it "
          "and name what comes back. Judge the lay first: it tells you when a "
          "candidate outside your list sits closer to the reference than one "
          "inside it.",
          {"candidates": {"type": "array", "items": _CANDIDATE,
                          "description": "Ranked, best first. Up to four."},
           "candidate": dict(_CANDIDATE, description="A single name, if you "
                                                     "only have one."),
           "why": {"type": "string"}},
          ["why"]),

    _tool("finish",
          "End the run. status: 'done' when the result is good enough; "
          "'budget_exhausted' when the images ran out and you are shipping the "
          "best of them; 'gave_up' when nothing is worth shipping; "
          "'no_candidates' when nothing was ever generated.",
          {"summary": {"type": "string",
                       "description": "What you did, what you shipped, and what "
                                      "it still carries. Use the numbers you "
                                      "actually saw."},
           "status": {"type": "string", "enum": list(FINISH_STATUSES)}},
          ["summary", "status"]),
]


class Tools:
    def __init__(self, workspace: Path, run_dir: Path, allow_outside: bool,
                 source_photo: Path | None = None):
        self.ws = workspace
        self.run_dir = run_dir
        self.allow_outside = allow_outside
        self.env = child_env(workspace)
        self.python = find_python()
        # The RAW photo, kept so segment("source") can be re-run. Everything
        # else works from archive/source_clean.jpg, which is what it produces.
        self.source_photo = source_photo or (workspace / "inputs" /
                                             "off_set_image.jpg")
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
        p = self.resolve_image(path)
        if p is None or not p.exists():
            return None, (f"ERROR: no such image: {path}\n"
                          f"Candidate names work here too - {self._known()} - "
                          f"and so does a bare filename from the run's archive.")
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

        Every answer carries NOT_PROMPT_TEXT. A comparison is phrased as a
        contrast - "X, whereas Y", "closed, not splayed open" - and that reads
        exactly like an instruction, so it gets pasted into a prompt section
        unedited. On runs/20260901_140946 the first call of the run described
        the reference hood as "folded flat and closed into a neat rounded
        shape"; four seconds later that clause was the pose section, and the
        generator spent the whole run building a closed rounded hood onto a
        garment whose reference has it splayed flat open.
        """
        a = self._prep_image(path_a, question, box_a)
        if a[0] is None:
            return a[1]
        b = self._prep_image(path_b, question, box_b)
        if b[0] is None:
            return b[1]
        name_a = (self.resolve_image(path_a) or Path(path_a)).name
        name_b = (self.resolve_image(path_b) or Path(path_b)).name
        answer = self._ask_vision(
            f"You are given two images. The FIRST is {name_a}. The SECOND is "
            f"{name_b}.\n\n{question}\n\nAnswer concretely in a few sentences, "
            f"naming which image each observation is about. Describe only "
            f"differences you can actually see; if you cannot tell, say so "
            f"rather than guessing.", [a[0], b[0]], max_tokens=4000)
        return (f"[1st: {a[1]}]\n[2nd: {b[1]}]\n{answer}\n"
                f"{self.NOT_PROMPT_TEXT}")

    # One bracketed line, so _lay_check's existing "[" filter drops it from the
    # automatic check and it only lands where it is needed - in front of the
    # agent, next to prose it is about to reuse.
    NOT_PROMPT_TEXT = (
        "[The answer above is a COMPARISON, not prompt text. It is written as "
        "a contrast between two pictures, and a contrast reads as an "
        "instruction once it reaches the generator: a phrase like \"closed, "
        "not splayed open\" tells it to build a closed shape. Do not paste any "
        "of it into a prompt section. Write pose from a fresh look at the "
        "reference on its own, describing how that garment lies in plain "
        "positive terms.]")

    # -- the three real capabilities, plus the free bookkeeping -----------

    def _script(self, name: str, argv: list[str], timeout: int = 900) -> str:
        """Run one of this project's scripts and hand back what it printed.

        Shelled out rather than imported so that a tool call and the same thing
        typed into bash behave identically - same interpreter, same environment,
        same steps.log line. A run that debugs one has debugged the other.
        """
        cmd = [self.python, str(HERE / "tools" / name), *argv]
        TR.info("tool", f"exec {name}", body=" ".join(cmd))
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=self.ws, env=self.env,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"TIMEOUT: {name} was killed after {timeout}s."
        out = (r.stdout or "") + (r.stderr or "")
        TR.info("tool", f"{name} exit={r.returncode}",
                secs=round(time.time() - t0, 2), body=out)
        return out.strip() or f"exit={r.returncode} (no output)"

    def _archive(self) -> Path:
        return self.run_dir / "archive"

    def _image_path(self, name: str) -> Path | None:
        """A candidate name to a file. None when the name is not one."""
        name = (name or "").strip()
        arch = self._archive()
        if name in ("source", "src", "original"):
            return arch / "source_clean.jpg"
        if re.fullmatch(r"cand_\d+[spc]*", name):
            return arch / f"{name}.png"
        return None

    def resolve_image(self, path: str) -> Path | None:
        """Anything that names an image: a candidate name, a bare filename, a path.

        The image tools used to take paths only, while every other tool took
        names, and the seam cost four turns of a real run: the model called
        view_image("cand_02s.png"), got "no such image" because that resolves
        against the workspace root, and then spent three turns running find and
        ls to rediscover a file it had just created. Accepting all three forms
        removes the class of mistake rather than documenting it.
        """
        p = self._image_path(path)                     # cand_03, cand_03s, source
        if p is not None:
            return p
        direct = self.resolve(path)                    # a real path, relative or not
        if direct.exists():
            return direct
        inside = self._archive() / Path(path).name     # a bare archive filename
        return inside if inside.exists() else direct

    def _known(self) -> str:
        have = sorted(p.stem for p in self._archive().glob("cand_*.png"))
        return ", ".join(["source"] + have)

    def segment(self, image: str) -> str:
        """Drop the background on the source or on any candidate.

        The destination is chosen here rather than passed in, and that is the
        whole point. A model free to segment to /tmp/x.png and then generate from
        that path would enter lineage.json with no parent and depth 0 - the drift
        warning would go quiet at exactly the point in the chain where it matters
        most. cand_03 becomes cand_03s, which every other tool accepts.
        """
        import generate as GEN

        arch = self._archive()
        arch.mkdir(parents=True, exist_ok=True)
        name = (image or "").strip()

        if name in ("source", "src", "original", ""):
            src, out, child = self.source_photo, arch / "source_clean.jpg", None
        else:
            src = self._image_path(name)
            if src is None:
                return (f"ERROR: '{image}' is not a candidate name. "
                        f"Known: {self._known()}")
            if not src.exists():
                return f"ERROR: {src.name} does not exist. Known: {self._known()}"
            child = f"{name.rstrip('s')}s" if not name.endswith("s") else name
            out = arch / f"{child}.png"

        text = self._script("segment.py", ["--run", str(self.run_dir),
                                           "--off-set", str(src),
                                           "--out", str(out)])
        if not out.exists():
            return (f"{text}\n\nSegmentation produced nothing; "
                    f"{src.name} is unchanged and still usable.")

        if child:
            # Depth is the PARENT's, unchanged. Segmentation drops a background;
            # it does not redraw anything and cannot invent detail, so it is not
            # a hop away from the real garment and must not read as one.
            parent_depth = GEN.depth_of(self.run_dir, name)
            GEN.record(self.run_dir, child, {
                "parent": name, "parent_kind": "segmentation",
                "depth": parent_depth, "prompt_hash": None, "seed": None,
                "created": datetime.now().isoformat(timespec="seconds"),
            })
            text += (f"\n\nSaved as {child} (depth {parent_depth}, same as "
                     f"{name} - segmentation adds no drift). It is a CUTOUT: "
                     f"the garment on flat white, the plate and its shadow "
                     f"gone. Measure it, or generate from it so a plate comes "
                     f"back - generate(source='{child}'). pick_best refuses "
                     f"it: a cutout shipped as best.png on runs/20260902_233202 "
                     f"and was rejected for exactly that look. Name {name}, or "
                     f"whatever you generate from {child}.")
        return text

    def prompt_show(self) -> str:
        import promptfile as PF
        try:
            return PF.show(self.run_dir, reference_path(self.run_dir))
        except RuntimeError as e:
            return f"ERROR: {e}"

    def prompt_set(self, section: str, text: str) -> str:
        import promptfile as PF
        return PF.set_section(self.run_dir, section, text)

    def prompt_remove(self, section: str) -> str:
        import promptfile as PF
        return PF.remove_section(self.run_dir, section)

    def prompt_replace(self, text: str) -> str:
        import promptfile as PF
        return PF.replace_all(self.run_dir, text)

    def measure(self, candidate: str) -> str:
        import metrics as MET
        src = self._archive() / "source_clean.jpg"
        p = self._image_path(candidate)
        if p is None:
            return (f"ERROR: '{candidate}' is not a candidate name. "
                    f"Known: {self._known()}")
        if not p.exists():
            return f"ERROR: {p.name} does not exist. Known: {self._known()}"
        if not src.exists():
            return f"ERROR: no {src.name} to measure against."
        ref = reference_path(self.run_dir)
        try:
            return MET.line(MET.compare(src, p, reference=ref if ref.exists()
                                        else None), candidate)
        except Exception as e:  # noqa: BLE001
            return f"ERROR measuring {candidate}: {type(e).__name__}: {e}"

    def note(self, candidate: str, text: str) -> str:
        """What you concluded about a candidate, kept where elision can find it."""
        f = self._archive() / "notes.json"
        book = {}
        if f.exists():
            try:
                book = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                book = {}
        book[candidate] = text.strip()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(book, indent=2) + "\n")
        return (f"noted on {candidate}. This is what stays in the conversation "
                f"once its image is elided.")

    def pick_best(self, candidates: list | None = None, why: str = "",
                  candidate: str | None = None) -> str:
        """Record the ranked delivery list. Up to TOP_N, best first.

        `candidate` is the old single-name form and still works; it is a list
        of one. Names are deduplicated by generation - cand_03 and cand_03p
        are the same buy - keeping the first mention, so a list cannot fill
        two of the four slots with one image.
        """
        names: list[str] = []
        raw = list(candidates or []) + ([candidate] if candidate else [])
        if not raw:
            return ("ERROR: name at least one candidate - `candidates` is a "
                    f"ranked list, best first, up to {TOP_N}. Known: "
                    f"{self._known()}")
        seen_gen: set[str] = set()
        for n in raw:
            n = str(n).strip()
            p = self._image_path(n)
            if p is None:
                return (f"ERROR: '{n}' is not a candidate name. "
                        f"Known: {self._known()}")
            if not p.exists():
                return f"ERROR: {p.name} does not exist. Known: {self._known()}"
            gen = generation_of(n)
            if n != gen:
                kind = ("a cutout - the garment on flat white, the plate and "
                        "its shadow gone" if is_cutout(n)
                        else "a second model's edit of a generation")
                return (f"ERROR: {n} is {kind}. Only untouched fal.ai "
                        f"generations ship. Name {gen}, or generate from {n} "
                        f"and name what comes back.")
            if gen in seen_gen:
                continue
            seen_gen.add(gen)
            names.append(n)
        dropped = names[TOP_N:]
        names = names[:TOP_N]
        f = self._archive() / "best.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(
            {"candidate": names[0], "candidates": names, "why": why.strip(),
             "at": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")

        # A nudge, not a veto. runs/20260902_100812 shipped a candidate at
        # 0.892 against the reference while two sat at 0.96, and nothing in the
        # run said so. Now the pick is told, and asked to name what disqualifies
        # the closer ones if it is passing them over.
        note = ""
        table = lay_ranking(self.run_dir, self.__dict__.setdefault("_lay_cache", {}))
        by_name = dict(table)
        mine = [by_name[n] for n in names if n in by_name]
        if mine:
            floor = min(mine)
            picked_gens = {generation_of(n) for n in names}
            ahead = [(n, v) for n, v in table
                     if generation_of(n) not in picked_gens and v >= floor + 0.03]
            if ahead:
                top = ", ".join(f"{n} {v:.3f}" for n, v in ahead[:3])
                note = (f"\n  NOTE: {top} sit closer to the reference's lay "
                        f"than the weakest of your picks ({floor:.3f}). The lay "
                        f"comes first and colour second, and a lighter render "
                        f"is not a different colour. If you are passing those "
                        f"over, say what disqualifies them - construction, a "
                        f"second garment, a hanger, a different hue - in a note.")
        if dropped:
            note += (f"\n  (only the first {TOP_N} are kept; "
                     f"{', '.join(dropped)} dropped)")
        empty = TOP_N - len(names)
        if empty:
            note += (f"\n  {empty} of {TOP_N} slot(s) still empty - the harness "
                     f"fills them from its lay ranking at the end unless you do.")
        return (f"delivery is now {', '.join(names)}. Call pick_best again with "
                f"the whole list to change it; what is named when the run ends "
                f"is what ships, after one last pairwise check that rank 1 is "
                f"the product's own colour." + note)

    def generate(self, source: str = "source", num: int = 1,
                 resolution: str = "2K", seed: int | None = None,
                 use_reference: bool = True, match_pose: bool = False,
                 force: bool = False) -> str:
        argv = ["--run", str(self.run_dir), "--source", str(source),
                "--num", str(int(num or 1)), "--resolution", str(resolution)]
        if seed is not None:
            argv += ["--seed", str(int(seed))]
        if not use_reference:
            argv.append("--no-reference")
        if match_pose:
            argv.append("--match-pose")
        if force:
            argv.append("--force")
        return self._script("generate.py", argv, timeout=1800)

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

# Your budget

{budget} image(s) of fal.ai generation for this whole run, and the counter never
refills. Everything else - looking, comparing, measuring, segmenting, rewriting
the prompt - is free and unlimited.

# How to work

- Take ONE step at a time. Call a tool, read the result, then decide the next step.
- LOOK before you spend and look after. The two images in your first message are
  the whole brief: image 1 is the garment you must keep, image 2 is the lay you
  are aiming at. compare_images puts two frames in a single vision call - use it
  whenever the question is how one differs from another, because two separate
  view_image calls give you two independent descriptions and the gap between two
  descriptions is not a measured difference.
- Build the prompt with prompt_set, one section at a time, and check it with
  prompt_show before generating. Both are free. When a batch comes back wrong in
  one respect, change the ONE section that governs it - rewriting the whole
  prompt re-rolls every decision that was already right.
- Buy small. One or two images, then look at them. A whole budget spent in one
  wave cannot learn anything from itself.
- The numbers under each candidate are in the order they matter. LAY is the
  candidate's silhouette against image 2, with where the untouched source
  starts from beside it - higher is closer, and a candidate that barely moved
  sits near the source's own number. size is the garment's share of the frame
  over image 2's; 1.0 is right. flat is surface energy over the source's;
  below 1 is wanted. colour dE is second-level: a flat render reads lighter
  than a creased photograph, so read the hue part - under 4 is the same
  colourway, over 8 is a different one and disqualifies.
  same-garment IoU is against the source and drops when the lay changes,
  which is the job. Nothing is blocked by any of them, and your eyes still
  have the last word on construction: a candidate that hit the lay by
  inventing a seam or losing the logo is not shippable.
- Ground every claim in output you actually saw. Never report a number you did
  not measure or a file you did not create.
- Keep tool output small. Print the few numbers you need, not whole arrays;
  pipe long output through head, grep or wc.
- write_file is for short files. Anything long - a report - must be written with
  bash and a quoted heredoc, appending section by section. A big string argument
  gets truncated mid-JSON and the whole call is rejected.
- Call pick_best with your ranked list, best first, as soon as anything is
  worth shipping, and re-issue the whole list as better ones arrive. The run
  delivers four files, so keep the list full; a slot you leave empty is filled
  by the harness from its lay ranking. It is free, and it means an interrupted
  run still delivers. Before rank 1 ships it is checked once more, pairwise
  against the product photo with colour first, and a candidate that copied
  the reference's tone loses there.
- When the result is good enough, or the images run out, call finish().

You have a limited context window. Candidate images are dropped from the
conversation after a couple of turns, and a candidate you left a note on keeps
its note when that happens. One with no note collapses to a filename and three
numbers.
"""


SKIP_DIRS = {"runs", ".git", "__pycache__", "node_modules", ".venv",
             "output", "notes",
             # The retired pipeline and its 45-image reference library. Not
             # inputs, not callable, and at 8 files per folder they would crowd
             # the two images this run is actually about out of the listing.
             "old"}

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


# How many candidate images stay inline. Two, so the model can hold a new
# candidate beside the one it is trying to beat.
KEEP_IMAGES = 2


def encode_image(path: Path, max_dim: int = 1024) -> str | None:
    """One image as a base64 JPEG, small enough to live in the conversation."""
    try:
        from PIL import Image
        im = Image.open(path)
        im.load()
        if max(im.size) > max_dim:
            im.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:  # noqa: BLE001 - a missing image is not a crash
        TR.warn("session", f"could not encode {path}", error=str(e))
        return None


def image_block(b64: str) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


class Session:
    def __init__(self, task: str, skill_text: str, workspace: Path, run_dir: Path,
                 tools: Tools, approver: Approver, max_iters: int, verbose: bool,
                 n_ctx: int = N_CTX_FALLBACK, budget: int = 10):
        self.ws = workspace
        self.run_dir = run_dir
        self.tools = tools
        self.approver = approver
        self.max_iters = max_iters
        self.verbose = verbose
        self.n_ctx = n_ctx
        self.budget = budget
        self.compact_at = int(n_ctx * COMPACT_FRACTION)
        self.prompt_tokens = 0
        self.total_completion = 0
        self.compactions = 0
        self.nudged = False

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
            skill_block=skill_block, task=task, budget=budget,
        )

        # Turn 1 carries both images, not just the words. The job is "make image
        # 1 look like image 2", and a run that has to spend a turn asking to see
        # its own inputs has started by describing the problem to itself instead
        # of looking at it.
        # Read the source's own fastenings once, up front. The standing clause
        # asks for them to go whether or not anyone looked, but knowing what is
        # actually there turns "did the pins go" from a guess into a comparison -
        # and the model cannot tell a pin that was removed from one that was
        # never in the picture unless it was told at the start.
        brief = self._brief() + self._bleed_warning()
        found = self._props_check_path(run_dir / "archive" / "source_clean.jpg")
        if found:
            brief += ("\n\nWHAT IS ON THE SOURCE RIGHT NOW (read automatically "
                      "from image 1):\n" + textwrap.indent(found, "  ")
                      + "\n\nEvery prompt already asks for the temporary "
                      "fastenings to go and the sewn-in detail to stay, so you "
                      "do not need to write that. This is here so you can tell "
                      "whether a candidate actually did it.")

        first = [{"type": "text", "text": task + "\n\n" + brief}]
        for label, p in (("IMAGE 1 - the garment, background already dropped. "
                          "This is what fal.ai receives as image 1.",
                          run_dir / "archive" / "source_clean.jpg"),
                         ("IMAGE 2 - the reference laydown, desaturated and "
                          "re-toned. THE TARGET: match its silhouette, its "
                          "flatness and its size in the frame. Its construction "
                          "and trim belong to a different product. Image 1 is "
                          "the only colour target.",
                          reference_path(run_dir))):
            b64 = encode_image(p) if p.exists() else None
            if b64:
                first.append({"type": "text", "text": label})
                first.append(image_block(b64))

        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": first},
        ]
        # Never elided, never compacted. These two images are the standard every
        # later judgement is made against, and losing them mid-run means the
        # model starts comparing candidates to each other instead.
        self.pinned = 2
        self.log_path = run_dir / "transcript.jsonl"

        # The system prompt is assembled here from the skill and a fingerprint
        # of the workspace, so it differs run to run - and half of "why did it
        # do that" is answered by what it was told at the start.
        TR.info("session", "system prompt", body=system, chars=len(system))
        TR.info("session", "task", body=task)
        TR.info("session", "skill loaded" if skill_text else "no skill",
                chars=len(skill_text or ""))

    def _bleed_warning(self) -> str:
        """What the chosen reference carries that this product does not.

        Step 0 asked the model, in the same breath as picking the reference,
        what still differs between the two garments - and scanned that sentence
        for words naming CONSTRUCTION rather than pose. Those words are the one
        early warning available, and they are available before a penny is spent.

        It matters because the failure is invisible afterwards. On one run of
        the old pipeline the reference carried a defined V-neckline and seam
        piping the real garment did not have, and four of ten candidates came
        back with a neckline seam and topstitching down the straps. Invented
        construction is plausible construction, so it survives casual review.

        Empty string when nothing was flagged, or when the reference was
        supplied by hand and never compared to anything.
        """
        f = self.run_dir / "reference_selection.json"
        if not f.exists():
            return ""
        try:
            risk = (json.loads(f.read_text()).get("construction_risk") or {})
        except (json.JSONDecodeError, OSError):
            return ""
        if not risk.get("flagged") or not risk.get("terms"):
            return ""
        return ("\n\nWHAT IMAGE 2 HAS AND IMAGE 1 DOES NOT (read automatically "
                "when the reference was chosen):\n"
                f"  {', '.join(risk['terms'])}\n"
                f"  \"{risk.get('line') or ''}\"\n\n"
                "The reference is a different product. It was picked for how it "
                "LIES and for nothing else, and the words above are the parts of "
                "it that do not belong to this garment. Do not describe any of "
                "them in a prompt, and check every candidate for them - a seam "
                "or a pocket copied off image 2 looks exactly like one the "
                "garment always had.")

    def _brief(self) -> str:
        arch = self.run_dir / "archive"
        return (
            "Both images are below. Image 2 is the target: its silhouette, its "
            "flatness and its size in the frame are what a candidate is judged "
            "on first. Image 1 is the product: its construction - seams, "
            "pockets, cuffs, labels, logo - must survive exactly, in image 1's "
            "colour. Colour comes second: a flat render reads lighter than a "
            "creased photograph and that is not a fault, but a different hue "
            "is. A candidate with the wrong silhouette is out first.\n\n"
            "THE ONLY TWO PATHS YOU NEED:\n"
            f"  image 1   {arch / 'source_clean.jpg'}\n"
            f"  image 2   {reference_path(self.run_dir)}\n\n"
            "Use those exact paths for view_image and compare_images. The raw "
            "photo in inputs/ is NOT what fal.ai receives - it still has the "
            "room, the wall and the shadow in it. Writing a prompt from the raw "
            "file describes things the generator was never sent, and it draws "
            "them: one run described the raw input's hang tag, and all four "
            "candidates came back wearing one.\n\n"
            "Start by looking at both and saying what actually differs. Then "
            "write the prompt. Then buy one or two images and look at them.")

    def log(self, kind: str, payload):
        with self.log_path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, "data": payload},
                               default=str) + "\n")

    # -- candidates in the conversation ------------------------------------
    def notes(self) -> dict:
        f = self.run_dir / "archive" / "notes.json"
        if not f.exists():
            return {}
        try:
            return json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def attach_image(self, name: str, path: Path, caption: str) -> None:
        """Put a candidate in front of the model as a picture, not a filename."""
        b64 = encode_image(path)
        if not b64:
            return
        self.messages.append({
            "role": "user",
            "_image": name, "_caption": caption,
            "content": [{"type": "text", "text": caption}, image_block(b64)],
        })
        TR.info("session", f"attached {name} to the conversation")

    def elide_images(self) -> int:
        """Keep the last KEEP_IMAGES candidate pictures; the rest become text.

        RUNS EVERY TURN, ON PURPOSE, and not as part of compact(). Full
        compaction only fires past COMPACT_FRACTION of the window, which three
        or four candidates will never reach - so this rule would sit untested
        until a ten-image run, which is the worst possible place to find out it
        is wrong. Unconditional means every run exercises it, and how many images
        are in the conversation stops being a function of how long the prompt is.

        A candidate the model re-viewed comes back inline through the normal
        vision path and is elided again next turn like any other. Re-viewing buys
        one turn of attention, not permanence.
        """
        book = self.notes()

        def as_text(m: dict) -> str:
            note = (book.get(m["_image"]) or "").strip()
            return (m.get("_caption", m["_image"])
                    + (f"\n  your note: {note}" if note else
                       "\n  (image dropped from the conversation to save "
                       "context, and you left no note on it. view_image brings "
                       "it back for one turn.)"))

        # Already-elided entries are re-rendered too, not just skipped. The
        # natural order is look-then-note, so a note is usually written a turn or
        # two AFTER the candidate it describes - by which time that candidate can
        # already be text. Rendering once at elision would drop every note
        # written late, which is most of them, and the model would have no way to
        # tell: the tool said "noted" and the note simply never appeared.
        for m in self.messages[self.pinned:]:
            if m.get("_elided") and m.get("_image"):
                m["content"] = as_text(m)

        live = [m for m in self.messages[self.pinned:]
                if m.get("_image") and not m.get("_elided")]
        if len(live) <= KEEP_IMAGES:
            return 0
        dropped = 0
        for m in live[:-KEEP_IMAGES]:
            m["content"] = as_text(m)
            m["_elided"] = True
            dropped += 1
        if dropped:
            TR.debug("context", f"elided {dropped} candidate image(s)",
                     kept=KEEP_IMAGES)
        return dropped

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
    def candidates(self) -> list[str]:
        """Billed images on disk. Segmented derivatives do not count as buys."""
        return sorted(p.stem for p in (self.run_dir / "archive").glob("cand_*.png")
                      if re.fullmatch(r"cand_\d+", p.stem))

    def best(self) -> dict | None:
        f = self.run_dir / "archive" / "best.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def images_left(self) -> int:
        return max(0, self.budget - len(self.candidates()))

    def run(self) -> dict:
        result = {"status": "max_iters", "summary": "Hit the iteration cap."}

        for i in range(1, self.max_iters + 1):
            # Unconditional, before anything else. See elide_images().
            self.elide_images()
            if self.prompt_tokens > self.compact_at:
                self.compact()

            # The nudge, once, and only when all three hold. The candidate count
            # is what makes it safe: without it, a zero-budget run or one whose
            # first generation failed gets told to "mark your best and finish"
            # with nothing on disk to mark, which is advice it cannot follow and
            # a turn it cannot spend well. A real run reached its cap mid-appeal
            # with ten paid-for images and nothing delivered - that is the case
            # this exists for, and that run had candidates.
            turns_left = self.max_iters - i + 1
            if (not self.nudged and self.candidates()
                    and (self.images_left() <= 2 or turns_left <= 3)
                    and self.best() is None):
                self.nudged = True
                TR.warn("session", "delivery nudge injected",
                        turns_left=turns_left, images_left=self.images_left(),
                        candidates=len(self.candidates()))
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"{turns_left} turn(s) and {self.images_left()} image(s) "
                        f"left, and nothing is marked as best yet. You have "
                        f"{len(self.candidates())} candidate(s) on disk, all "
                        f"already paid for.\n\n"
                        f"Call pick_best now with the ones you would defend, "
                        f"best first, even if none of them is what you wanted - "
                        f"ranking the least bad and saying what they carry is a "
                        f"delivery; leaving them unnamed is not. Then finish() "
                        f"with an honest "
                        f"status: 'done' if it is good enough, "
                        f"'budget_exhausted' if you simply ran out of images, "
                        f"'gave_up' if nothing here is shippable."),
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
                              "summary": args.get("summary", ""),
                              "candidates": len(self.candidates()),
                              "best": (self.best() or {}).get("candidate")}
                    # A status can never claim more than the run delivered.
                    # finish("done") with nothing generated would otherwise exit
                    # 0 and ship nothing, which reads downstream as a successful
                    # delivery of a garment nobody ever produced.
                    if not result["candidates"] and result["status"] != "no_candidates":
                        TR.warn("session",
                                f"finish({result['status']}) with no candidates "
                                f"- recorded as no_candidates",
                                claimed=result["status"])
                        result["claimed_status"] = result["status"]
                        result["status"] = "no_candidates"
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

                if name == "generate" and allowed:
                    self._attach_new_candidates()

        TR.warn("session", "hit the iteration cap without finish()",
                max_iters=self.max_iters)
        return result

    # Asked about the LAY and nothing else. The two garments are different
    # products, so any open question about how they differ comes back describing
    # collars and colours - which is true, irrelevant, and drowns the one thing
    # the reference exists to settle.
    #
    # FALLBACK ONLY. This fixed list of parts is what the check used to ask on
    # every run, and it is blind to any part not named in it. On
    # runs/20260901_140946 the pose defect was a hood - splayed flat open in the
    # reference, rebuilt as a raised peak in the candidate - and the check waved
    # it through four times because sleeves, cuffs, shoulders and hem were the
    # only things it knew how to ask about. _lay_question() below builds the
    # real question from the run's own pose section instead; this text is what
    # is left when that section is still empty.
    LAY_QUESTION = (
        "Compare ONLY how the two garments are arranged - their pose on the "
        "plate. Ignore colour, pattern, trim, collar and construction entirely; "
        "these are different products and are meant to differ there.\n\n"
        "Answer these in order:\n"
        "1. In the SECOND image (the reference), what angle do the sleeves make "
        "away from the body, and how far are the cuffs from the sides?\n"
        "2. In the FIRST image, what angle and distance?\n"
        "3. Do they match? If not, say which way the first one is wrong - "
        "sleeves too far in, too far out, too high, too low.\n"
        "4. Are the shoulders and hem level, and is the garment square to the "
        "frame?\n"
        "Be specific about angles. Two or three sentences.")

    # The same scoping as LAY_QUESTION, but the list of parts to inspect comes
    # from the pose section the agent wrote for THIS run, so a hood, a strap, a
    # trouser leg or a bag handle gets checked without anything here knowing
    # those words.
    #
    # THE POSE TEXT IS THE CHECKLIST, NEVER THE ANSWER KEY. It is the agent's
    # description of the arrangement it was aiming for, and it can be wrong: on
    # runs/20260901_140946 it said the reference hood was "folded flat and
    # closed into a neat rounded shape at the top, not splayed open" when the
    # reference hood is flat and spread wide. A check that graded the candidate
    # against those words would have passed the peaked hood a second time. Only
    # the reference image decides what is correct; the pose text decides what
    # gets looked at.
    LAY_QUESTION_HEAD = (
        "Compare ONLY how the two garments are arranged - their pose on the "
        "plate. Ignore colour, pattern, trim, collar and construction entirely; "
        "these are different products and are meant to differ there.\n\n"
        "Look at each of these parts in turn: {parts}.\n\n"
        "For EACH part, answer in this order:\n"
        "1. How does that part sit in the SECOND image (the reference)? "
        "Describe what you can see there, in your own words.\n"
        "2. How does it sit in the FIRST image?\n"
        "3. Do they match? If not, say which way the first one is wrong.\n\n"
        "Then: are the shoulders and hem level, and is the garment square to "
        "the frame? Be specific about angles and distances. One or two "
        "sentences per part, no more.")

    # Pull the part NAMES out of the pose section and throw its adjectives away.
    #
    # Handing the vision model the pose prose whole does name the right parts,
    # but it also hands over the wording, and the answer comes back wearing it:
    # asked with runs/20260901_140946's pose text in the question, the model
    # described the reference hood as "folded flat and closed into a neat
    # rounded shape, not splayed" - which is the pose section's error, not what
    # is in the reference. It still caught the peaked hood, because peak versus
    # flat is too big to miss whatever vocabulary you arrive with, but a subtler
    # defect described in confident wrong words is exactly what would slip
    # through. Nouns carry the checklist; adjectives carry the answer key, and
    # only the nouns are wanted here.
    PARTS_QUESTION = (
        "Below is a description of how a garment should be laid out.\n\n"
        "--- description ---\n{pose}\n--- end ---\n\n"
        "List the PARTS of the garment it mentions - the physical pieces, such "
        "as sleeves, cuffs, hem, hood, collar, straps, legs, waistband, "
        "fastenings. Reply with the part names only, lower case, separated by "
        "commas, nothing else. Do not include how they are arranged, do not "
        "include adjectives, and do not include the garment itself.")

    # Long enough to name parts, short enough that runaway output is capped.
    PARTS_MAX_WORDS = 24

    def _lay_parts(self, pose: str) -> str:
        """The parts named in `pose`, as a comma list, or "" if unavailable."""
        try:
            data = call_model(
                [{"role": "user",
                  "content": self.PARTS_QUESTION.format(pose=pose)}],
                tools=None, max_tokens=2000, temperature=0.1, retries=1)
            parts = (data["choices"][0]["message"].get("content") or "").strip()
        except (ModelError, KeyError, IndexError) as e:
            TR.warn("session", "part extraction failed for the lay check",
                    error=str(e))
            return ""
        # A reasoning model that spends its budget thinking returns nothing, and
        # a chatty one returns a sentence. Neither is a checklist.
        parts = parts.replace("\n", " ").strip()
        if not parts or len(parts.split()) > self.PARTS_MAX_WORDS:
            TR.warn("session", "part extraction gave no usable list",
                    got=parts[:200])
            return ""
        return parts

    def _lay_question(self) -> str:
        """The lay check's question, built from this run's pose section.

        Falls back to LAY_QUESTION whenever the pose section cannot produce a
        checklist: unreadable, empty, still a stub, or the extraction call came
        back with something that is not a list of parts. The fallback asks about
        sleeves and cuffs only, which is thin, but a check that asks the wrong
        question beats a check that crashes on the way to asking it.
        """
        try:
            import promptfile as PF
            pose = (PF.load(self.run_dir)["sections"].get("pose") or "").strip()
        except Exception as e:  # noqa: BLE001 - a check is not the delivery
            TR.warn("session", "could not read pose for the lay check",
                    error=str(e))
            return self.LAY_QUESTION
        if len(pose.split()) < 12:
            return self.LAY_QUESTION
        parts = self._lay_parts(pose)
        if not parts:
            return self.LAY_QUESTION
        return self.LAY_QUESTION_HEAD.format(parts=parts)

    # Both halves of the standing clause, asked as one question so the answer
    # cannot report a clean garment while the logo has quietly gone with the
    # pins. The clause promises two things and this checks two things.
    PROPS_QUESTION = (
        "Look at this garment photograph and answer two questions.\n\n"
        "1. TEMPORARY FASTENINGS. Is there any pin, pearl-headed pin, clip, "
        "tack, safety pin, hook, hanger, price ticket, swing tag or piece of "
        "string visible anywhere - on the garment, at the collar, along the hem "
        "or at the cuffs? These were holding the garment up for the photograph "
        "and should all be gone. List anything you can still see and say where.\n"
        "2. SEWN-IN DETAIL. Are the garment's own sewn-in features still there "
        "and intact - brand or care labels, embroidery, appliqué, printed or "
        "knitted logos, and the stitching around them? These are part of the "
        "product and must NOT have been removed.\n\n"
        "Answer both in two or three sentences. If you cannot see something "
        "clearly, say so rather than assuming it is fine.")

    def _props_check_path(self, img: Path) -> str:
        """The props question against any file. Used on the source at turn 1."""
        if not img.exists():
            return ""
        try:
            out = self.tools.view_image(str(img), self.PROPS_QUESTION)
        except Exception as e:  # noqa: BLE001 - a check is not the delivery
            TR.warn("session", f"props check failed for {img.name}", error=str(e))
            return ""
        return "\n".join(l for l in out.splitlines()
                         if not l.startswith("[")).strip()

    def _props_check(self, name: str) -> str:
        """Did the pins actually go, and did the logo survive?

        The standing clause in generate.py asks for both. Nothing verified
        either, and an instruction nobody checks is an instruction you find out
        about from the delivery. Segmentation cannot remove a pin - it drops the
        background only - so the clause is the ONLY thing standing between a pin
        in the source and a pin in the shipped image, which makes it exactly the
        thing worth confirming.

        Asked about the sewn-in half too, because the two failures are opposite
        halves of one instruction: a model told to remove attached objects can
        take the brand label with them, and a clean garment missing its logo
        would otherwise read as a success.
        """
        return self._props_check_path(self.run_dir / "archive" / f"{name}.png")

    def _lay_check(self, name: str) -> str:
        """Hold the new candidate up against the reference and ask about the pose.

        AUTOMATIC, because leaving it to the model does not work. On
        runs/20260828_110544 the whole run contained two compare_images calls -
        source against reference to write the prompt, and source against a
        candidate to check fidelity - and not one comparing a candidate to the
        reference. The run checked "is this still the same garment" every time
        and "did I hit the pose I was aiming at" never once, then shipped a
        candidate whose sleeves were wrong.

        Free: the vision model is the same local server the agent runs on. It is
        deliberately a QUESTION rather than a rule - nothing here knows what the
        right sleeve angle is, and it should not. The reference knows, and this
        puts the two pictures side by side so the model can see for itself.

        Which parts it asks about come from the run's own pose section - see
        _lay_question(). A fixed list of parts only ever catches the defects
        someone anticipated when they wrote the list.
        """
        ref = reference_path(self.run_dir)
        cand = self.run_dir / "archive" / f"{name}.png"
        if not ref.exists() or not cand.exists():
            return ""
        try:
            out = self.tools.compare_images(str(cand), str(ref),
                                            self._lay_question())
        except Exception as e:  # noqa: BLE001 - a check is not the delivery
            TR.warn("session", f"lay check failed for {name}", error=str(e))
            return ""
        # The two "[...]" preamble lines are about image preparation and say
        # nothing about the garment.
        body = "\n".join(l for l in out.splitlines()
                         if not l.startswith("[")).strip()
        return body

    def _attach_new_candidates(self):
        """Show what was just bought, with its reading printed underneath.

        generate.py leaves last_generation.json rather than being parsed out of
        its stdout: the numbers matter enough to be structured, and text scraping
        would break the first time a print statement moved.
        """
        f = self.run_dir / "archive" / "last_generation.json"
        if not f.exists():
            return
        try:
            info = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            return
        rows = {m.get("candidate", ""): m for m in info.get("metrics", [])}
        depth = info.get("depth", 1)
        for name in info.get("candidates", []):
            p = self.run_dir / "archive" / f"{name}.png"
            m = next((v for k, v in rows.items() if Path(k).stem == name), None)
            caption = f"{name} - just generated from {info.get('source')}"
            if m:
                import metrics as MET
                caption += "\n  " + MET.line(m, name)
            caption += (f"\n  depth {depth}"
                        + (" - measured against the ORIGINAL source, not its "
                           "parent" if depth > 1 else ""))
            lay = self._lay_check(name)
            if lay:
                caption += ("\n\n  LAY vs the reference (asked automatically, "
                            "pose only):\n"
                            + textwrap.indent(lay, "    "))
            props = self._props_check(name)
            if props:
                caption += ("\n\n  PINS AND LABELS (asked automatically):\n"
                            + textwrap.indent(props, "    "))
            self.attach_image(name, p, caption)
        # Deleted so the next generate cannot re-attach the same wave if it
        # fails before writing its own.
        f.unlink(missing_ok=True)

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
        elif name == "generate":
            detail = (f"{args.get('num', 1)} at {args.get('resolution', '2K')} "
                      f"from {args.get('source', 'source')}  "
                      f"({self.images_left()} left)")
        elif name in ("segment", "measure", "note", "pick_best"):
            detail = f"{args.get('image') or args.get('candidate', '')}"
        elif name.startswith("prompt_"):
            detail = args.get("section", "")
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
    "Re-lay the garment in image 1 into the laydown in image 2: the same "
    "silhouette, ironed completely flat, the same size in the frame, on a clean "
    "white plate with no shadows. It must still be the SAME product - every "
    "seam, pocket, cuff, label and logo is image 1's, in image 1's colour. "
    "Judge candidates on the lay first and on colour second.\n\n"
    "Work iteratively. Look at both images and say what actually differs. Write "
    "the prompt section by section. Buy one or two images, look at what came "
    "back, and decide: change a section and try again from the original, or take "
    "the best candidate so far and edit that one further. Segmenting a candidate "
    "to clean up its plate is free and available at any point.\n\n"
    "The delivery is the best FOUR candidates of the run, ranked. Mark them "
    "with pick_best as soon as you have any worth shipping, best first, and "
    "re-issue the list as it changes. Once one lay is right, spend the rest of "
    "the budget on new seeds to collect alternates rather than refining one "
    "image. Finish when you have four you would defend, or the images run out."
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def preflight() -> tuple[str, int]:
    """Confirm the server is reachable; return (model id, context window).

    vLLM reports `max_model_len` per model on /v1/models and LiteLLM reports
    `max_input_tokens`; llama.cpp reports neither, hence the fallback. Reading it
    beats a constant: this project has already run a 262144-token server while
    every budget in the file said 32768.

    Also sets the module-level MODEL, because a proxy routes on the name and a
    list of one is only the common case - QWEN_MODEL picks out of a longer list.
    """
    global MODEL
    TR.info("preflight", "GET /v1/models", url=f"{BASE_URL}/v1/models")
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/models",
                                     headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        TR.debug("preflight", "models response",
                 body=json.dumps(data, indent=2, default=str))
        served = data["data"]
        if MODEL:
            m = next((x for x in served if x["id"] == MODEL), None)
            if m is None:
                raise SystemExit(
                    f"QWEN_MODEL={MODEL!r} is not served by {BASE_URL}. "
                    f"Available: {', '.join(x['id'] for x in served) or '(none)'}"
                )
        else:
            m = served[0]
        MODEL = m["id"]
        reported = m.get("max_model_len") or m.get("max_input_tokens")
        n_ctx = int(reported or N_CTX_FALLBACK)
        TR.info("preflight", "server ready", model=MODEL, n_ctx=n_ctx,
                from_server=bool(reported))
        return MODEL, n_ctx
    except SystemExit:
        raise
    except Exception as e:
        TR.error("preflight", "cannot reach the model server",
                 error=f"{type(e).__name__}: {e}")
        raise SystemExit(
            f"Cannot reach the model server at {BASE_URL}: {e}\n"
            f"Start it, or set QWEN_BASE_URL (with or without a /v1 suffix)."
        )


def stage_jpeg(path: Path, out: Path, what: str) -> Path:
    """The path the run should read `what` from: `path` itself when the tools
    can already open it, or a JPEG decoded from it into the run folder when it
    is a HEIC straight off the phone.

    This is what lets one command take the photo as the phone saved it. Before
    it, the operator ran scripts/heic_to_jpg.py by hand, from the right folder,
    to the right name, and then ran the harness - two commands whose only
    coupling was memory. The operator's file is read, never written.
    """
    if not is_heif(path):
        return path
    try:
        heif_to_jpeg(path, out)
    except ImportError:
        raise SystemExit(f"{path.name} is a HEIC and pillow_heif is not "
                         f"installed in {sys.executable}. Either "
                         f"`pip install pillow-heif` there, or convert it "
                         f"first: scripts/heic_to_jpg.py {path}") from None
    except Exception as e:  # noqa: BLE001 - nothing to run against otherwise
        raise SystemExit(f"could not decode {what} {path}: "
                         f"{type(e).__name__}: {e}") from None
    print(c(DIM, f"  {what:<9} {path.name} -> archive/{out.name}: HEIC decoded "
                 f"to JPEG"))
    TR.info("step0", f"{what} decoded from HEIC", supplied=str(path),
            jpeg=str(out))
    return out


def prepare_source(run_dir: Path, source: Path, python: str,
                   skip: bool) -> tuple[Path, list[str]]:
    """Segment the off-set photo into archive/source_clean.jpg.

    Returns (what generate.py will send as image 1, gate failures).

    A failure here is survivable: segmentation is a background drop, and a run
    against the raw photo still produces a garment - it just arrives with the
    room still in it, which the model can see for itself. So a broken service
    degrades rather than stopping the run. What does NOT degrade is a result that
    came back with most of the garment missing; segment.py rejects that itself
    and the caller decides.
    """
    arch = run_dir / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    out = arch / "source_clean.jpg"

    if skip:
        shutil.copy2(source, out)
        print(c(YEL, "  --no-pre-clean: sending the raw photo, background and "
                     "all."))
        TR.warn("step0", "pre-clean skipped", source=str(source))
        return out, []

    print(c(BOLD, "\nstep 0") + c(DIM, "  segmenting the source"))
    rc = stream_subprocess(
        [python, str(HERE / "tools" / "segment.py"),
         "--run", str(run_dir), "--off-set", str(source), "--out", str(out)],
        cwd=HERE, comp="step0")

    if rc == 0 and out.exists():
        return out, []

    rejected = out.with_suffix(".seg_rejected.jpg")
    if rejected.exists():
        # The service answered and the answer was wrong - most of the garment
        # gone. Generating from that spends the whole budget on a source the
        # pipeline already knows is damaged, and every check downstream then
        # agrees the garment always looked like that.
        shutil.copy2(source, out)
        return out, [f"the segmenter returned an image missing most of the "
                     f"garment; it is at {rejected.name} for inspection"]

    shutil.copy2(source, out)
    print(c(YEL, "  segmentation unavailable - continuing from the raw photo."))
    TR.warn("step0", "segmentation failed; using the raw photo", rc=rc)
    return out, []


def prepare_reference(run_dir: Path, reference: Path,
                      tone_match: bool = True) -> Path:
    """Install the operator's reference, desaturated and re-toned, and keep the
    colour original.

    Two files, and the operator's own file is neither of them - it is read, never
    written:

      archive/reference_greyscale.jpg   what the model is shown as image 2
      archive/reference_original.jpg    the file as supplied, untouched

    Two things are done TO the reference, and they are the only two. Greyscale,
    because it is a shape and lay reference, never a colour target, and sent in
    colour it is read as one. And re-toned, because greyscale removes hue and
    keeps lightness, and lightness is copied off image 2 just as readily -
    runs/20260901_222258 put an L* 80 reference in front of an L* 45 garment
    and every candidate shipped pale, dE 37 from the product. The reference
    garment is scaled in linear light until its mean lightness equals the
    source garment's, measured on archive/source_clean.jpg; the plate is left
    alone. Same function as the library path uses, so turn 1 cannot tell which
    way the reference arrived.

    The colour copy exists because the desaturated one cannot answer "what did
    the reference actually look like" - which is the first question asked when a
    candidate's tone drifts, and the run folder could not answer it on its own.
    """
    arch = run_dir / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    out = arch / REFERENCE_GREY
    original = arch / REFERENCE_ORIGINAL
    try:
        # A HEIC reference was decoded straight into this slot by stage_jpeg(),
        # in which case the colour original is already in place.
        if reference.resolve() != original.resolve():
            shutil.copy2(reference, original)
    except OSError as e:
        TR.warn("step0", f"could not keep a colour copy of the reference: {e}")
    try:
        match_to = (arch / "source_clean.jpg") if tone_match else None
        rec = prepare_reference_image(reference, out, match_to)
        if not tone_match:
            rec["why"] = "--no-tone-match"
        if rec.get("applied"):
            print(c(DIM, f"  reference {reference.name} -> {REFERENCE_GREY}: "
                         f"desaturated, garment re-toned "
                         f"{rec['grey_before']:.0f} -> {rec['grey_after']:.0f} "
                         f"to match image 1 (source {rec['target_grey']:.0f})"
                         + (f", SHORT by {rec['shortfall']:.0f} levels"
                            if rec.get("shortfall") else "")
                         + ". The colour original is kept beside it."))
        else:
            print(c(DIM, f"  reference {reference.name} -> {REFERENCE_GREY}: "
                         f"desaturated; tone left alone ({rec.get('why')}). "
                         f"The colour original is kept beside it."))
        TR.info("step0", "reference prepared", tone=rec)
    except Exception as e:  # noqa: BLE001 - a copy still beats no reference
        TR.warn("step0", f"could not desaturate the reference: {e}")
        shutil.copy2(reference, out)
    return out


def library_has_images(library: Path) -> int:
    """How many images the reference library holds. 0 for a missing folder."""
    if not library.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for p in library.rglob("*")
               if p.is_file() and p.suffix.lower() in exts
               and not p.name.startswith("."))


def select_reference(run_dir: Path, library: Path, python: str,
                     args) -> tuple[int, dict]:
    """Search the library for this garment's lay reference. (rc, record).

    THREE outcomes, not two, and the caller has to keep them apart: 0 found one,
    2 searched and found nothing close enough, anything else broke. Collapsing
    the last two into "no reference" reports a server that was down as "the
    library holds nothing of this garment", which sends someone to shoot a new
    laydown over a network blip. They also exit differently - 20 against 1.

    Runs AFTER prepare_source and matches against archive/source_clean.jpg, the
    segmented image - never the raw input. Every library asset is a garment on a
    white plate, so scoring a raw phone photo against them charges the query for
    its room and its hang tag, and that score is the number the whole run is
    anchored to. Under --no-pre-clean that file is a copy of the raw photo, and
    the run says so rather than pretending otherwise.

    The tool writes the reference itself, greyscale, straight to
    archive/reference_greyscale.jpg - the same path and the same treatment
    prepare_reference() gives an operator-supplied file, so turn 1 cannot tell
    which way the reference arrived.
    """
    argv = ["--run", str(run_dir), "--library", str(library),
            "--threshold", str(args.reference_threshold),
            "--top-k", str(args.reference_top_k)]
    if args.no_pre_clean:
        argv.append("--query-raw")
    if args.no_reference_veto:
        argv.append("--no-model-veto")
    if args.reference_silhouette:
        argv.append("--silhouette")
    if args.no_tone_match:
        argv.append("--no-tone-match")

    rc = stream_subprocess([python, str(HERE / "tools" / "select_reference.py"),
                            *argv], cwd=HERE, comp="reference")

    record = {}
    f = run_dir / "reference_selection.json"
    if f.exists():
        try:
            record = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            TR.warn("step0", f"reference_selection.json unreadable: {e}")

    # A 0 that did not leave a reference on disk is a failure however the tool
    # exited: turn 1 pins that file into the conversation, so a run that starts
    # without it has no image 2 and no way to say what it is aiming at.
    if rc == 0 and not reference_path(run_dir).exists():
        TR.error("step0", "the search reported success but wrote no reference")
        return 1, record
    return rc, record


def resolve_reference(args, workspace: Path) -> tuple[Path | None, Path | None]:
    """What to use as the reference: (explicit file, library to search).

    Exactly one is not None. The rule is the absence of --reference, which is
    why that flag no longer carries a default - with one, "the operator chose
    this file" and "the operator said nothing" were the same value and the
    harness could not tell them apart.

    Order: an explicit file wins outright, then the library, then the file the
    workspace has always used. The last one keeps every existing command line
    working on a checkout whose library is still empty.
    """
    if args.reference is not None:
        return args.reference, None
    library = args.reference_library
    n = library_has_images(library)
    if n:
        return None, library
    fallback = workspace / "inputs" / "reference_greyscale.jpg"
    if fallback.exists():
        print(c(YEL, f"  {library} is empty - falling back to "
                     f"{fallback.name}. Populate the library, or pass "
                     f"--reference, to stop seeing this."))
        return fallback, None
    raise SystemExit(
        f"No reference, and nothing to find one with. Either:\n"
        f"  put images in {library}\n"
        f"  pass --reference <file>\n"
        f"  put one at {fallback}")


# There is no polish step and no recolour step. Both ran here once - the
# recolour on runs/20260902_104708, the de-wrinkle on request via --polish -
# and both were taken out at the operator's request: what ships is a fal.ai
# generation exactly as it came back. tools/polish.py and tools/recolour.py
# remain on disk as manual tools, and no run calls either.

# How many ranked candidates a run delivers. The operator asked for the best
# four of each run, whatever they look like, rather than one winner: an
# alternate that a person can choose between is worth more than a second
# opinion from the model about which single image to keep.
TOP_N = 4


def generation_of(name: str) -> str:
    """cand_03, cand_03s and cand_03p are one purchase: 'cand_03'."""
    m = re.match(r"(cand_\d+)", name or "")
    return m.group(1) if m else name


def is_cutout(name: str) -> bool:
    """cand_03s, and anything made from it, is the garment on flat white with
    the plate and its shadow gone. Not a deliverable - pick_best refuses it."""
    return "s" in (name or "")[len(generation_of(name)):]


def lay_ranking(run_dir: Path, cache: dict | None = None) -> list[tuple[str, float]]:
    """(name, lay IoU vs the reference) for every candidate on disk, best
    first. Each file is measured once per mtime when a cache dict is given."""
    sys.path.insert(0, str(HERE / "tools"))
    import metrics as MET
    arch = run_dir / "archive"
    src = arch / "source_clean.jpg"
    ref = reference_path(run_dir)
    if not src.exists() or not ref.exists():
        return []
    cache = cache if cache is not None else {}
    rows = []
    for p in sorted(arch.glob("cand_*.png")):
        if not re.fullmatch(r"cand_\d+[spc]*", p.stem):
            continue
        key = (p.stem, p.stat().st_mtime_ns)
        if key not in cache:
            try:
                m = MET.compare(src, p, reference=ref)
                cache[key] = (m.get("lay_iou"), bool(m.get("cue_note")), m)
            except Exception:  # noqa: BLE001 - a ranking is not the delivery
                cache[key] = (None, True, {})
        iou, unreliable, _ = cache[key]
        if iou is not None and not unreliable:
            rows.append((p.stem, float(iou)))
    rows.sort(key=lambda r: -r[1])
    return rows


# ---------------------------------------------------------------------------
# The final judge: a pairwise knockout on rank 1
# ---------------------------------------------------------------------------
#
# runs/20260901_222258 shipped cand_08: the best lay of the run at 0.934, and
# dE 30 from the product. The reference was pale, every candidate that hit the
# pose copied its tone, and nothing objected. The agent named it knowing the
# number ("the task is about matching the reference's shape"); the hue part
# read 2.6, "the same colourway" by the rule the agent is given, because
# lightness is not hue; and the harness's own fill ranks by lay, which is
# exactly what a pale candidate wins on. The two candidates that WERE the
# product's colour, cand_05 and cand_09 at dE 6, had the hood wrong, a lay of
# 0.78, and sat outside the delivery altogether.
#
# So the last word on rank 1 is a different question, put to the same model:
# "which of these two ships" - two candidates beside the product photo, the
# product's colour a hard requirement. Measured on that run's pairs on
# 2026-09-02, each pair in both orders: six for six against the pale one. The
# retired grader failed at scoring one image 0-100 (it saturated) and at
# ranking a list of N (a slot bias); a pair with one criterion is neither.
#
# A knockout, incumbent first, and a VETO, not a re-ranking. The model's
# rank 1 is replaced only when both orders agree it is the wrong colour and
# the challenger is not; a challenger that wins both orders on lay alone
# changes nothing, and a split on order is a tie. runs/20260902_225122 is why
# the lay does not count: the judge swapped the run's best silhouette (0.970)
# for a flatter one at 0.945, and on the way called two dE 2.5 candidates
# "distinctly paler" - a flag that flipped with image order for four of the
# ten. Under dE 4 the judge has no reliable opinion; at dE 20 and 30 it has
# never once been wrong, in either order. The model's shape judgement has the
# whole run behind it and stays in charge. An unreachable server changes
# nothing either. The pool is every distinct generation on disk up to
# JUDGE_POOL, the delivery first, for the reason above: the fill's lay order
# is the wrong order here. Each duel is a form-filling call with thinking
# off, like stage C of the reference search. Every verdict is written to
# archive/judge.json.
#
# The pool cap is a safety valve, not a budget. Measured 2026-09-02 on the
# 27B vision model with thinking off: 1.5-1.9 s per call, so a pool of 16 is
# fifteen duels and under a minute. It has to be wide enough to reach the
# BOTTOM of the lay order, because that is where the product-coloured
# candidates of a pale run sit - at 8, the first test cut off cand_05 of
# runs/20260901_222258, the best colour in the batch, for having the worst lay.
JUDGE_POOL = 16
JUDGE_IMAGE_DIM = 1024

# The eye alone is not enough to throw a pick out. runs/20260902_233202: the
# model's pick had the reference's pose at lay 0.957 and read 14 dE paler
# than the product (hue 3.9, the same colourway); the judge called it "washed
# out" in both orders and shipped a candidate at the product's colour that
# had never been re-laid - size x1.55, legs splayed, lay 0.893. The operator
# called it a really bad shot, and it was: the shape is the job, and 14 dE
# of lightness is not worth it. The hoodie run that the judge exists for was
# 30 dE. So the veto needs the numbers to agree that the colour is FAR off:
# lit dE at or past this floor, or the hue part past the skill's own
# "different colourway" line. Under both, the model's pick holds however
# paler the judge reads it, and the duel is recorded as `colour_minor`.
JUDGE_DE_FLOOR = 20.0
JUDGE_HUE_FLOOR = 8.0

JUDGE_PROMPT = """\
You are the last check before one of two candidate images ships as this \
product's laydown photograph.

Image 1 is the PRODUCT, photographed off-set. It is the truth for colour, \
lightness, fabric and construction.
Image 2 is the LAYDOWN TO MATCH, in greyscale. It is the truth for pose and \
shape only. Its tone is not a target, and its construction is not to be copied.
Image 3 is candidate A. Image 4 is candidate B.

Choose the candidate to ship. Apply these rules in order:
1. It must be the product in image 1: the same hue AND the same lightness. \
Compare each candidate's fabric directly against image 1 before anything \
else. A garment that reads clearly paler, darker or a different colour than \
image 1 is disqualified, however good its pose.
2. Construction must be intact: nothing invented (a seam, a label, a tag, a \
hanger, a second garment) and nothing lost (a logo, a pocket, a drawstring).
3. Among candidates that pass 1 and 2, the one whose lay is closest to image \
2 wins: the same pose and arrangement of sleeves, hood, hem and legs, a \
similar size in the frame, flatter.
If both fail rule 1, the one closer to image 1's colour wins. If both fail \
rule 2, the one with less damage wins.

Return ONE JSON object, nothing else:
{"winner": "A" or "B",
 "colour_ok": {"A": true or false, "B": true or false},
 "reason": "<one sentence naming what decided it>"}"""


def _json_blob(text: str) -> dict:
    """The first {...} in a reply, fences and prose stripped."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in the reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def judge_duel(src_b64: str, ref_b64: str, a: tuple[str, str],
               b: tuple[str, str]) -> dict:
    """One vision call: `a` shown as candidate A, `b` as candidate B, each a
    (name, base64) pair. Returns the verdict with `winner` as a candidate
    NAME, or None with `error` set. Never raises - the judge is not the
    delivery, and a run that reaches this point has already paid for its
    images."""
    def txt(s: str) -> dict:
        return {"type": "text", "text": s}
    content = [txt("Image 1 - the PRODUCT, photographed off-set:"),
               image_block(src_b64),
               txt("Image 2 - the LAYDOWN TO MATCH, greyscale:"),
               image_block(ref_b64),
               txt("Image 3 - candidate A:"), image_block(a[1]),
               txt("Image 4 - candidate B:"), image_block(b[1]),
               txt(JUDGE_PROMPT)]
    out = {"A": a[0], "B": b[0], "winner": None, "colour_ok": None,
           "reason": None, "error": None, "seconds": 0.0}
    t0 = time.time()
    try:
        data = call_model([{"role": "user", "content": content}], tools=None,
                          max_tokens=600, temperature=0.0, retries=1,
                          think=False)
        text = (data["choices"][0]["message"].get("content") or "").strip()
        rec = _json_blob(text)
        label = str(rec.get("winner", "")).strip().upper()
        if label not in ("A", "B"):
            raise ValueError(f"winner is {rec.get('winner')!r}, not A or B")
        out["winner"] = a[0] if label == "A" else b[0]
        ok = rec.get("colour_ok")
        if isinstance(ok, dict):
            out["colour_ok"] = {a[0]: ok.get("A"), b[0]: ok.get("B")}
        out["reason"] = str(rec.get("reason") or "").strip()
    except Exception as e:  # noqa: BLE001 - recorded, never raised
        out["error"] = f"{type(e).__name__}: {e}"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def judge_final(run_dir: Path, pool: list[Path],
                numbers: dict[str, tuple] | None = None) -> dict:
    """Run the knockout over `pool`, pool[0] the incumbent. Returns the record
    that lands in archive/judge.json: the pool, every duel with both calls,
    the winner, and `skipped` with the reason when nothing was asked.

    A challenger takes the title only when it wins both orders AND both
    orders flag the incumbent's colour as wrong and its own as right AND the
    incumbent's measured colour is far off - `numbers` maps a candidate name
    to (lit dE, hue dE) against the product, and one of them has to clear
    its floor. A candidate with no numbers is judged by eye alone. A win on
    lay alone is recorded as `lay_only`, a colour call the numbers do not
    back as `colour_minor`, and the incumbent holds in both; a split on order
    is a tie. The first duel that errors ends the judging - a server that has
    gone away would otherwise be waited on once per remaining challenger, at
    the end of a run that is otherwise complete.
    """
    numbers = numbers or {}
    rec = {"pool": [p.stem for p in pool],
           "incumbent": pool[0].stem if pool else None,
           "winner": None, "overrode": False, "duels": [], "skipped": None,
           "floors": {"de_lit": JUDGE_DE_FLOOR, "hue": JUDGE_HUE_FLOOR},
           "seconds": 0.0, "at": datetime.now().isoformat(timespec="seconds")}
    if len(pool) < 2:
        rec["skipped"] = "fewer than two distinct candidates"
        return rec
    arch = run_dir / "archive"
    src, ref = arch / "source_clean.jpg", reference_path(run_dir)
    if not src.exists():
        rec["skipped"] = "no archive/source_clean.jpg to hold the colour against"
        return rec
    if not ref.exists():
        rec["skipped"] = f"no {ref.name} to hold the lay against"
        return rec

    t0 = time.time()
    b64: dict[Path, str | None] = {}

    def enc(p: Path) -> str | None:
        if p not in b64:
            b64[p] = encode_image(p, JUDGE_IMAGE_DIM)
        return b64[p]

    if enc(src) is None or enc(ref) is None:
        rec["skipped"] = "could not encode the source or the reference"
        return rec

    champion = pool[0]
    for challenger in pool[1:]:
        if enc(champion) is None or enc(challenger) is None:
            rec["duels"].append({"incumbent": champion.stem,
                                 "challenger": challenger.stem,
                                 "outcome": "error",
                                 "error": "could not encode an image"})
            continue
        first = judge_duel(enc(src), enc(ref), (champion.stem, enc(champion)),
                           (challenger.stem, enc(challenger)))
        second = judge_duel(enc(src), enc(ref),
                            (challenger.stem, enc(challenger)),
                            (champion.stem, enc(champion)))
        wins = [first["winner"], second["winner"]]

        def wrong_in_both(name: str) -> bool:
            # Explicitly False in both orders. The flag flips with image order
            # at small differences, so one order's "wrong" is not a finding.
            return all((d["colour_ok"] or {}).get(name) is False
                       for d in (first, second))

        inc_wrong = wrong_in_both(champion.stem)
        ch_wrong = wrong_in_both(challenger.stem)
        inc_de, inc_hue = numbers.get(champion.stem, (None, None))
        far_off = ((inc_de is None and inc_hue is None)
                   or (inc_de is not None and inc_de >= JUDGE_DE_FLOOR)
                   or (inc_hue is not None and inc_hue >= JUDGE_HUE_FLOOR))
        if None in wins:
            outcome = "error"
        elif (wins == [challenger.stem] * 2 and inc_wrong and not ch_wrong
              and far_off):
            outcome = "challenger"
        elif wins == [challenger.stem] * 2 and inc_wrong and not ch_wrong:
            outcome = "colour_minor"
        elif wins == [challenger.stem] * 2:
            outcome = "lay_only"
        elif wins == [champion.stem] * 2:
            outcome = "holds"
        else:
            outcome = "split"
        duel = {"incumbent": champion.stem, "challenger": challenger.stem,
                "outcome": outcome, "incumbent_colour_wrong": inc_wrong,
                "challenger_colour_wrong": ch_wrong,
                "incumbent_de_lit": inc_de, "incumbent_hue_de": inc_hue,
                "calls": [first, second]}
        rec["duels"].append(duel)
        TR.info("judge", f"{champion.stem} vs {challenger.stem}: {outcome}",
                first=first["winner"], second=second["winner"],
                incumbent_colour_wrong=inc_wrong,
                challenger_colour_wrong=ch_wrong,
                reasons=[first["reason"], second["reason"]],
                errors=[first["error"], second["error"]],
                seconds=first["seconds"] + second["seconds"])
        head = f"    {champion.stem} vs {challenger.stem}: "
        if outcome == "error":
            err = first["error"] or second["error"]
            print(c(YEL, head + f"the call failed ({err}); {champion.stem} "
                                f"holds and the judging stops here"))
            rec["skipped"] = (f"stopped after an error on {challenger.stem}: "
                              f"{err}")
            break
        if outcome == "challenger":
            why = first["reason"] or second["reason"] or ""
            measured = (f"dE {inc_de:.0f}, hue {inc_hue:.1f}"
                        if inc_de is not None else "unmeasured")
            print(c(YEL, head + f"{challenger.stem} takes rank 1 - both "
                                f"orders read {champion.stem} as the wrong "
                                f"colour ({measured}): {why}"))
            champion = challenger
        elif outcome == "colour_minor":
            print(c(DIM, head + f"{champion.stem} holds - read as the wrong "
                                f"colour in both orders, but it measures dE "
                                f"{inc_de:.0f}, hue {inc_hue:.1f}, under the "
                                f"{JUDGE_DE_FLOOR:.0f} / {JUDGE_HUE_FLOOR:.0f} "
                                f"floors, and the shape is the job"))
        elif outcome == "lay_only":
            print(c(DIM, head + f"{challenger.stem} won on lay alone; lay is "
                                f"the model's call, {champion.stem} holds"))
        elif outcome == "holds" and ch_wrong:
            print(c(DIM, head + f"{champion.stem} holds - both orders read "
                                f"{challenger.stem} as the wrong colour"))
        elif outcome == "holds":
            print(c(DIM, head + f"{champion.stem} holds"))
        else:
            print(c(DIM, head + f"split on order, {champion.stem} holds"))
    rec["winner"] = champion.stem
    rec["overrode"] = champion.stem != pool[0].stem
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def ship_best(run_dir: Path, top: int = TOP_N,
              judge: bool = True) -> tuple[list[str], list[str], str]:
    """Write output/best.png and best_2.png .. best_<top>.png, plus picks.json.
    Returns (candidates in rank order, who chose each rank, note); the chooser
    is "model", "harness" or "judge".

    UNTOUCHED GENERATIONS ONLY: cand_NN as fal.ai returned it. The segmented
    form (cand_NNs) is a cutout with the plate and its shadow gone, and the
    polished and recoloured forms are a second model's edit; none of them
    ships, whoever names them. pick_best refuses the names, and a legacy
    best.json that carries one is read past it here.

    Runs only when at least one candidate exists, so a run that generated nothing
    ends with an empty output/ and says so, rather than shipping a placeholder
    that a downstream reader would take for a delivery.

    The model's ranked list comes first, one slot per generation. Slots it left
    empty are filled from the harness's own lay ranking - closest silhouette to
    the reference first - and marked as the harness's choice; with no reference
    to rank against, newest first. There is no minimum: the operator asked for
    the best four of whatever the run produced. Fewer than `top` candidates on
    disk means fewer files, never a duplicate.

    Then, with `judge`, rank 1 defends its place pairwise against every other
    candidate - see judge_final(). It loses only to a challenger both orders
    agree is the product's colour when it is not, and only when its own
    measured colour is far off (JUDGE_DE_FLOOR / JUDGE_HUE_FLOOR): a paler
    render with the right shape stays. A winner from outside the delivery
    displaces the last slot; one from inside it moves up. Ranks 2 onward keep
    their order either way: the judge answers "is best.png the right colour",
    not "sort these four".
    """
    arch, outdir = run_dir / "archive", run_dir / "output"
    cands = sorted(p for p in arch.glob("cand_*.png")
                   if re.fullmatch(r"cand_\d+", p.stem))
    if not cands:
        return [], [], ""

    ranked: list[Path] = []
    chosen_by: list[str] = []
    used: set[str] = set()

    def take(p: Path, who: str) -> None:
        gen = generation_of(p.stem)
        if gen in used or len(ranked) >= top:
            return
        used.add(gen)
        ranked.append(p)
        chosen_by.append(who)

    bf = arch / "best.json"
    if bf.exists():
        try:
            rec = json.loads(bf.read_text())
            names = rec.get("candidates") or ([rec["candidate"]]
                                             if rec.get("candidate") else [])
            for n in names:
                p = arch / f"{n}.png"
                if p.exists() and re.fullmatch(r"cand_\d+", str(n)):
                    take(p, "model")
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    # Lay order first, then newest first: the model's last attempt is its
    # most informed one. Not a judgement - better than nothing, and recorded
    # as the harness's choice. Computed even when the delivery is already
    # full, because the judge's pool is drawn from the same order.
    cache: dict = {}
    fallback = [arch / f"{n}.png" for n, _ in lay_ranking(run_dir, cache)
                if re.fullmatch(r"cand_\d+", n)]
    fallback += list(reversed(cands))
    # The same measurements, for the judge's colour floors.
    numbers = {stem: (m.get("colour_de_lit"), m.get("hue_de"))
               for (stem, _), (_, _, m) in cache.items() if m}
    if len(ranked) < top:
        for p in fallback:
            if p.exists():
                take(p, "harness")

    notes = []
    if judge and len({generation_of(p.stem) for p in cands}) > 1:
        pool = list(ranked)
        gens = {generation_of(p.stem) for p in pool}
        for p in fallback:
            if len(pool) >= JUDGE_POOL:
                break
            g = generation_of(p.stem)
            if p.exists() and g not in gens:
                pool.append(p)
                gens.add(g)
        print(c(BOLD, "\nfinal step") + c(DIM, f"  is {ranked[0].stem} the "
                                            f"product's colour? pairwise "
                                            f"against {len(pool) - 1} other(s)"))
        try:
            jrec = judge_final(run_dir, pool, numbers)
        except Exception as e:  # noqa: BLE001 - the judge is not the delivery
            TR.exception("judge", f"the judge crashed: {e}")
            jrec = {"pool": [p.stem for p in pool], "incumbent": pool[0].stem,
                    "winner": None, "overrode": False, "duels": [],
                    "skipped": f"crashed: {type(e).__name__}: {e}"}
        try:
            (arch / "judge.json").write_text(json.dumps(jrec, indent=2) + "\n")
        except OSError as e:
            TR.warn("judge", f"could not write judge.json: {e}")
        winner = jrec.get("winner")
        if winner and winner != ranked[0].stem:
            wp = arch / f"{winner}.png"
            if wp in ranked:
                i = ranked.index(wp)
                ranked.pop(i)
                chosen_by.pop(i)
            elif len(ranked) >= top:
                ranked.pop()
                chosen_by.pop()
            ranked.insert(0, wp)
            chosen_by.insert(0, "judge")
            notes.append(f"the judge replaced {jrec['incumbent']} with "
                         f"{winner} at rank 1: both orders read "
                         f"{jrec['incumbent']} as the wrong colour - see "
                         f"archive/judge.json")
            print(c(YEL, f"    verdict: {winner} ships, not "
                         f"{jrec['incumbent']} - the wrong colour"))
        elif jrec.get("skipped") and not jrec["duels"]:
            notes.append(f"judge skipped: {jrec['skipped']}")
            print(c(DIM, f"    judge skipped: {jrec['skipped']}"))
        else:
            fought = len(jrec["duels"])
            notes.append(f"the judge upheld {ranked[0].stem} as the product's "
                         f"colour against {fought} challenger(s)"
                         + (f" ({jrec['skipped']})" if jrec.get("skipped")
                            else ""))
            print(c(DIM, f"    verdict: {ranked[0].stem} holds - the "
                         f"product's colour"
                         f"{' (' + jrec['skipped'] + ')' if jrec.get('skipped') else ''}"))
    else:
        jrec = None

    filled = chosen_by.count("harness")
    if filled:
        notes.append(f"{filled} slot(s) filled by the harness from its lay "
                     f"ranking, not chosen by the model")
    if len(ranked) < top:
        notes.append(f"only {len(ranked)} distinct candidate(s) on disk, so "
                     f"{top - len(ranked)} slot(s) are empty")
    note = "; ".join(n for n in notes if n)

    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("best*.png"):
        stale.unlink()
    sys.path.insert(0, str(HERE / "tools"))
    import common as C
    import metrics as MET
    src = arch / "source_clean.jpg"
    ref = reference_path(run_dir)
    picks = []
    for i, p in enumerate(ranked, 1):
        out = outdir / ("best.png" if i == 1 else f"best_{i}.png")
        shutil.copy2(p, out)
        row = {"rank": i, "candidate": p.stem, "file": out.name,
               "chosen_by": chosen_by[i - 1]}
        try:
            if src.exists():
                m = MET.compare(src, p, reference=ref if ref.exists() else None)
                row.update({k: m.get(k) for k in
                            ("lay_iou", "size_vs_ref", "wrinkle_ratio",
                             "colour_de_lit", "hue_de", "silhouette_iou")})
        except Exception:  # noqa: BLE001 - the file is written either way
            pass
        picks.append(row)
    (outdir / "picks.json").write_text(json.dumps(
        {"top": top, "picks": picks, "note": note,
         "judge": (None if jrec is None else
                   {k: jrec.get(k) for k in ("incumbent", "winner", "overrode",
                                             "skipped", "seconds")}
                   | {"duels": len(jrec.get("duels") or [])}),
         "at": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")
    try:
        C.log(run_dir, f"shipped {', '.join(p.stem for p in ranked)} as "
                       f"best.png .. best_{len(ranked)}.png"
                       + (f" ({note})" if note else ""))
    except Exception:  # noqa: BLE001 - the files are written either way
        pass
    return [p.stem for p in ranked], chosen_by, note


def exit_code_for(result: dict, budget: int, best_shipped: bool) -> int:
    """What a finish status means to a caller. See DOCKER.md for the same table.

    The awkward case is `no_candidates`, and it is awkward because the obvious
    test - "0 if best.png exists" - cannot ever be true when nothing was
    generated, so it would mark every such run a failure including the ones that
    did exactly what they were told. It splits on INTENT instead: a run
    configured with a zero image budget succeeded at buying zero images; a run
    that had budget and still produced nothing did not.
    """
    status = result.get("status")
    if status == "done":
        return 0
    if status == "no_candidates":
        return 0 if budget == 0 else 1
    if status in ("budget_exhausted", "gave_up"):
        return 0 if best_shipped else 1
    return 1          # error, blocked, interrupted, max_iters


def main():
    ap = argparse.ArgumentParser(
        description="Iterate an off-set garment photo towards a reference laydown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              ./run.sh --yolo
              ./run.sh --source inputs/off_set_image.jpg \\
                       --reference inputs/reference_greyscale.jpg --max-images 6
              ./run.sh --max-images 0 --yolo    # free: runs the loop, buys nothing
        """),
    )
    ap.add_argument("--source", type=Path,
                    default=HERE / "inputs" / "off_set_image.jpg",
                    help="The off-set photo to re-lay.")
    # NO DEFAULT, on purpose. Its absence is what triggers the library search,
    # and a default made "the operator chose this file" indistinguishable from
    # "the operator said nothing". See resolve_reference().
    ap.add_argument("--reference", type=Path, default=None,
                    help="The laydown to aim at. Desaturated before use - it is "
                         "a shape reference, never a colour target. Omit it and "
                         "the reference is chosen from --reference-library.")
    ap.add_argument("--reference-library", type=Path,
                    default=HERE / "inputs" / "reference_library",
                    help="Searched for a reference when --reference is not "
                         "given. tools/select_reference.py does the choosing; "
                         "run it with --index after adding images so a live run "
                         "does not pay to describe them.")
    # Kept in step with select_reference.DEFAULT_THRESHOLD by hand rather than
    # imported, because harness.py builds its parser before it has put tools/ on
    # the path. If they drift, this one wins - it is passed explicitly on every
    # run the harness starts.
    ap.add_argument("--reference-threshold", type=float, default=78.0,
                    metavar="SCORE",
                    help="How well a library image must match to be used "
                         "(default 78). PROVISIONAL - measured on nine assets, "
                         "not on your library. Run tools/select_reference.py "
                         "--calibrate and set this from the numbers.")
    # 3 is the endpoint's cap, not a taste: it takes at most FOUR images in
    # one request, and stage C sends the garment plus one per survivor. At the
    # old default of 5 every run with a library big enough to qualify four
    # was a guaranteed 400, and the search fell back - silently - to a contact
    # sheet where each candidate arrives at a fraction of the resolution.
    # Measured 2026-09-02: six images fails, four is accepted.
    ap.add_argument("--reference-top-k", type=int, default=3, metavar="N",
                    help="How many survivors the model sees before it confirms "
                         "or vetoes the pick (default 3, the most the endpoint "
                         "takes beside the garment in one request; more falls "
                         "back to a low-resolution contact sheet).")
    ap.add_argument("--no-reference-veto", action="store_true",
                    help="Install the top-scoring library image even when the "
                         "model rejects it on sight. Trusts the score alone.")
    ap.add_argument("--reference-silhouette", action="store_true",
                    help="Install a line drawing of the chosen reference's "
                         "OUTLINE instead of the photograph. Same pose, no "
                         "construction, so nothing can bleed across. Check "
                         "archive/reference_greyscale.jpg before trusting it: the "
                         "outline "
                         "is built from tone, so a striped or colour-blocked "
                         "reference comes back with its dark bands cut out.")
    ap.add_argument("--no-tone-match", action="store_true",
                    help="Leave the reference garment at its own lightness. By "
                         "default it is scaled in linear light to the source "
                         "garment's mean L* before turn 1, because greyscale "
                         "removes hue and not tone, and the generator copies "
                         "tone off image 2 - one run shipped dE 37 from an "
                         "L* 80 reference on an L* 45 garment. Applies to both "
                         "a library pick and --reference.")
    ap.add_argument("--max-images", type=int, default=None, metavar="N",
                    help=f"fal.ai images for the whole run. Default "
                         f"{DEFAULT_BUDGET} (LAYDOWN_MAX_IMAGES). 0 exercises "
                         f"the loop and bills nothing.")
    ap.add_argument("--skill", help="Skill name under skills/<name>/SKILL.md, or a path.")
    ap.add_argument("--skill-file", help="Explicit path to a SKILL.md.")
    ap.add_argument("--task", help="What to do. Defaults to the skill's own goal.")
    ap.add_argument("--workspace", type=Path, default=HERE)
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--yolo", action="store_true",
                    help="Run tools without asking. You are trusting the model "
                         "with a shell. Required when stdin is not a terminal.")
    ap.add_argument("--allow-outside", action="store_true",
                    help="Permit writes outside the workspace.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show a slice of the model's reasoning each turn.")
    ap.add_argument("--no-pre-clean", action="store_true",
                    help="Skip segmentation and send the raw photo, background "
                         "and all.")
    ap.add_argument("--top", type=int, default=TOP_N, metavar="N",
                    help=f"How many ranked candidates to deliver (default "
                         f"{TOP_N}): output/best.png, best_2.png ...")
    ap.add_argument("--no-judge", action="store_true",
                    help="Skip the final colour check on rank 1. By default "
                         "the model is shown the product beside rank 1 and "
                         "each other candidate in turn, both orders, and rank "
                         "1 is replaced only when both orders agree it is the "
                         "wrong colour and the other is not. Lay stays the "
                         "model's call. A few vision calls at the end of the "
                         "run, nothing billed.")
    ap.add_argument("--allow-dirty-source", action="store_true",
                    help=f"Start the agent even when the segmenter returned an "
                         f"image missing most of the garment. Without this the "
                         f"run stops before the agent and exits "
                         f"{EXIT_UNCLEAN_SOURCE}, because every image generated "
                         f"from that source inherits the loss.")
    ap.add_argument("--trace-level", default="DEBUG",
                    choices=["DEBUG", "INFO", "WARN", "ERROR"],
                    help="Detail written to runs/<session>/run.log.")
    ap.add_argument("--no-trace", action="store_true",
                    help="Do not write runs/<session>/run.log.")
    args = ap.parse_args()

    if args.no_trace:
        TR.enabled = False
    else:
        TR.tee()
    TR.info("harness", "invoked", argv=" ".join(sys.argv[1:]), pid=os.getpid(),
            cwd=os.getcwd(), python=sys.executable)

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise SystemExit(f"No such workspace: {workspace}")
    if not args.source.exists():
        raise SystemExit(f"No source photo: {args.source}")
    # Resolved before anything else spends time: an unusable reference setup is
    # the cheapest possible failure and belongs next to the missing-source one.
    ref_file, ref_library = resolve_reference(args, workspace)
    if ref_file is not None and not ref_file.exists():
        raise SystemExit(f"No reference: {ref_file}")

    budget = DEFAULT_BUDGET if args.max_images is None else max(0, args.max_images)
    # The child scripts read this, and generate.py treats it as a ceiling its own
    # --max-total can lower but never raise. Set here so a model that reaches for
    # bash instead of the generate tool meets the same limit.
    os.environ["LAYDOWN_MAX_IMAGES"] = str(budget)

    skill_path = resolve_skill(workspace, args.skill, args.skill_file)
    skill_text, skill_desc = load_skill(skill_path) if skill_path else ("", "")
    task = args.task or (f"{skill_desc}\n\n{DEFAULT_TASK}" if skill_desc
                         else DEFAULT_TASK)

    # One run means one folder, and run.sh stamps it. Deriving a second stamp
    # here put the transcript and the tools' own steps.log in folders that only
    # agreed because they were a second apart.
    stamp = os.environ.get("LAYDOWN_SESSION") or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["LAYDOWN_SESSION"] = stamp
    run_dir = workspace / "runs" / stamp
    (run_dir / "archive").mkdir(parents=True, exist_ok=True)

    # A HEIC off the phone is decoded here, once, into the run folder. From this
    # line on `source` and `ref_file` are files every tool can open; args.source
    # stays what the operator typed, for the log.
    source = stage_jpeg(args.source, run_dir / "archive" / SOURCE_ORIGINAL,
                        "source")
    if ref_file is not None:
        ref_file = stage_jpeg(ref_file, run_dir / "archive" / REFERENCE_ORIGINAL,
                              "reference")

    # Attached before preflight, not after: a run that dies because the server is
    # unreachable is exactly the one someone wants a trace of.
    if TR.enabled:
        TR.attach(run_dir / "run.log", level=args.trace_level, header={
            "session": stamp,
            "argv": " ".join(sys.argv[1:]),
            "workspace": workspace,
            "source": args.source if source == args.source
                      else f"{args.source} -> {source}",
            "reference": ref_file or f"(search {ref_library})",
            "budget": budget,
            "skill": skill_path or "(none)",
            "server": BASE_URL,
            "python": sys.executable,
            "pid": os.getpid(),
        })

    model, n_ctx = preflight()

    tools = Tools(workspace, run_dir, args.allow_outside,
                  source_photo=source)
    TR.info("harness", "child interpreter for the tools", python=tools.python)

    # Step 0, before the Session is built: turn 1 puts both images into the
    # conversation, so they have to exist on disk by then.
    clean, gate_fails = prepare_source(run_dir, source, tools.python,
                                       args.no_pre_clean)
    if gate_fails and not args.allow_dirty_source:
        print(c(RED, "\nsegmentation gate FAILED - the result is not the same "
                     "garment."))
        for f in gate_fails:
            print(c(DIM, f"    {f}"))
        print(c(DIM, "  The run stops before the agent starts, because every "
                     "image generated from this source would inherit the loss."))
        print(c(DIM, "  --allow-dirty-source runs anyway; --no-pre-clean skips "
                     "segmentation entirely."))
        TR.error("step0", "segmentation gate failed", fails=len(gate_fails))
        return EXIT_UNCLEAN_SOURCE
    if gate_fails:
        print(c(YEL, "\n  --allow-dirty-source: continuing from a source the "
                     "gate rejected."))
        for f in gate_fails:
            print(c(DIM, f"    {f}"))

    # The reference, after the source and never before it: the search matches
    # against the SEGMENTED photo, so it needs prepare_source to have run.
    selection = {}
    if ref_library is not None:
        rc, selection = select_reference(run_dir, ref_library, tools.python,
                                         args)
        if rc == 2:
            print(c(RED, "\nNO REFERENCE - the library holds nothing that can "
                         "serve as this garment's laydown."))
            closest = (selection.get("closest") or {})
            if selection.get("model_vetoed"):
                print(c(DIM, f"    {selection.get('n_qualifying')} candidate(s) "
                             f"cleared {args.reference_threshold:.0f}, but the "
                             f"model rejected them on sight"))
            elif closest.get("file"):
                print(c(DIM, f"    closest was {closest['file']} at "
                             f"{closest.get('score')}, needed "
                             f"{args.reference_threshold:.0f}"))
            if selection.get("contact_sheet"):
                print(c(DIM, f"    near misses: {selection['contact_sheet']}"))
            print(c(DIM, "  Nothing was generated and nothing was billed. Add "
                         "a suitable laydown to the library, or pass "
                         "--reference to supply one directly."))
            TR.error("step0", "no reference found in the library",
                     closest=closest.get("score"),
                     vetoed=selection.get("model_vetoed"),
                     threshold=args.reference_threshold)
            return EXIT_NO_REFERENCE
        if rc != 0:
            # A broken search, not an empty library. Exits 1 like any other
            # broken step, and says which it was so nobody goes looking for a
            # garment to photograph over an unreachable server.
            print(c(RED, f"\nthe reference search FAILED (exit {rc}). This is "
                         f"not 'no match found' - the step itself broke."))
            print(c(DIM, f"  see {run_dir / 'run.log'}"))
            return 1
        ref_file = Path(selection["source"])
    else:
        prepare_reference(run_dir, ref_file, tone_match=not args.no_tone_match)

    approver = Approver(args.yolo)
    sess = Session(task, skill_text, workspace, run_dir, tools, approver,
                   args.max_iters, args.verbose, n_ctx, budget=budget)

    print(c(BOLD, "\nlaydown harness"))
    print(c(DIM, f"  model     {model}"))
    print(c(DIM, f"  context   {n_ctx} tokens (compacting past {sess.compact_at})"))
    print(c(DIM, f"  server    {BASE_URL}"))
    print(c(DIM, f"  source    {args.source.name}"))
    if selection:
        print(c(DIM, f"  reference {ref_file.name}  "
                     f"({selection.get('score')}/100 from "
                     f"{ref_library.name}/, {selection.get('library_count')} "
                     f"images)"))
        runner = selection.get("runner_up") or {}
        if runner.get("file"):
            print(c(DIM, f"            runner-up {runner['file']} "
                         f"({runner.get('score')})"))
        risk = selection.get("construction_risk") or {}
        if risk.get("flagged"):
            print(c(YEL, f"            bleed risk: "
                         f"{', '.join(risk.get('terms', []))} - named in the "
                         f"opening brief"))
        tone = selection.get("tone_match") or {}
        if tone.get("applied"):
            print(c(DIM, f"            tone: garment re-toned "
                         f"{tone['grey_before']:.0f} -> {tone['grey_after']:.0f} "
                         f"to match the source"))
        else:
            print(c(YEL, f"            tone: left alone - {tone.get('why')}"))
    else:
        print(c(DIM, f"  reference {ref_file.name}"))
    print(c(DIM, f"  budget    {budget} image(s)"
                 + ("  - nothing will be billed" if budget == 0 else
                    f"  (~{budget * 15}c at 2K)")))
    print(c(DIM, f"  skill     {skill_path or '(none)'}"))
    print(c(DIM, f"  run       {run_dir}"))
    print(c(DIM, f"  approval  {'OFF (--yolo)' if args.yolo else 'on'}"))
    sess.log("start", {"task": task, "model": model, "budget": budget,
                       "source": str(source),
                       "source_supplied": str(args.source),
                       "reference": str(ref_file),
                       "reference_selection": selection or None})

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

    # Paid-for images are never abandoned - but only if there are any. A run that
    # generated nothing ends with an empty output/ and says so.
    shipped, chosen_by, ship_note = ship_best(run_dir, top=max(1, args.top),
                                              judge=not args.no_judge)
    rc = exit_code_for(result, budget, bool(shipped))

    colour = {"done": GRN, "budget_exhausted": GRN,
              "gave_up": YEL, "no_candidates": YEL}.get(result["status"], RED)
    print("\n" + c(BOLD, "─" * 60))
    print(c(colour, result["status"].upper()) +
          c(DIM, f"  ·  {dt/60:.1f} min  ·  {len(sess.candidates())}/{budget} "
                 f"image(s) used  ·  {sess.total_completion} tokens  ·  "
                 f"{sess.compactions} compactions"))
    if result.get("claimed_status"):
        print(c(YEL, f"  the model called finish('{result['claimed_status']}') "
                     f"having generated nothing - recorded as no_candidates."))
    if result.get("summary"):
        print("\n" + textwrap.indent(textwrap.fill(result["summary"], 76), "  "))
    if shipped:
        print()
        tags = {"harness": "   filled by the harness, not the model",
                "judge": "   put here by the judge: the model's pick was "
                         "the wrong colour"}
        for i, (name, who) in enumerate(zip(shipped, chosen_by), 1):
            out = "best.png" if i == 1 else f"best_{i}.png"
            print(c(GRN if i == 1 else DIM, f"  rank {i}    {name:<10} -> "
                                              f"{run_dir / 'output' / out}")
                  + (c(YEL, tags[who]) if who in tags else ""))
        if ship_note:
            print(c(DIM, f"            {ship_note}"))
    else:
        print(c(DIM, "\n  nothing shipped - no candidates were generated."))
    print(c(DIM, f"\n  transcript: {sess.log_path}"))
    if TR.path:
        print(c(DIM, f"  trace:      {TR.path}"))

    sess.log("end", {"result": result, "seconds": dt,
                     "shipped": shipped[0] if shipped else None,
                     "shipped_all": shipped, "shipped_by": chosen_by,
                     "note": ship_note, "exit_code": rc})
    TR.info("harness", f"run finished: {result['status']}",
            body=result.get("summary"), seconds=round(dt, 1),
            images=len(sess.candidates()), budget=budget,
            shipped=", ".join(shipped) or None, exit_code=rc)
    return rc


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
