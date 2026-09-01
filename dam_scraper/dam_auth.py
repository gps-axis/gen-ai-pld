"""Capture and verify a reusable authenticated session for the Gap DAM."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from auth_session import (
    AuthStateError,
    AuthenticationExpired,
    is_authenticated_dam_observation,
    load_storage_state,
    require_authenticated_url,
    secure_storage_state,
)


DEFAULT_DAM_URL = (
    "https://digitalassets.gapinc.com/asset-management/270HRGZOLFO1H"
)
DEFAULT_AUTH_STATE = Path(
    os.environ.get(
        "DAM_AUTH_STATE",
        Path(__file__).resolve().parent / "secrets" / "dam-auth.json",
    )
)
DEFAULT_TIMEOUT_MS = 60_000
AUTH_DETECTION_TIMEOUT_MS = 10_000


def _playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run: uv sync --locked"
        ) from exc
    return sync_playwright


def _has_asset_management_link(page: Any) -> bool:
    for frame in page.frames:
        locator = frame.get_by_role("link", name="Asset management", exact=True)
        if locator.count() > 0 and locator.first.is_visible():
            return True
    return False


def _find_authenticated_page(context: Any, dam_url: str) -> Any | None:
    deadline = time.monotonic() + AUTH_DETECTION_TIMEOUT_MS / 1000
    while True:
        for page in reversed(context.pages):
            if page.is_closed():
                continue
            try:
                if is_authenticated_dam_observation(
                    url=page.url,
                    has_asset_management_link=_has_asset_management_link(page),
                    dam_url=dam_url,
                ):
                    return page
            except Exception:
                # A page can disappear while SSO replaces or closes it.
                continue

        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def _safe_page_summary(page: Any) -> str:
    try:
        parsed_url = urlsplit(page.url)
        safe_url = urlunsplit((parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", ""))
        title = page.title()
    except Exception:
        return "closed page"
    return f"title={title!r}, url={safe_url!r}"


def capture(args: argparse.Namespace) -> int:
    """Open a visible browser, wait for manual login, and save its state."""

    auth_state = args.auth_state.expanduser().resolve()
    auth_state.parent.mkdir(parents=True, exist_ok=True)

    launch_options: dict[str, Any] = {"headless": False}
    if args.channel:
        launch_options["channel"] = args.channel

    with _playwright()() as playwright:
        browser = playwright.chromium.launch(**launch_options)
        context = browser.new_context()
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)

        print("Complete the DAM login in the browser window.")
        try:
            input("After the Gap asset page loads, press Enter here: ")
        except EOFError as exc:
            browser.close()
            raise RuntimeError("Capture requires an interactive terminal.") from exc

        authenticated_page = _find_authenticated_page(context, args.url)
        if authenticated_page is None:
            page_summaries = "; ".join(
                _safe_page_summary(open_page) for open_page in context.pages
            )
            raise AuthenticationExpired(
                "No authenticated Gap DAM asset page was open after login. "
                f"Observed pages: {page_summaries or 'none'}."
            )
        context.storage_state(path=auth_state, indexed_db=True)
        secure_storage_state(auth_state)
        captured_page_summary = _safe_page_summary(authenticated_page)
        browser.close()

    print(
        "Saved authenticated browser state from "
        f"{captured_page_summary} to {auth_state}"
    )
    return 0


def check(args: argparse.Namespace) -> int:
    """Load saved state and prove that it reaches the authenticated DAM page."""

    auth_state = args.auth_state.expanduser().resolve()
    load_storage_state(auth_state)

    with _playwright()() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(storage_state=auth_state)
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        authenticated_page = _find_authenticated_page(context, args.url)
        if authenticated_page is None:
            require_authenticated_url(page.url)
            raise AuthenticationExpired(
                "The saved session did not reach a Gap DAM asset page; "
                "run the capture command again."
            )
        current_page_summary = _safe_page_summary(authenticated_page)
        browser.close()

    print(f"Authentication is valid: {current_page_summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture or verify a reusable Gap DAM browser session."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--url", default=DEFAULT_DAM_URL)
        command.add_argument(
            "--auth-state",
            type=Path,
            default=DEFAULT_AUTH_STATE,
            help="Storage-state path; defaults to DAM_AUTH_STATE.",
        )
        command.add_argument(
            "--timeout-ms",
            type=int,
            default=DEFAULT_TIMEOUT_MS,
        )

    capture_parser = subparsers.add_parser(
        "capture", help="Log in manually once and save the browser state."
    )
    add_common_options(capture_parser)
    capture_parser.add_argument(
        "--channel",
        help="Installed Chromium channel, such as 'chrome'.",
    )
    capture_parser.set_defaults(handler=capture)

    check_parser = subparsers.add_parser(
        "check", help="Verify that saved state still reaches the DAM."
    )
    add_common_options(check_parser)
    check_parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser while checking the session.",
    )
    check_parser.set_defaults(handler=check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except AuthenticationExpired as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr)
        return 3
    except (AuthStateError, RuntimeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
