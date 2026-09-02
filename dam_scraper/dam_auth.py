"""Capture and verify a reusable authenticated session for the Gap DAM."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from auth_session import (
    AuthStateError,
    AuthenticationExpired,
    is_login_url,
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
DAM_LOGIN_HOST = "digitalassets.gapinc.com"
IDP_HOST = "onelogon.gap.com"
SSO_GATEWAY_URL = "https://digitalassets.gapinc.com/saml2/login.aspx"
IDP_PATH_SUFFIX = "/resumeSAML20/idp/SSO.ping"
USERNAME_SELECTOR = 'input#username[name="pf.username"][type="text"]'
PASSWORD_SELECTOR = 'input#password[name="pf.pass"][type="password"]'
SIGN_ON_SELECTOR = 'a[title="Sign On"][onclick="postOk();"]'


@dataclass(frozen=True)
class Credentials:
    login_id: str = field(repr=False)
    password: str = field(repr=False)


@dataclass(frozen=True)
class _SsoGateway:
    page: Any
    link: Any


@dataclass(frozen=True)
class _IdpLoginForm:
    username: Any
    password: Any
    submit: Any


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


def _authenticated_page(context: Any, dam_url: str) -> Any | None:
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
            # SSO can replace or close a page while Playwright inspects it.
            continue
    return None


def _find_authenticated_page(
    context: Any,
    dam_url: str,
    timeout_ms: int = AUTH_DETECTION_TIMEOUT_MS,
) -> Any | None:
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        authenticated_page = _authenticated_page(context, dam_url)
        if authenticated_page is not None:
            return authenticated_page

        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def _visible_controls(locator: Any) -> list[Any]:
    return [
        locator.nth(index)
        for index in range(locator.count())
        if locator.nth(index).is_visible()
    ]


def _verified_sso_gateway(page: Any) -> _SsoGateway:
    location = urlsplit(page.url)
    if (
        location.scheme != "https"
        or location.hostname != DAM_LOGIN_HOST
        or not is_login_url(page.url)
    ):
        raise AuthenticationExpired(
            f"Gap SSO reached an unexpected page: {_safe_location(page)}."
        )

    areas = _visible_controls(page.get_by_text("Gap Inc User Login", exact=True))
    links = _visible_controls(
        page.get_by_role("link", name="Login with SSO", exact=True)
    )
    if (
        len(areas) != 1
        or len(links) != 1
        or urljoin(page.url, links[0].get_attribute("href") or "")
        != SSO_GATEWAY_URL
    ):
        raise AuthenticationExpired(
            "The expected Gap DAM SSO gateway was not found."
        )
    return _SsoGateway(page=page, link=links[0])


def _click_sso_gateway(gateway: _SsoGateway, timeout_ms: int) -> None:
    try:
        gateway.link.click()
    except Exception:
        raise AuthenticationExpired("Gap SSO could not be opened.") from None
    try:
        gateway.page.wait_for_url(
            "https://onelogon.gap.com/**",
            timeout=timeout_ms,
            wait_until="domcontentloaded",
        )
    except Exception:
        raise AuthenticationExpired(
            "Gap SSO did not reach the trusted employee login page."
        ) from None


def _is_trusted_idp_origin(page: Any) -> bool:
    try:
        location = urlsplit(page.url)
        return location.scheme == "https" and location.hostname == IDP_HOST
    except Exception:
        return False


def _is_expected_idp_page(page: Any) -> bool:
    if not _is_trusted_idp_origin(page):
        return False
    try:
        path = urlsplit(page.url).path
        return (
            path.startswith("/idp/")
            and path.endswith(IDP_PATH_SUFFIX)
            and page.title() == "Gap Inc Login"
            and len(page.frames) == 1
        )
    except Exception:
        return False


def _verified_idp_form(page: Any) -> _IdpLoginForm:
    if not _is_trusted_idp_origin(page):
        raise AuthenticationExpired(
            f"Gap SSO reached an unexpected page: {_safe_location(page)}."
        )
    if not _is_expected_idp_page(page):
        raise AuthenticationExpired(
            "The expected Gap SSO credential form was not found."
        )

    usernames = _visible_controls(page.locator(USERNAME_SELECTOR))
    passwords = _visible_controls(page.locator(PASSWORD_SELECTOR))
    submits = _visible_controls(page.locator(SIGN_ON_SELECTOR))
    if (
        len(usernames) != 1
        or len(passwords) != 1
        or len(submits) != 1
        or submits[0].inner_text().strip() != "Sign On"
    ):
        raise AuthenticationExpired(
            "The expected Gap SSO credential form was not found."
        )
    return _IdpLoginForm(usernames[0], passwords[0], submits[0])


def _find_idp_page(context: Any, timeout_ms: int) -> Any | None:
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        for page in reversed(context.pages):
            if page.is_closed():
                continue
            try:
                _verified_idp_form(page)
                return page
            except AuthenticationExpired:
                continue
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def _prompt_credentials() -> Credentials:
    if not sys.stdin.isatty():
        raise RuntimeError("Capture requires an interactive terminal.")

    try:
        login_id = input("Gap SSO login ID: ").strip()
        if not login_id:
            raise RuntimeError("The login ID cannot be blank.")
        password = getpass.getpass("Gap SSO password: ")
    except EOFError as exc:
        raise RuntimeError("Capture requires an interactive terminal.") from exc

    if not password.strip():
        raise RuntimeError("The password cannot be blank.")
    return Credentials(login_id=login_id, password=password)


def _login_from_terminal(page: Any) -> None:
    form = _verified_idp_form(page)
    credentials = _prompt_credentials()
    try:
        form.username.fill(credentials.login_id)
    except Exception:
        raise AuthenticationExpired(
            "Gap SSO login ID could not be entered."
        ) from None
    try:
        form.password.fill(credentials.password)
    except Exception:
        raise AuthenticationExpired(
            "Gap SSO password could not be entered."
        ) from None
    try:
        form.submit.click()
    except Exception:
        raise AuthenticationExpired("Gap SSO sign-in could not be submitted.") from None


def _safe_location(page: Any) -> str:
    try:
        location = urlsplit(page.url)
        if not location.scheme or not location.hostname:
            return "unknown page"
        return f"{location.scheme}://{location.hostname}{location.path}"
    except Exception:
        return "unknown page"


def _classify_failed_sso(pages: list[Any]) -> AuthenticationExpired:
    open_pages = [page for page in reversed(pages) if not page.is_closed()]
    for page in open_pages:
        if not _is_trusted_idp_origin(page):
            continue
        try:
            _verified_idp_form(page)
        except AuthenticationExpired:
            return AuthenticationExpired(
                "Gap SSO requires additional verification or an unsupported SSO step."
            )
        return AuthenticationExpired(
            "Gap SSO rejected the credentials or restarted sign-in."
        )

    location = _safe_location(open_pages[0]) if open_pages else "no open page"
    return AuthenticationExpired(f"Gap SSO reached an unexpected page: {location}.")


def _persist_storage_state(context: Any, auth_state: Path) -> None:
    state = context.storage_state(indexed_db=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=auth_state.parent,
            prefix=".dam-auth-",
            suffix=".json",
            delete=False,
        ) as temporary_file:
            json.dump(state, temporary_file)
            temporary_path = Path(temporary_file.name)
        secure_storage_state(temporary_path)
        os.replace(temporary_path, auth_state)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _safe_page_summary(page: Any) -> str:
    try:
        parsed_url = urlsplit(page.url)
        safe_url = urlunsplit((parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", ""))
        title = page.title()
    except Exception:
        return "closed page"
    return f"title={title!r}, url={safe_url!r}"


def capture(args: argparse.Namespace) -> int:
    """Sign in from the terminal and save the authenticated browser state."""

    auth_state = args.auth_state.expanduser().resolve()
    auth_state.parent.mkdir(parents=True, exist_ok=True)

    launch_options: dict[str, Any] = {"headless": not args.headed}
    if args.channel:
        launch_options["channel"] = args.channel

    with _playwright()() as playwright:
        browser = playwright.chromium.launch(**launch_options)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(
                args.url,
                wait_until="domcontentloaded",
                timeout=args.timeout_ms,
            )

            authenticated_page = _authenticated_page(context, DEFAULT_DAM_URL)
            if authenticated_page is None:
                gateway = _verified_sso_gateway(page)
                _click_sso_gateway(gateway, args.timeout_ms)
                idp_page = _find_idp_page(context, args.timeout_ms)
                if idp_page is None:
                    raise _classify_failed_sso(context.pages)
                _login_from_terminal(idp_page)
                try:
                    _verified_idp_form(idp_page)
                except AuthenticationExpired:
                    pass
                else:
                    raise AuthenticationExpired(
                        "Gap SSO rejected the credentials or restarted sign-in."
                    )
                authenticated_page = _find_authenticated_page(
                    context, DEFAULT_DAM_URL, args.timeout_ms
                )
            if authenticated_page is None:
                raise _classify_failed_sso(context.pages)

            _persist_storage_state(context, auth_state)
            captured_page_summary = _safe_page_summary(authenticated_page)
        finally:
            closing_during_exception = sys.exception() is not None
            try:
                browser.close()
            except Exception:
                if not closing_during_exception:
                    raise

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
        "capture", help="Log in from the terminal and save the browser state."
    )
    add_common_options(capture_parser)
    capture_parser.add_argument(
        "--channel",
        help="Installed Chromium channel, such as 'chrome'.",
    )
    capture_parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser for login diagnostics.",
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
