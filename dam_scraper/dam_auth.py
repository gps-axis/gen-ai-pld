"""Capture and verify a reusable authenticated session for the Gap DAM."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
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

# Where the sign-in comes from when a login is needed, in this order: the
# environment (a value, or the path of a file holding it - the Docker and
# Compose secrets convention, where only the path travels in the environment
# and the secret sits read-only under /run/secrets), the macOS Keychain entry
# that `store` writes, then the terminal. Nothing here ever goes into a file
# in the repository.
LOGIN_ID_ENV = "DAM_LOGIN_ID"
PASSWORD_ENV = "DAM_PASSWORD"
LOGIN_ID_FILE_ENV = "DAM_LOGIN_ID_FILE"
PASSWORD_FILE_ENV = "DAM_PASSWORD_FILE"
KEYCHAIN_SERVICE = "gap-dam-sso"
KEYCHAIN_LABEL = "Gap DAM sign-in (PLD harness)"
SECURITY_TOOL = "/usr/bin/security"
STORE_COMMAND = "cd dam_scraper && uv run --locked python dam_auth.py store"
# `security find-generic-password -g` prints the account among the item's
# attributes on stdout, and the password on stderr - quoted when it is plain
# printable ASCII, otherwise as 0x-prefixed hex (a backslash, a tab or any
# non-ASCII character is enough to switch it). The two forms cannot be told
# apart from the bare `-w` output, which is why -g is the one parsed here.
_KEYCHAIN_ACCOUNT = re.compile(r'^\s*"acct"<blob>="(.*)"$', re.MULTILINE)
_KEYCHAIN_HEX_PASSWORD = re.compile(r"^password: 0x([0-9A-Fa-f]*)", re.MULTILINE)
_KEYCHAIN_TEXT_PASSWORD = re.compile(r'^password: "(.*)"$', re.MULTILINE)


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


def _setting_from_environment(name: str, file_name: str) -> tuple[str, str]:
    """(value, which variable supplied it): NAME's value, else the contents of
    the file NAME_FILE names, else ('', ''). One trailing line break is dropped
    from the file, the way `echo` and most secret stores leave one."""

    value = os.environ.get(name, "")
    path = os.environ.get(file_name, "")
    if value and path:
        raise RuntimeError(f"Both {name} and {file_name} are set; set one of them.")
    if value:
        return value, name
    if not path:
        return "", ""
    try:
        contents = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"{file_name} names a file that could not be read: {path} ({exc.strerror})."
        ) from exc
    return contents.rstrip("\r\n"), file_name


def _credentials_from_environment() -> tuple[Credentials, str] | None:
    """The login from the environment and where it came from, or None when
    nothing is set there."""

    login_id, login_source = _setting_from_environment(LOGIN_ID_ENV, LOGIN_ID_FILE_ENV)
    password, password_source = _setting_from_environment(PASSWORD_ENV, PASSWORD_FILE_ENV)
    login_id = login_id.strip()
    if not login_id and not password.strip():
        return None
    if not login_id:
        raise RuntimeError(
            f"{password_source} is set but {LOGIN_ID_ENV} is not "
            f"(nor {LOGIN_ID_FILE_ENV}); set both or neither."
        )
    if not password.strip():
        raise RuntimeError(
            f"{login_source} is set but {PASSWORD_ENV} is not "
            f"(nor {PASSWORD_FILE_ENV}); set both or neither."
        )
    return Credentials(login_id=login_id, password=password), f"{login_source} and {password_source}"


def _keychain_available() -> bool:
    return sys.platform == "darwin" and os.access(SECURITY_TOOL, os.X_OK)


def _security(*arguments: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [SECURITY_TOOL, *arguments],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _credentials_from_keychain() -> Credentials | None:
    """The login `store` put in the macOS Keychain, or None when there is none.

    A Keychain that cannot be read at all (locked, or the tool failing) is
    reported on stderr and treated as empty, so the terminal prompt still
    works; an entry that exists but is unusable is an error, because the fix
    is different and the prompt would only hide it.
    """

    if not _keychain_available():
        return None
    result = _security("find-generic-password", "-s", KEYCHAIN_SERVICE, "-g")
    if result.returncode == 44:  # errSecItemNotFound
        return None
    if result.returncode != 0:
        print(
            "The macOS Keychain could not be read "
            f"(security exit {result.returncode}): {result.stderr.strip()}",
            file=sys.stderr,
        )
        return None

    account = _KEYCHAIN_ACCOUNT.search(result.stdout)
    login_id = account.group(1).strip() if account else ""
    hex_password = _KEYCHAIN_HEX_PASSWORD.search(result.stderr)
    text_password = _KEYCHAIN_TEXT_PASSWORD.search(result.stderr)
    password = ""
    if hex_password is not None:
        try:
            password = bytes.fromhex(hex_password.group(1)).decode("utf-8")
        except ValueError:
            password = ""
    elif text_password is not None:
        password = text_password.group(1)
    if not login_id or not password.strip():
        raise RuntimeError(
            f"The Keychain entry '{KEYCHAIN_SERVICE}' is missing its login ID or "
            f"password; store it again: {STORE_COMMAND}"
        )
    return Credentials(login_id=login_id, password=password)


def _keychain_quote(value: str) -> str:
    """One argument for a line of `security -i` input.

    Interactive mode splits the line itself: double quotes group, backslash
    escapes, and a newline ends the command. A newline cannot come from the
    terminal prompts, so it is refused rather than escaped.
    """

    if "\n" in value or "\r" in value:
        raise RuntimeError("The login ID and password cannot contain a line break.")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _keychain_store(credentials: Credentials) -> None:
    """Write the login to the Keychain and read it back to prove it landed.

    The item goes in through `security -i` on stdin rather than as an
    argument, so the password never shows in the process list. Interactive
    mode does not report a failed command in its exit status, hence the
    read-back.
    """

    line = " ".join(
        [
            "add-generic-password",
            "-U",
            "-s", _keychain_quote(KEYCHAIN_SERVICE),
            "-a", _keychain_quote(credentials.login_id),
            "-l", _keychain_quote(KEYCHAIN_LABEL),
            "-T", SECURITY_TOOL,
            "-w", _keychain_quote(credentials.password),
        ]
    )
    result = _security("-i", stdin=line + "\n")
    if result.returncode != 0:
        raise RuntimeError(
            "The macOS Keychain refused the login "
            f"(security exit {result.returncode}): {result.stderr.strip()}"
        )
    try:
        stored = _credentials_from_keychain()
    except RuntimeError:
        stored = None
    if stored != credentials:
        raise RuntimeError(
            "The macOS Keychain did not keep the login as entered; "
            f"nothing usable is stored under '{KEYCHAIN_SERVICE}'."
        )


def stored_credentials() -> tuple[Credentials, str] | None:
    """The login that needs no terminal, and where it came from: the
    environment, else the macOS Keychain, else None. What the scraper signs
    in with on its own - it never asks."""

    from_environment = _credentials_from_environment()
    if from_environment is not None:
        return from_environment
    from_keychain = _credentials_from_keychain()
    if from_keychain is not None:
        return from_keychain, f"the macOS Keychain ('{KEYCHAIN_SERVICE}')"
    return None


def _resolve_credentials() -> Credentials:
    """A stored login, else the terminal - and say which."""

    stored = stored_credentials()
    if stored is not None:
        credentials, source = stored
        print(f"Using the Gap SSO login from {source}.")
        return credentials
    return _prompt_credentials()


def _prompt_credentials() -> Credentials:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "No stored Gap SSO login and no interactive terminal to ask from. "
            f"Store one with: {STORE_COMMAND} "
            f"(or set {LOGIN_ID_ENV} and {PASSWORD_ENV})."
        )

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


def _sign_in(page: Any, credentials: Credentials | None = None) -> None:
    form = _verified_idp_form(page)
    if credentials is None:
        credentials = _resolve_credentials()
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


def store(args: argparse.Namespace) -> int:
    """Ask once for the Gap SSO login and keep it in the macOS Keychain."""

    if not _keychain_available():
        raise RuntimeError(
            f"Storing the login needs the macOS Keychain ({SECURITY_TOOL}). "
            f"Elsewhere, set {LOGIN_ID_ENV} and {PASSWORD_ENV} instead."
        )
    if not sys.stdin.isatty():
        raise RuntimeError("store needs an interactive terminal to ask for the login.")
    _keychain_store(_prompt_credentials())
    print(
        f"Saved the Gap SSO login to the macOS Keychain as '{KEYCHAIN_SERVICE}'. "
        "capture uses it from now on. Run store again after a password change, "
        "or forget to remove it."
    )
    return 0


def forget(args: argparse.Namespace) -> int:
    """Remove the login that `store` put in the macOS Keychain."""

    if not _keychain_available():
        raise RuntimeError(
            f"There is no macOS Keychain here ({SECURITY_TOOL}); nothing to forget."
        )
    result = _security("delete-generic-password", "-s", KEYCHAIN_SERVICE)
    if result.returncode == 44:  # errSecItemNotFound
        print("No stored Gap SSO login to remove.")
        return 0
    if result.returncode != 0:
        raise RuntimeError(
            "The macOS Keychain entry could not be removed "
            f"(security exit {result.returncode}): {result.stderr.strip()}"
        )
    print(f"Removed the Gap SSO login '{KEYCHAIN_SERVICE}' from the macOS Keychain.")
    return 0


def sign_in(
    auth_state: Path,
    *,
    credentials: Credentials | None = None,
    url: str = DEFAULT_DAM_URL,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    headed: bool = False,
    channel: str | None = None,
) -> str:
    """Log in to the DAM, save the browser state at auth_state, and return a
    summary of the page that proved the login.

    With credentials=None the login is resolved only once the SSO form is on
    screen - environment, Keychain, then the terminal - so a browser that
    turns out to be signed in already never asks for anything. The scraper
    passes the login it found itself, so that it never reaches a prompt.
    """

    auth_state = auth_state.expanduser().resolve()
    auth_state.parent.mkdir(parents=True, exist_ok=True)

    launch_options: dict[str, Any] = {"headless": not headed}
    if channel:
        launch_options["channel"] = channel

    with _playwright()() as playwright:
        browser = playwright.chromium.launch(**launch_options)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            authenticated_page = _authenticated_page(context, DEFAULT_DAM_URL)
            if authenticated_page is None:
                gateway = _verified_sso_gateway(page)
                _click_sso_gateway(gateway, timeout_ms)
                idp_page = _find_idp_page(context, timeout_ms)
                if idp_page is None:
                    raise _classify_failed_sso(context.pages)
                _sign_in(idp_page, credentials)
                try:
                    _verified_idp_form(idp_page)
                except AuthenticationExpired:
                    pass
                else:
                    raise AuthenticationExpired(
                        "Gap SSO rejected the credentials or restarted sign-in."
                    )
                authenticated_page = _find_authenticated_page(
                    context, DEFAULT_DAM_URL, timeout_ms
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

    return captured_page_summary


def capture(args: argparse.Namespace) -> int:
    """Sign in and save the authenticated browser state - see sign_in."""

    auth_state = args.auth_state.expanduser().resolve()
    captured_page_summary = sign_in(
        auth_state,
        url=args.url,
        timeout_ms=args.timeout_ms,
        headed=args.headed,
        channel=args.channel,
    )
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
        "capture",
        help=(
            "Log in and save the browser state. The login comes from "
            f"{LOGIN_ID_ENV}/{PASSWORD_ENV} (or the files {LOGIN_ID_FILE_ENV}/"
            f"{PASSWORD_FILE_ENV} name), else the macOS Keychain entry written "
            "by 'store', else the terminal."
        ),
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

    store_parser = subparsers.add_parser(
        "store",
        help=(
            "Ask once for the Gap SSO login and keep it in the macOS Keychain "
            f"(item '{KEYCHAIN_SERVICE}'), so capture never prompts again."
        ),
    )
    store_parser.set_defaults(handler=store)

    forget_parser = subparsers.add_parser(
        "forget", help="Remove the login that 'store' put in the macOS Keychain."
    )
    forget_parser.set_defaults(handler=forget)

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
