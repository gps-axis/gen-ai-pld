"""Storage-state validation shared by the DAM authentication commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


LOGIN_URL_MARKERS = ("loginregistration",)


class AuthStateError(RuntimeError):
    """The saved browser state is absent or cannot be used."""


class AuthenticationExpired(RuntimeError):
    """The DAM redirected a browser with saved state back to login."""


def is_login_url(url: str) -> bool:
    """Return whether a DAM URL is the login/registration route."""

    normalized_url = url.casefold()
    return any(marker in normalized_url for marker in LOGIN_URL_MARKERS)


def find_authenticated_dam_url(urls: list[str], dam_url: str) -> str | None:
    """Find an authenticated asset-management page among open browser tabs."""

    expected_host = urlsplit(dam_url).hostname
    for url in reversed(urls):
        parsed_url = urlsplit(url)
        is_asset_page = parsed_url.path == "/asset-management" or (
            parsed_url.path.startswith("/asset-management/")
        )
        if parsed_url.hostname == expected_host and is_asset_page:
            return url
    return None


def is_authenticated_dam_observation(
    *,
    url: str,
    has_asset_management_link: bool,
    dam_url: str,
) -> bool:
    """Recognize the signed-in DAM by route or by its authenticated shell."""

    if find_authenticated_dam_url([url], dam_url) is not None:
        return True

    expected_host = urlsplit(dam_url).hostname
    current_host = urlsplit(url).hostname
    return current_host == expected_host and has_asset_management_link


def require_authenticated_url(url: str) -> None:
    """Raise when the current page URL shows that authentication failed."""

    if is_login_url(url):
        raise AuthenticationExpired(
            "The DAM session expired or was rejected; run the capture command again."
        )


def load_storage_state(path: Path) -> dict[str, Any]:
    """Read a Playwright storage-state file and reject malformed input early."""

    if not path.is_file():
        raise AuthStateError(f"Authentication state does not exist: {path}")

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthStateError(f"Authentication state is not valid JSON: {path}") from exc

    if not isinstance(state, dict):
        raise AuthStateError("Authentication state must be a JSON object.")

    for key in ("cookies", "origins"):
        if not isinstance(state.get(key), list):
            raise AuthStateError(
                f"Authentication state must contain a '{key}' list."
            )

    return state


def secure_storage_state(path: Path) -> None:
    """Restrict a newly written storage-state file to its owner."""

    try:
        path.chmod(0o600)
    except OSError as exc:
        raise AuthStateError(f"Could not secure authentication state: {path}") from exc
