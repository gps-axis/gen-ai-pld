from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from auth_session import (
    AuthStateError,
    AuthenticationExpired,
    find_authenticated_dam_url,
    is_login_url,
    is_authenticated_dam_observation,
    load_storage_state,
    require_authenticated_url,
    secure_storage_state,
)


class AuthenticationUrlTests(unittest.TestCase):
    DAM_URL = "https://digitalassets.gapinc.com/asset-management/270HRGZOLFO1H"

    def test_login_redirect_is_detected_case_insensitively(self) -> None:
        self.assertTrue(
            is_login_url(
                "https://digitalassets.gapinc.com/CS.aspx?VP3=LoginRegistration"
            )
        )

    def test_asset_page_is_authenticated(self) -> None:
        url = "https://digitalassets.gapinc.com/asset-management/270HRGZOLFO1H"
        self.assertFalse(is_login_url(url))
        require_authenticated_url(url)

    def test_login_redirect_raises_expired(self) -> None:
        with self.assertRaises(AuthenticationExpired):
            require_authenticated_url(
                "https://digitalassets.gapinc.com/CS.aspx?VP3=LoginRegistration"
            )

    def test_authenticated_tab_wins_over_original_login_tab(self) -> None:
        login_url = (
            "https://digitalassets.gapinc.com/CS.aspx?VP3=LoginRegistration"
        )

        self.assertEqual(
            find_authenticated_dam_url([login_url, self.DAM_URL], self.DAM_URL),
            self.DAM_URL,
        )

    def test_unrelated_sso_page_is_not_treated_as_authenticated(self) -> None:
        urls = [
            "https://login.microsoftonline.com/example/oauth2/authorize",
            "https://digitalassets.gapinc.com/CS.aspx?VP3=LoginRegistration",
        ]

        self.assertIsNone(find_authenticated_dam_url(urls, self.DAM_URL))

    def test_authenticated_ui_can_live_on_the_legacy_login_route(self) -> None:
        login_url = (
            "https://digitalassets.gapinc.com/CS.aspx?VP3=LoginRegistration"
        )

        self.assertTrue(
            is_authenticated_dam_observation(
                url=login_url,
                has_asset_management_link=True,
                dam_url=self.DAM_URL,
            )
        )


class StorageStateTests(unittest.TestCase):
    def test_valid_storage_state_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "state.json"
            expected = {"cookies": [], "origins": []}
            path.write_text(json.dumps(expected), encoding="utf-8")

            self.assertEqual(load_storage_state(path), expected)

    def test_missing_storage_state_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "missing.json"

            with self.assertRaises(AuthStateError):
                load_storage_state(path)

    def test_malformed_storage_state_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "state.json"
            path.write_text("not-json", encoding="utf-8")

            with self.assertRaises(AuthStateError):
                load_storage_state(path)

    def test_storage_state_requires_playwright_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "state.json"
            path.write_text(json.dumps({"cookies": []}), encoding="utf-8")

            with self.assertRaisesRegex(AuthStateError, "origins"):
                load_storage_state(path)

    def test_storage_state_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "state.json"
            path.write_text('{"cookies": [], "origins": []}', encoding="utf-8")
            path.chmod(0o644)

            secure_storage_state(path)

            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
