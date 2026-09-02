from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dam_auth
from auth_session import AuthenticationExpired


DAM_URL = "https://digitalassets.gapinc.com/asset-management/270HRGZOLFO1H"
DAM_LOGIN_URL = (
    "https://digitalassets.gapinc.com/CS.aspx?VP3=LoginRegistration&secret=query"
)
IDP_URL = (
    "https://onelogon.gap.com/idp/abc/resumeSAML20/idp/SSO.ping?secret=query"
)


class FakeControl:
    def __init__(
        self,
        *,
        visible: bool = True,
        attributes: dict[str, str] | None = None,
        text: str = "",
    ):
        self.visible = visible
        self.attributes = attributes or {}
        self.text = text
        self.filled: list[str] = []
        self.clicks = 0

    def is_visible(self) -> bool:
        return self.visible

    def get_attribute(self, name: str) -> str | None:
        return self.attributes.get(name)

    def inner_text(self) -> str:
        return self.text

    def fill(self, value: str) -> None:
        self.filled.append(value)

    def click(self) -> None:
        self.clicks += 1


class RejectingControl(FakeControl):
    def fill(self, value: str) -> None:
        raise RuntimeError(f"rejected field value {value}")


class FakeLocator:
    def __init__(self, controls: list[FakeControl]) -> None:
        self.controls = controls

    def count(self) -> int:
        return len(self.controls)

    def nth(self, index: int) -> FakeControl:
        return self.controls[index]

    @property
    def first(self) -> FakeControl:
        return self.controls[0]


class FakePage:
    def __init__(
        self,
        url: str,
        *,
        title: str = "",
        roles: dict[tuple[str, str], list[FakeControl]] | None = None,
        selectors: dict[str, list[FakeControl]] | None = None,
        texts: dict[str, list[FakeControl]] | None = None,
        frame_count: int = 1,
        wait_for_url_error: Exception | None = None,
    ) -> None:
        self.url = url
        self.title_value = title
        self.roles = roles or {}
        self.selectors = selectors or {}
        self.texts = texts or {}
        self.frames = [self, *[object() for _ in range(frame_count - 1)]]
        self.goto_calls: list[tuple[str, str, int]] = []
        self.wait_for_url_calls: list[tuple[str, int, str]] = []
        self.wait_for_url_error = wait_for_url_error

    def get_by_role(
        self, role: str, *, name: str, exact: bool = False
    ) -> FakeLocator:
        if not exact:
            raise AssertionError("trusted controls must use exact accessible names")
        return FakeLocator(self.roles.get((role, name), []))

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self.selectors.get(selector, []))

    def get_by_text(self, text: str, *, exact: bool = False) -> FakeLocator:
        if not exact:
            raise AssertionError("trusted text must use an exact match")
        return FakeLocator(self.texts.get(text, []))

    def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_calls.append((url, wait_until, timeout))

    def wait_for_url(self, url: str, *, timeout: int, wait_until: str) -> None:
        self.wait_for_url_calls.append((url, timeout, wait_until))
        if self.wait_for_url_error is not None:
            raise self.wait_for_url_error

    def is_closed(self) -> bool:
        return False

    def title(self) -> str:
        return self.title_value


class FakeContext:
    def __init__(self, page: FakePage, state: dict[str, object] | None = None) -> None:
        self.pages = [page]
        self.page = page
        self.state = state or {"cookies": [], "origins": []}

    def new_page(self) -> FakePage:
        return self.page

    def storage_state(self, *, indexed_db: bool) -> dict[str, object]:
        if not indexed_db:
            raise AssertionError("IndexedDB must be captured")
        return self.state


class FakeBrowser:
    def __init__(self, context: FakeContext) -> None:
        self.context = context
        self.closed = False

    def new_context(self) -> FakeContext:
        return self.context

    def close(self) -> None:
        self.closed = True


class FailingCloseBrowser(FakeBrowser):
    def close(self) -> None:
        self.closed = True
        raise RuntimeError("browser driver connection closed")


class FakeChromium:
    def __init__(self, browser: FakeBrowser) -> None:
        self.browser = browser
        self.launch_calls: list[dict[str, object]] = []

    def launch(self, **options: object) -> FakeBrowser:
        self.launch_calls.append(options)
        return self.browser


class FakePlaywright:
    def __init__(self, browser: FakeBrowser) -> None:
        self.chromium = FakeChromium(browser)


class FakePlaywrightManager:
    def __init__(self, playwright: FakePlaywright) -> None:
        self.playwright = playwright

    def __enter__(self) -> FakePlaywright:
        return self.playwright

    def __exit__(self, *_: object) -> None:
        return None


def fake_playwright(browser: FakeBrowser) -> tuple[object, FakePlaywright]:
    playwright = FakePlaywright(browser)

    def start() -> FakePlaywrightManager:
        return FakePlaywrightManager(playwright)

    return start, playwright


def gateway_page(
    *,
    url: str = DAM_LOGIN_URL,
    links: list[FakeControl] | None = None,
    areas: list[FakeControl] | None = None,
) -> tuple[FakePage, FakeControl, FakeControl]:
    external_login = FakeControl()
    external_password = FakeControl()
    page = FakePage(
        url,
        roles={
            ("link", "Login with SSO"): links
            if links is not None
            else [
                FakeControl(
                    attributes={
                        "href": "https://digitalassets.gapinc.com/saml2/login.aspx"
                    }
                )
            ]
        },
        texts={
            "Gap Inc User Login": areas
            if areas is not None
            else [FakeControl()]
        },
        selectors={
            "external-login": [external_login],
            "external-password": [external_password],
        },
    )
    return page, external_login, external_password


def idp_page(
    *,
    url: str = IDP_URL,
    title: str = "Gap Inc Login",
    username: list[FakeControl] | None = None,
    password: list[FakeControl] | None = None,
    sign_on: list[FakeControl] | None = None,
    frame_count: int = 1,
) -> FakePage:
    return FakePage(
        url,
        title=title,
        selectors={
            'input#username[name="pf.username"][type="text"]': username
            if username is not None
            else [FakeControl()],
            'input#password[name="pf.pass"][type="password"]': password
            if password is not None
            else [FakeControl()],
            'a[title="Sign On"][onclick="postOk();"]': sign_on
            if sign_on is not None
            else [
                FakeControl(
                    attributes={"title": "Sign On", "onclick": "postOk();"},
                    text="Sign On",
                )
            ],
        },
        frame_count=frame_count,
    )


class CaptureParserTests(unittest.TestCase):
    def test_capture_is_headless_by_default_and_accepts_headed(self) -> None:
        parser = dam_auth.build_parser()
        self.assertFalse(parser.parse_args(["capture"]).headed)
        self.assertTrue(parser.parse_args(["capture", "--headed"]).headed)


class SsoGatewayTests(unittest.TestCase):
    def test_gateway_clicks_only_the_exact_sso_link(self) -> None:
        page, external_login, external_password = gateway_page()

        gateway = dam_auth._verified_sso_gateway(page)
        dam_auth._click_sso_gateway(gateway, 1234)

        self.assertEqual(gateway.link.clicks, 1)
        self.assertEqual(
            page.wait_for_url_calls,
            [("https://onelogon.gap.com/**", 1234, "domcontentloaded")],
        )
        self.assertEqual(external_login.filled, [])
        self.assertEqual(external_password.filled, [])

    def test_gateway_navigation_timeout_is_sanitized(self) -> None:
        page, _, _ = gateway_page()
        page.wait_for_url_error = RuntimeError(
            "timeout at https://evil.example/?password=secret-value"
        )
        gateway = dam_auth._verified_sso_gateway(page)

        with self.assertRaises(AuthenticationExpired) as raised:
            dam_auth._click_sso_gateway(gateway, 1234)

        self.assertEqual(
            str(raised.exception),
            "Gap SSO did not reach the trusted employee login page.",
        )

    def test_untrusted_or_ambiguous_gateway_never_prompts(self) -> None:
        valid_link = FakeControl(
            attributes={"href": "https://digitalassets.gapinc.com/saml2/login.aspx"}
        )
        cases = [
            gateway_page(
                url="https://digitalassets.gapinc.com.example/CS.aspx?VP3=LoginRegistration"
            )[0],
            gateway_page(links=[])[0],
            gateway_page(links=[valid_link, valid_link])[0],
            gateway_page(links=[FakeControl(attributes={"href": "/other"})])[0],
            gateway_page(areas=[])[0],
        ]

        for page in cases:
            with self.subTest(url=page.url), patch("builtins.input") as prompt:
                with self.assertRaises(AuthenticationExpired):
                    dam_auth._verified_sso_gateway(page)
                prompt.assert_not_called()


class IdpLoginTests(unittest.TestCase):
    def test_exact_idp_form_accepts_terminal_credentials(self) -> None:
        username = FakeControl()
        password = FakeControl()
        sign_on = FakeControl(
            attributes={"title": "Sign On", "onclick": "postOk();"},
            text="Sign On",
        )
        page = idp_page(username=[username], password=[password], sign_on=[sign_on])

        with patch("dam_auth.sys.stdin.isatty", return_value=True), patch(
            "builtins.input", return_value="person@example.com"
        ), patch("dam_auth.getpass.getpass", return_value="secret-value"):
            dam_auth._login_from_terminal(page)

        self.assertEqual(username.filled, ["person@example.com"])
        self.assertEqual(password.filled, ["secret-value"])
        self.assertEqual(sign_on.clicks, 1)

    def test_wrong_idp_host_or_form_never_prompts(self) -> None:
        cases = [
            idp_page(
                url=IDP_URL.replace("onelogon.gap.com", "onelogon.gap.com.example")
            ),
            idp_page(title="Gap Login"),
            idp_page(username=[]),
            idp_page(frame_count=2),
        ]

        for page in cases:
            with self.subTest(url=page.url), patch("builtins.input") as login, patch(
                "dam_auth.getpass.getpass"
            ) as password:
                with self.assertRaises(AuthenticationExpired):
                    dam_auth._login_from_terminal(page)
                login.assert_not_called()
                password.assert_not_called()

    def test_returned_exact_form_classifies_rejected_credentials(self) -> None:
        error = dam_auth._classify_failed_sso([idp_page()])
        self.assertEqual(
            str(error), "Gap SSO rejected the credentials or restarted sign-in."
        )

    def test_changed_trusted_page_classifies_additional_verification(self) -> None:
        error = dam_auth._classify_failed_sso([idp_page(username=[])])
        self.assertIn("additional verification or an unsupported SSO step", str(error))

    def test_unexpected_page_reports_only_scheme_host_and_path(self) -> None:
        page = FakePage("https://evil.example/path?credential=secret-value")
        error = dam_auth._classify_failed_sso([page])
        self.assertIn("https://evil.example/path", str(error))
        self.assertNotIn("credential", str(error))
        self.assertNotIn("secret-value", str(error))

    def test_credentials_and_submission_errors_are_sanitized(self) -> None:
        credentials = dam_auth.Credentials("person@example.com", "secret-value")
        page = idp_page(username=[RejectingControl()])

        self.assertNotIn("person@example.com", repr(credentials))
        self.assertNotIn("secret-value", repr(credentials))
        with patch("dam_auth.sys.stdin.isatty", return_value=True), patch(
            "builtins.input", return_value="person@example.com"
        ), patch("dam_auth.getpass.getpass", return_value="secret-value"):
            with self.assertRaises(AuthenticationExpired) as raised:
                dam_auth._login_from_terminal(page)
        self.assertNotIn("person@example.com", str(raised.exception))
        self.assertNotIn("secret-value", str(raised.exception))

    def test_blank_or_noninteractive_credentials_are_rejected(self) -> None:
        page = idp_page()

        with patch("dam_auth.sys.stdin.isatty", return_value=False), patch(
            "builtins.input"
        ) as login:
            with self.assertRaisesRegex(RuntimeError, "interactive terminal"):
                dam_auth._login_from_terminal(page)
            login.assert_not_called()

        with patch("dam_auth.sys.stdin.isatty", return_value=True), patch(
            "builtins.input", return_value=""
        ), patch("dam_auth.getpass.getpass") as password:
            with self.assertRaisesRegex(RuntimeError, "cannot be blank"):
                dam_auth._login_from_terminal(page)
            password.assert_not_called()


class CaptureFlowTests(unittest.TestCase):
    def _args(self, auth_state: Path, *, url: str = DAM_URL) -> argparse.Namespace:
        return argparse.Namespace(
            auth_state=auth_state,
            channel=None,
            headed=False,
            timeout_ms=1234,
            url=url,
        )

    def test_success_persists_owner_only_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            auth_state = Path(temporary_directory) / "secrets" / "state.json"
            gateway, _, _ = gateway_page()
            context = FakeContext(
                gateway, {"cookies": [{"name": "session"}], "origins": []}
            )
            browser = FakeBrowser(context)
            start_playwright, playwright = fake_playwright(browser)
            form_page = idp_page()

            def complete_login(_: FakePage) -> None:
                form_page.url = DAM_URL

            with patch("dam_auth._playwright", return_value=start_playwright), patch(
                "dam_auth._click_sso_gateway"
            ), patch("dam_auth._find_idp_page", return_value=form_page), patch(
                "dam_auth._login_from_terminal", side_effect=complete_login
            ), patch("dam_auth._find_authenticated_page", return_value=gateway):
                result = dam_auth.capture(self._args(auth_state))

            self.assertEqual(result, 0)
            self.assertEqual(playwright.chromium.launch_calls, [{"headless": True}])
            self.assertEqual(
                json.loads(auth_state.read_text(encoding="utf-8")), context.state
            )
            self.assertEqual(auth_state.stat().st_mode & 0o777, 0o600)

    def test_failed_login_preserves_existing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            auth_state = Path(temporary_directory) / "state.json"
            original = '{"cookies": [{"name": "old"}], "origins": []}'
            auth_state.write_text(original, encoding="utf-8")
            gateway, _, _ = gateway_page()
            context = FakeContext(gateway)
            browser = FakeBrowser(context)
            start_playwright, _ = fake_playwright(browser)

            with patch("dam_auth._playwright", return_value=start_playwright), patch(
                "dam_auth._click_sso_gateway"
            ), patch("dam_auth._find_idp_page", return_value=idp_page()), patch(
                "dam_auth._login_from_terminal"
            ), patch("dam_auth._find_authenticated_page", return_value=None):
                with self.assertRaises(AuthenticationExpired):
                    dam_auth.capture(self._args(auth_state))

            self.assertEqual(auth_state.read_text(encoding="utf-8"), original)

    def test_returned_idp_form_rejects_without_waiting_for_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            gateway, _, _ = gateway_page()
            form_page = idp_page()
            context = FakeContext(gateway)
            context.pages.append(form_page)
            browser = FakeBrowser(context)
            start_playwright, _ = fake_playwright(browser)

            with patch("dam_auth._playwright", return_value=start_playwright), patch(
                "dam_auth._login_from_terminal"
            ), patch(
                "dam_auth._find_authenticated_page",
                side_effect=AssertionError("waited for timeout"),
            ):
                with self.assertRaisesRegex(
                    AuthenticationExpired, "rejected the credentials"
                ):
                    dam_auth.capture(
                        self._args(Path(temporary_directory) / "state.json")
                    )

    def test_already_authenticated_page_saves_without_prompting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            auth_state = Path(temporary_directory) / "state.json"
            page = FakePage(DAM_URL)
            browser = FakeBrowser(FakeContext(page))
            start_playwright, _ = fake_playwright(browser)

            with patch("dam_auth._playwright", return_value=start_playwright), patch(
                "dam_auth._authenticated_page", return_value=page
            ), patch("builtins.input") as login:
                dam_auth.capture(self._args(auth_state))

            login.assert_not_called()
            self.assertTrue(auth_state.is_file())

    def test_url_option_cannot_redefine_the_trusted_dam_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            auth_state = Path(temporary_directory) / "state.json"
            url = "https://evil.example/asset-management/lookalike"
            page = FakePage(url)
            browser = FakeBrowser(FakeContext(page))
            start_playwright, _ = fake_playwright(browser)

            with patch("dam_auth._playwright", return_value=start_playwright), patch(
                "builtins.input"
            ) as login:
                with self.assertRaises(AuthenticationExpired):
                    dam_auth.capture(self._args(auth_state, url=url))

            login.assert_not_called()
            self.assertFalse(auth_state.exists())

    def test_close_failure_does_not_mask_keyboard_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            gateway, _, _ = gateway_page()
            browser = FailingCloseBrowser(FakeContext(gateway))
            start_playwright, _ = fake_playwright(browser)

            with patch("dam_auth._playwright", return_value=start_playwright), patch(
                "dam_auth._click_sso_gateway", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    dam_auth.capture(
                        self._args(Path(temporary_directory) / "state.json")
                    )

    def test_close_failure_after_success_still_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            page = FakePage(DAM_URL)
            browser = FailingCloseBrowser(FakeContext(page))
            start_playwright, _ = fake_playwright(browser)

            with patch("dam_auth._playwright", return_value=start_playwright), patch(
                "dam_auth._authenticated_page", return_value=page
            ):
                with self.assertRaisesRegex(RuntimeError, "connection closed"):
                    dam_auth.capture(
                        self._args(Path(temporary_directory) / "state.json")
                    )


if __name__ == "__main__":
    unittest.main()
