from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import dam_auth
from auth_session import AuthStateError, AuthenticationExpired
from playwright.sync_api import Error as PlaywrightError
from dam_scrape import (
    ASSET_PRODUCTION_TYPE,
    EXIT_NOTHING_TO_DOWNLOAD,
    FINAL_ASSET_VALUE,
    FacetOption,
    FacetUnavailableError,
    ITEM_DETAILS_LIMIT,
    MAX_PER_CODE,
    NoFinalImageError,
    NoLaydownAssetsError,
    REQUIRED_FILTERS,
    ScrapeError,
    SessionRejectedError,
    SHOT_REQUEST_ID,
    ShotBatch,
    apply_exclusive_facet,
    batch_limit,
    choose_shot_batches,
    dismiss_popups,
    extract_archive,
    find_facet_containers,
    inspect_jpg_archive,
    is_canceled_download,
    is_complete_manifest_reusable,
    main,
    normalize_style_number,
    open_search_page,
    parse_selected_asset_count,
    parse_total_result_count,
    read_result_total,
    safe_query_directory,
    select_asset_limit,
    toggle_facet_checkbox,
    wrap_bare_jpg_as_archive,
    write_json_atomic,
)


class SearchResultTests(unittest.TestCase):
    def test_parses_total_from_dam_pagination(self) -> None:
        self.assertEqual(parse_total_result_count("1 - 50 of 14507"), 14507)

    def test_missing_pagination_returns_none(self) -> None:
        self.assertIsNone(parse_total_result_count("No results yet"))

    def test_parses_selected_asset_count(self) -> None:
        self.assertEqual(parse_selected_asset_count("IMAGES 2 assets"), 2)

    def test_query_directory_is_safe(self) -> None:
        self.assertEqual(safe_query_directory(" 73/8569 "), "73-8569")

    def test_nine_digit_style_uses_first_six_digits(self) -> None:
        self.assertEqual(normalize_style_number("853417012"), "853417")

    def test_six_digit_style_is_unchanged(self) -> None:
        self.assertEqual(normalize_style_number("738569"), "738569")

    def test_malformed_style_is_rejected(self) -> None:
        for value in ("85341", "85341701", "8534170123", "85341A012"):
            with self.subTest(value=value), self.assertRaises(ScrapeError):
                normalize_style_number(value)

    def test_empty_query_directory_is_rejected(self) -> None:
        with self.assertRaises(ScrapeError):
            safe_query_directory(" / ")


class ShotSelectionPolicyTests(unittest.TestCase):
    def test_cap_is_ten_per_code(self) -> None:
        self.assertEqual(MAX_PER_CODE, 10)

    def test_p01_takes_priority_and_is_capped(self) -> None:
        self.assertEqual(
            choose_shot_batches({"AV5": 12, "P01": 15, "P02": 4}),
            (ShotBatch("P01", 10),),
        )

    def test_p01_uses_every_available_asset_below_the_cap(self) -> None:
        self.assertEqual(
            choose_shot_batches({"AV5": 8, "P01": 2}),
            (ShotBatch("P01", 2),),
        )

    def test_av5_is_used_only_when_p01_is_absent(self) -> None:
        self.assertEqual(
            choose_shot_batches({"AV2": 9, "AV5": 1}),
            (ShotBatch("AV5", 1),),
        )

    def test_every_available_code_is_used_when_preferred_codes_are_absent(self) -> None:
        self.assertEqual(
            choose_shot_batches({"AV2": 12, "P02": 2, "AV1": 0}),
            (ShotBatch("AV2", 10), ShotBatch("P02", 2)),
        )

    def test_no_available_shots_is_rejected(self) -> None:
        with self.assertRaisesRegex(ScrapeError, "Shot Request ID"):
            choose_shot_batches({})


class FacetSelectionTests(unittest.TestCase):
    def test_facet_heading_is_found_by_visible_text_without_original_title(self) -> None:
        sidebar = _VirtualFacetSidebar(heading_title=ASSET_PRODUCTION_TYPE)
        page = _FacetPage()
        with patch("dam_scrape.find_filter_sidebar", return_value=sidebar):
            containers = find_facet_containers(
                page, ASSET_PRODUCTION_TYPE, timeout_ms=100
            )

        self.assertIs(containers, sidebar.containers)

    def test_virtualized_sidebar_scrolls_until_later_facet_is_rendered(self) -> None:
        sidebar = _VirtualFacetSidebar(
            heading_title=SHOT_REQUEST_ID,
            heading_scroll_top=300,
        )
        page = _FacetPage()
        with patch("dam_scrape.find_filter_sidebar", return_value=sidebar):
            containers = find_facet_containers(page, SHOT_REQUEST_ID, timeout_ms=100)

        self.assertIs(containers, sidebar.containers)
        self.assertGreater(max(sidebar.scroll_positions), 0)

    def test_search_page_applies_final_before_it_is_returned(self) -> None:
        page = _SearchPage()
        context = _SearchContext(page)
        facet_events: list[tuple[str, str, str | None]] = []
        with (
            patch("dam_scrape._find_authenticated_page", return_value=page),
            patch("dam_scrape.find_visible", return_value=page.search),
            patch(
                "dam_scrape.wait_for_post",
                side_effect=lambda _, action, __: action(),
            ),
            patch(
                "dam_scrape.clear_facet_filters",
                side_effect=lambda _, title, __: facet_events.append(
                    ("clear", title, None)
                ),
            ),
            patch("dam_scrape.read_result_total", return_value=12),
            patch(
                "dam_scrape.apply_exclusive_facet",
                side_effect=lambda _, title, value, __: facet_events.append(
                    ("apply", title, value)
                ),
            ),
        ):
            self.assertIs(
                open_search_page(context, "https://dam.test", "738569", 100),
                page,
            )

        self.assertEqual(
            facet_events,
            [
                ("clear", SHOT_REQUEST_ID, None),
                ("apply", ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE),
            ],
        )

    def test_final_is_selected_after_other_production_types_are_cleared(self) -> None:
        options = (
            FacetOption("WORKING", 2, True),
            FacetOption(FINAL_ASSET_VALUE, 4, False),
        )
        settled = (
            FacetOption("WORKING", 2, False),
            FacetOption(FINAL_ASSET_VALUE, 4, True),
        )
        with (
            patch(
                "dam_scrape.read_facet_options",
                side_effect=(options, options, settled),
            ),
            patch("dam_scrape.toggle_facet_checkbox") as toggle,
        ):
            apply_exclusive_facet(
                object(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 100
            )

        self.assertEqual(
            [call.args[2] for call in toggle.call_args_list],
            ["WORKING", FINAL_ASSET_VALUE],
        )

    def test_final_selection_is_idempotent(self) -> None:
        options = (FacetOption(FINAL_ASSET_VALUE, 4, True),)
        with (
            patch("dam_scrape.read_facet_options", return_value=options),
            patch("dam_scrape.toggle_facet_checkbox") as toggle,
        ):
            apply_exclusive_facet(
                object(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 100
            )

        toggle.assert_not_called()

    def test_missing_final_has_specific_user_facing_error(self) -> None:
        options = (FacetOption("WORKING", 2, False),)
        with patch("dam_scrape.read_facet_options", return_value=options):
            with self.assertRaisesRegex(
                ScrapeError, r"^No FINAL image is available for this style\.$"
            ):
                apply_exclusive_facet(
                    object(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 100
                )

    def test_missing_production_type_facet_means_no_final_image(self) -> None:
        missing_facet = FacetUnavailableError(
            "The Asset Production Type filter was not available."
        )
        with patch("dam_scrape.read_facet_options", side_effect=missing_facet):
            with self.assertRaisesRegex(
                ScrapeError, r"^No FINAL image is available for this style\.$"
            ):
                apply_exclusive_facet(
                    object(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 100
                )

    def test_unreadable_production_type_facet_remains_operational_error(self) -> None:
        unreadable_facet = ScrapeError(
            "The Asset Production Type filter could not be read."
        )
        with patch("dam_scrape.read_facet_options", side_effect=unreadable_facet):
            with self.assertRaisesRegex(
                ScrapeError,
                r"^The Asset Production Type filter could not be read\.$",
            ):
                apply_exclusive_facet(
                    object(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 100
                )

    def test_zero_count_final_has_specific_user_facing_error(self) -> None:
        options = (FacetOption(FINAL_ASSET_VALUE, 0, False),)
        with patch("dam_scrape.read_facet_options", return_value=options):
            with self.assertRaisesRegex(
                ScrapeError, r"^No FINAL image is available for this style\.$"
            ):
                apply_exclusive_facet(
                    object(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 100
                )

    def test_missing_final_is_a_miss_not_a_failure(self) -> None:
        # Laydown shots with none tagged FINAL are as much "nothing to
        # download" as an empty search: the same fallback, the same exit.
        for read_facet_options in (
            {"return_value": (FacetOption("WORKING", 2, False),)},
            {"return_value": (FacetOption(FINAL_ASSET_VALUE, 0, False),)},
            {"side_effect": FacetUnavailableError("The Asset Production Type filter was not available.")},
        ):
            with patch("dam_scrape.read_facet_options", **read_facet_options):
                with self.assertRaises(NoLaydownAssetsError):
                    apply_exclusive_facet(
                        object(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 100
                    )

    def test_final_read_straight_after_the_click_may_lag_the_sidebar(self) -> None:
        # The first read after the click still shows the old state; the next one
        # has caught up. One click, no retry.
        unchecked = (FacetOption(FINAL_ASSET_VALUE, 4, False),)
        checked = (FacetOption(FINAL_ASSET_VALUE, 4, True),)
        with (
            patch(
                "dam_scrape.read_facet_options",
                side_effect=(unchecked, unchecked, unchecked, (), checked),
            ),
            patch("dam_scrape.toggle_facet_checkbox") as toggle,
        ):
            settled = apply_exclusive_facet(
                _SettlePage(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 10_000
            )
        self.assertEqual(settled, checked)
        self.assertEqual(
            [call.args[2] for call in toggle.call_args_list], [FINAL_ASSET_VALUE]
        )

    def test_final_click_the_dam_dropped_is_clicked_again(self) -> None:
        unchecked = (FacetOption(FINAL_ASSET_VALUE, 4, False),)
        checked = (FacetOption(FINAL_ASSET_VALUE, 4, True),)
        with (
            patch("dam_scrape.FACET_SETTLE_MS", 0),
            patch(
                "dam_scrape.read_facet_options",
                # initial, pre-click, settle (still unchecked: click lost),
                # pre-click again, settle (checked)
                side_effect=(unchecked, unchecked, unchecked, unchecked, checked),
            ),
            patch("dam_scrape.toggle_facet_checkbox") as toggle,
        ):
            settled = apply_exclusive_facet(
                _SettlePage(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 10_000
            )
        self.assertEqual(settled, checked)
        self.assertEqual(
            [call.args[2] for call in toggle.call_args_list],
            [FINAL_ASSET_VALUE, FINAL_ASSET_VALUE],
        )

    def test_final_that_never_sticks_fails_after_the_last_attempt(self) -> None:
        unchecked = (FacetOption(FINAL_ASSET_VALUE, 4, False),)
        with (
            patch("dam_scrape.FACET_SETTLE_MS", 0),
            patch("dam_scrape.read_facet_options", return_value=unchecked),
            patch("dam_scrape.toggle_facet_checkbox") as toggle,
        ):
            with self.assertRaisesRegex(
                ScrapeError, r"^Asset Production Type FINAL did not update\.$"
            ):
                apply_exclusive_facet(
                    _SettlePage(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 10_000
                )
        self.assertEqual(toggle.call_count, 3)

    def test_checkbox_lookup_is_scoped_to_the_named_facet(self) -> None:
        containers = _FacetContainers()
        page = _FacetPage()
        selected = (FacetOption(FINAL_ASSET_VALUE, 4, True),)
        with (
            patch("dam_scrape.find_facet_containers", return_value=containers),
            patch("dam_scrape.read_facet_options", return_value=selected),
            patch(
                "dam_scrape.wait_for_post",
                side_effect=lambda _, action, __: action(),
            ),
        ):
            toggle_facet_checkbox(
                page, ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 100
            )

        self.assertEqual(
            containers.selector,
            'input[type=\'checkbox\'][aria-label="FINAL"]',
        )
        self.assertEqual(containers.checkbox.clicks, 1)

    def test_facet_click_the_dam_dropped_is_clicked_again(self) -> None:
        # The settle window closes with the checkbox still as it was: the DAM
        # dropped the click. The second one takes.
        containers = _FacetContainers()
        unchecked = (FacetOption(FINAL_ASSET_VALUE, 4, False),)
        checked = (FacetOption(FINAL_ASSET_VALUE, 4, True),)
        with (
            patch("dam_scrape.FACET_SETTLE_MS", 0),
            patch("dam_scrape.find_facet_containers", return_value=containers),
            patch("dam_scrape.read_facet_options", side_effect=(unchecked, checked)),
            patch(
                "dam_scrape.wait_for_post",
                side_effect=lambda _, action, __: action(),
            ),
        ):
            toggle_facet_checkbox(
                _FacetPage(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 10_000
            )

        self.assertEqual(containers.checkbox.clicks, 2)

    def test_facet_click_that_never_sticks_fails_after_the_last_attempt(self) -> None:
        containers = _FacetContainers()
        unchecked = (FacetOption(FINAL_ASSET_VALUE, 4, False),)
        with (
            patch("dam_scrape.FACET_SETTLE_MS", 0),
            patch("dam_scrape.find_facet_containers", return_value=containers),
            patch("dam_scrape.read_facet_options", return_value=unchecked),
            patch(
                "dam_scrape.wait_for_post",
                side_effect=lambda _, action, __: action(),
            ),
        ):
            with self.assertRaisesRegex(
                ScrapeError, r"^Asset Production Type FINAL did not update\.$"
            ):
                toggle_facet_checkbox(
                    _FacetPage(), ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE, 10_000
                )

        self.assertEqual(containers.checkbox.clicks, 3)

    def test_value_the_facet_drops_while_unchecking_counts_as_cleared(self) -> None:
        # The leftover P01 is clicked off; the settle read still shows it
        # checked, and by the retry the redrawn facet no longer lists P01 at
        # all. Off the filter is what the uncheck was for: no error, no click.
        listed = _FacetContainers(checked=True)
        gone = _FacetContainers(checked=True, listed=False)
        still_checked = (FacetOption("P01", 22, True),)
        with (
            patch("dam_scrape.FACET_SETTLE_MS", 0),
            patch("dam_scrape.find_facet_containers", side_effect=(listed, gone)),
            patch("dam_scrape.read_facet_options", return_value=still_checked),
            patch(
                "dam_scrape.wait_for_post",
                side_effect=lambda _, action, __: action(),
            ),
        ):
            toggle_facet_checkbox(_FacetPage(), SHOT_REQUEST_ID, "P01", 10_000)

        self.assertEqual(listed.checkbox.clicks, 1)
        self.assertEqual(gone.checkbox.clicks, 0)

    def test_value_missing_at_the_first_look_cannot_be_toggled(self) -> None:
        gone = _FacetContainers(listed=False)
        with patch("dam_scrape.find_facet_containers", return_value=gone):
            with self.assertRaisesRegex(
                ScrapeError, r"^The Shot Request ID value P01 could not be read\.$"
            ):
                toggle_facet_checkbox(_FacetPage(), SHOT_REQUEST_ID, "P01", 10_000)


class _SettlePage:
    def wait_for_timeout(self, _: int) -> None:
        pass


class _SearchControl:
    def __init__(self) -> None:
        self.value = ""

    def fill(self, value: str) -> None:
        self.value = value

    def press(self, key: str) -> None:
        if key != "Enter":
            raise AssertionError(key)


class _SearchPage:
    def __init__(self) -> None:
        self.search = _SearchControl()

    def goto(self, *_: object, **__: object) -> None:
        pass

    def wait_for_timeout(self, _: int) -> None:
        pass


class _SearchContext:
    def __init__(self, page: _SearchPage) -> None:
        self.page = page

    def new_page(self) -> _SearchPage:
        return self.page


class _FacetCheckbox:
    def __init__(self, checked: bool = False) -> None:
        self.clicks = 0
        self.checked = checked

    def is_checked(self) -> bool:
        return self.checked

    def evaluate(self, _: str) -> None:
        self.clicks += 1


class _FacetCheckboxCandidates:
    def __init__(self, checkbox: _FacetCheckbox, listed: bool) -> None:
        self.first = checkbox
        self.listed = listed

    def count(self) -> int:
        return 1 if self.listed else 0


class _FacetContainers:
    """A facet's containers holding one checkbox - or, with `listed` False,
    a redrawn facet that no longer lists the value at all."""

    def __init__(self, checked: bool = False, listed: bool = True) -> None:
        self.checkbox = _FacetCheckbox(checked)
        self.listed = listed
        self.selector = ""

    def locator(self, selector: str) -> _FacetCheckboxCandidates:
        self.selector = selector
        return _FacetCheckboxCandidates(self.checkbox, self.listed)


class _FacetPage:
    def wait_for_timeout(self, _: int) -> None:
        pass


class _VirtualLocatorList:
    def __init__(self, items: list[object]) -> None:
        self.items = items

    @property
    def first(self) -> object:
        return self.items[0]

    def count(self) -> int:
        return len(self.items)

    def nth(self, index: int) -> object:
        return self.items[index]


class _VirtualFacetHeader:
    def get_attribute(self, name: str) -> str | None:
        return "facet:HeaderPnl" if name == "id" else None


class _VirtualFacetHeading:
    def __init__(self, title: str) -> None:
        self.title = title

    def inner_text(self) -> str:
        return f"{self.title}\N{NO-BREAK SPACE}"

    def locator(self, _: str) -> _VirtualFacetHeader:
        return _VirtualFacetHeader()


class _VirtualFacetContainers:
    def count(self) -> int:
        return 1


class _VirtualFacetSidebar:
    def __init__(self, heading_title: str, heading_scroll_top: int = 0) -> None:
        self.heading = _VirtualFacetHeading(heading_title)
        self.heading_scroll_top = heading_scroll_top
        self.scroll_top = 0
        self.scroll_positions: list[int] = []
        self.containers = _VirtualFacetContainers()

    def evaluate(self, script: str, value: int | None = None) -> object:
        if "scrollHeight" in script and "clientHeight" in script:
            return {"clientHeight": 200, "scrollHeight": 600}
        if value is not None:
            self.scroll_top = value
            self.scroll_positions.append(value)
        return None

    def locator(self, selector: str) -> object:
        if selector == "[id$=':FacetNameLbl_Lbl']":
            items = (
                [self.heading]
                if self.scroll_top >= self.heading_scroll_top
                else []
            )
            return _VirtualLocatorList(items)
        if selector.startswith("[original-title="):
            return _VirtualLocatorList([])
        if selector.startswith("[id^='facet:FacetContainer']"):
            return self.containers
        raise AssertionError(selector)


NO_LAYDOWN_ASSETS = r"^The Gap DAM has no laydown assets for style 440760\.$"


def _patched_search(
    *, clear_side_effect: object, total: int
) -> tuple[ExitStack, _SearchContext, list[tuple[str, str]]]:
    """open_search_page with the browser out of the way: the leftover clear
    does `clear_side_effect`, the results pane says `total`, and every facet
    application is appended to the returned list instead of clicked."""
    page = _SearchPage()
    applied: list[tuple[str, str]] = []
    stack = ExitStack()
    stack.enter_context(
        patch("dam_scrape._find_authenticated_page", return_value=page)
    )
    stack.enter_context(patch("dam_scrape.find_visible", return_value=page.search))
    stack.enter_context(
        patch("dam_scrape.wait_for_post", side_effect=lambda _, action, __: action())
    )
    stack.enter_context(
        patch("dam_scrape.clear_facet_filters", side_effect=clear_side_effect)
    )
    stack.enter_context(patch("dam_scrape.read_result_total", return_value=total))
    stack.enter_context(
        patch(
            "dam_scrape.apply_exclusive_facet",
            side_effect=lambda _, title, value, __: applied.append((title, value)),
        )
    )
    return stack, _SearchContext(page), applied


class LeftoverShotRequestTests(unittest.TestCase):
    """The DAM keeps the Shot Request ID a run leaves checked (P01, say), so
    every search first unchecks it - and now and then the DAM will not let it
    go. That ends a style search, whose plan is built from the facet; a text
    search needs only FINAL and Shot Type L, so it goes on and says so."""

    STUCK = "Shot Request ID P01 did not update."

    def test_style_search_still_fails_on_a_stuck_leftover(self) -> None:
        stack, context, applied = _patched_search(
            clear_side_effect=ScrapeError(self.STUCK), total=22
        )
        with stack, self.assertRaisesRegex(
            ScrapeError, r"^Shot Request ID P01 did not update\.$"
        ):
            open_search_page(context, "https://dam.test", "440760", 100)
        self.assertEqual(applied, [])

    def test_text_search_goes_on_under_a_stuck_leftover_and_says_so(self) -> None:
        stack, context, applied = _patched_search(
            clear_side_effect=ScrapeError(self.STUCK), total=22
        )
        stderr = io.StringIO()
        with stack, redirect_stderr(stderr):
            page = open_search_page(
                context,
                "https://dam.test",
                "blue hoodie",
                100,
                subject="the search 'blue hoodie'",
                shot_request_clear_required=False,
            )
        self.assertIs(page, context.page)
        self.assertEqual(applied, [(ASSET_PRODUCTION_TYPE, FINAL_ASSET_VALUE)])
        self.assertIn(self.STUCK, stderr.getvalue())
        self.assertIn("the search 'blue hoodie'", stderr.getvalue())
        self.assertIn("FINAL and Shot Type L still apply", stderr.getvalue())

    def test_text_search_still_reports_a_facet_missing_with_results(self) -> None:
        # Results but no facet is the sidebar failing to render, not a
        # leftover: an operational error in either mode.
        missing_facet = FacetUnavailableError(
            "The Shot Request ID filter was not available."
        )
        stack, context, applied = _patched_search(
            clear_side_effect=missing_facet, total=7
        )
        with stack, self.assertRaises(FacetUnavailableError):
            open_search_page(
                context,
                "https://dam.test",
                "blue hoodie",
                100,
                subject="the search 'blue hoodie'",
                shot_request_clear_required=False,
            )
        self.assertEqual(applied, [])


class BatchLimitTests(unittest.TestCase):
    def test_style_batch_short_of_its_plan_is_an_error(self) -> None:
        with self.assertRaisesRegex(
            ScrapeError,
            r"^Shot Request ID AV5 for style 671304 returned 2 assets; "
            r"the selection plan expected 3\.$",
        ):
            batch_limit("style 671304", "AV5", 3, 2)

    def test_batch_takes_its_plan_when_the_page_holds_it(self) -> None:
        self.assertEqual(batch_limit("style 671304", "AV5", 3, 3), 3)
        self.assertEqual(batch_limit("the search 'x'", None, 50, 900), 50)

    def test_text_search_takes_what_the_page_holds(self) -> None:
        self.assertEqual(batch_limit("the search 'x'", None, 50, 12), 12)


class EmptySearchTests(unittest.TestCase):
    """An empty search must read as "no laydown assets for this style", not as
    the Shot Request ID facet going missing - which is what the sidebar does
    when there is nothing left to facet."""

    def _patched_search(
        self, *, clear_side_effect: object, total: int
    ) -> tuple[ExitStack, _SearchContext, list[tuple[str, str]]]:
        return _patched_search(clear_side_effect=clear_side_effect, total=total)

    def test_empty_search_with_missing_facet_names_the_style(self) -> None:
        # Nothing was left checked from the previous run, so the sidebar never
        # shows a Shot Request ID section for an empty search.
        missing_facet = FacetUnavailableError(
            "The Shot Request ID filter was not available."
        )
        stack, context, applied = self._patched_search(
            clear_side_effect=missing_facet, total=0
        )
        with stack, self.assertRaisesRegex(ScrapeError, NO_LAYDOWN_ASSETS) as caught:
            open_search_page(context, "https://dam.test", "440760", 100)
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual(applied, [])

    def test_empty_search_after_clearing_leftover_filter_names_the_style(self) -> None:
        # The leftover P01 came off cleanly and the facet stayed rendered, yet
        # the style still has nothing.
        stack, context, applied = self._patched_search(
            clear_side_effect=lambda *_: (), total=0
        )
        with stack, self.assertRaisesRegex(ScrapeError, NO_LAYDOWN_ASSETS):
            open_search_page(context, "https://dam.test", "440760", 100)
        self.assertEqual(applied, [])

    def test_empty_text_search_is_named_by_its_subject(self) -> None:
        stack, context, applied = self._patched_search(
            clear_side_effect=lambda *_: (), total=0
        )
        with stack, self.assertRaisesRegex(
            NoLaydownAssetsError,
            r"^The Gap DAM has no laydown assets for the search 'blue hoodie'\.$",
        ):
            open_search_page(
                context, "https://dam.test", "blue hoodie", 100,
                subject="the search 'blue hoodie'",
            )
        self.assertEqual(applied, [])

    def test_no_final_image_is_named_for_the_subject(self) -> None:
        no_final = NoFinalImageError("No FINAL image is available for this style.")
        for query, subject, expected in (
            ("523570", None, "style 523570"),
            ("blue hoodie", "the search 'blue hoodie'", "the search 'blue hoodie'"),
        ):
            stack, context, _ = self._patched_search(
                clear_side_effect=lambda *_: (), total=2
            )
            with stack:
                stack.enter_context(
                    patch("dam_scrape.apply_exclusive_facet", side_effect=no_final)
                )
                with self.assertRaisesRegex(
                    NoLaydownAssetsError,
                    rf"^The Gap DAM has laydown shots for {expected}, but none of them is FINAL\.$",
                ):
                    open_search_page(context, "https://dam.test", query, 100, subject=subject)

    def test_missing_facet_with_results_remains_operational_error(self) -> None:
        missing_facet = FacetUnavailableError(
            "The Shot Request ID filter was not available."
        )
        stack, context, applied = self._patched_search(
            clear_side_effect=missing_facet, total=7
        )
        with stack, self.assertRaisesRegex(
            FacetUnavailableError, r"^The Shot Request ID filter was not available\.$"
        ):
            open_search_page(context, "https://dam.test", "440760", 100)
        self.assertEqual(applied, [])


class _CountBody:
    def __init__(self, text: str) -> None:
        self.text = text

    def inner_text(self, timeout: int) -> str:
        return self.text


class _CountFrame:
    def __init__(self, text: str) -> None:
        self.text = text

    def locator(self, selector: str) -> _CountBody:
        if selector != "body":
            raise AssertionError(selector)
        return _CountBody(self.text)


class _CountPage:
    def __init__(self, *texts: str) -> None:
        self.frames = [_CountFrame(text) for text in texts]


class ResultTotalTests(unittest.TestCase):
    def test_empty_search_reads_as_zero(self) -> None:
        page = _CountPage(
            "Photo Studio Filters",
            'Gap Standard Folder 0 - 0 of 0 No matches found in "Gap"',
        )
        self.assertEqual(read_result_total(page, 100), 0)

    def test_count_is_taken_from_whichever_frame_renders_it(self) -> None:
        page = _CountPage("File import date", "Gap Standard Folder 1 - 50 of 14507")
        self.assertEqual(read_result_total(page, 100), 14507)

    def test_missing_count_times_out(self) -> None:
        page = _CountPage("Photo Studio Filters", "Loading")
        with self.assertRaisesRegex(ScrapeError, r"did not finish loading"):
            read_result_total(page, 100)


class AssetCardSelectionTests(unittest.TestCase):
    def test_duplicate_filenames_select_distinct_asset_cards(self) -> None:
        filename = "PB_gp_4401137_1_RAV5_56288657.psd"
        cards = [_AssetCard(filename), _AssetCard(filename)]
        page = _AssetPage(cards)

        self.assertEqual(
            select_asset_limit(page, limit=2, timeout_ms=100),
            (filename, filename),
        )
        self.assertEqual([card.clicks for card in cards], [1, 1])


class _AssetCard:
    def __init__(self, filename: str) -> None:
        self.label = f"Gap Image: {filename}"
        self.clicks = 0

    def get_attribute(self, name: str) -> str | None:
        return self.label if name == "aria-label" else None

    def is_visible(self) -> bool:
        return True

    def bounding_box(self) -> dict[str, int]:
        return {"width": 200, "height": 300}

    def click(self, **_: object) -> None:
        self.clicks += 1


class _AssetCardList:
    def __init__(self, cards: list[_AssetCard]) -> None:
        self.cards = cards

    @property
    def first(self) -> _AssetCard:
        return self.cards[0]

    def count(self) -> int:
        return len(self.cards)

    def nth(self, index: int) -> _AssetCard:
        return self.cards[index]


class _AssetFrame:
    def __init__(self, cards: list[_AssetCard]) -> None:
        self.cards = cards

    def locator(self, selector: str) -> _AssetCardList:
        if selector == "[role='region'][aria-label^='Gap Image:']":
            return _AssetCardList(self.cards)
        return _AssetCardList([])

    def get_by_role(self, *_: object, **__: object) -> None:
        raise RuntimeError("strict mode violation: duplicate accessible name")


class _AssetPage:
    def __init__(self, cards: list[_AssetCard]) -> None:
        self.frames = [_AssetFrame(cards)]

    def expect_response(self, *_: object, **__: object) -> nullcontext[None]:
        return nullcontext()

    def wait_for_timeout(self, _: int) -> None:
        pass


class _PopupCount:
    def __init__(self, page: "_PopupPage") -> None:
        self.page = page

    def count(self) -> int:
        return self.page.open_popups


class _PopupFrame:
    def __init__(self, page: "_PopupPage") -> None:
        self.page = page

    def locator(self, selector: str) -> _PopupCount:
        if selector != "#PopupLayer *:visible":
            raise AssertionError(selector)
        return _PopupCount(self.page)


class _Keyboard:
    def __init__(self, page: "_PopupPage") -> None:
        self.page = page
        self.presses: list[str] = []

    def press(self, key: str) -> None:
        self.presses.append(key)
        self.page.open_popups = max(self.page.open_popups - 1, 0)


class _PopupPage:
    def __init__(self, open_popups: int) -> None:
        self.open_popups = open_popups
        self.frames = [_PopupFrame(self)]
        self.keyboard = _Keyboard(self)

    def wait_for_timeout(self, _: int) -> None:
        pass


class PopupDismissalTests(unittest.TestCase):
    def test_nothing_is_pressed_when_nothing_floats_over_the_results(self) -> None:
        page = _PopupPage(open_popups=0)
        dismiss_popups(page)
        self.assertEqual(page.keyboard.presses, [])

    def test_escape_is_pressed_until_the_popup_layer_is_empty(self) -> None:
        page = _PopupPage(open_popups=2)
        dismiss_popups(page)
        self.assertEqual(page.keyboard.presses, ["Escape", "Escape"])
        self.assertEqual(page.open_popups, 0)


class StoredLoginRetryTests(unittest.TestCase):
    """A missing or dead session is not the operator's problem when a login is
    stored: the scraper signs in once and carries on. Without one it stops and
    names both fixes, and a fresh session the DAM rejects is not retried."""

    STORED = (
        dam_auth.Credentials("person@example.com", "secret-value"),
        "DAM_LOGIN_ID and DAM_PASSWORD",
    )
    REJECTED = SessionRejectedError("The saved DAM session was rejected.")

    def _run(
        self,
        *,
        state: object = None,
        style: list[object],
        stored: object = None,
        sign_in: object = None,
    ) -> tuple[int, str, str, object, object]:
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory, ExitStack() as stack:
            auth_state = Path(temporary_directory) / "state.json"
            stack.enter_context(patch("dam_scrape.load_storage_state", side_effect=state))
            download_style = stack.enter_context(
                patch("dam_scrape.download_style", side_effect=style)
            )
            stack.enter_context(patch("dam_scrape.stored_credentials", return_value=stored))
            signed_in = stack.enter_context(patch("dam_scrape.sign_in", side_effect=sign_in))
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            code = main(["440760022", "--auth-state", str(auth_state)])
            self.auth_state = auth_state.resolve()
        return code, out.getvalue(), err.getvalue(), download_style, signed_in

    def test_rejected_session_signs_in_once_and_tries_again(self) -> None:
        manifest = Path("/tmp/downloads/440760/manifest.json")
        code, out, err, download_style, signed_in = self._run(
            style=[self.REJECTED, manifest], stored=self.STORED
        )
        self.assertEqual(code, 0)
        self.assertEqual(download_style.call_count, 2)
        signed_in.assert_called_once()
        self.assertEqual(signed_in.call_args.args, (self.auth_state,))
        self.assertEqual(signed_in.call_args.kwargs["credentials"], self.STORED[0])
        self.assertIn(
            "The saved DAM session was rejected. Signing in with the login from "
            "DAM_LOGIN_ID and DAM_PASSWORD.",
            err,
        )
        self.assertEqual(out.splitlines()[-1], f"manifest {manifest}")

    def test_missing_state_signs_in_before_the_first_search(self) -> None:
        manifest = Path("/tmp/downloads/440760/manifest.json")
        code, out, err, download_style, signed_in = self._run(
            state=AuthStateError("Authentication state does not exist: /x/state.json"),
            style=[manifest],
            stored=self.STORED,
        )
        self.assertEqual(code, 0)
        signed_in.assert_called_once()
        self.assertEqual(download_style.call_count, 1)
        self.assertIn("does not exist: /x/state.json Signing in with the login from", err)
        self.assertEqual(out.splitlines()[-1], f"manifest {manifest}")

    def test_no_stored_login_stops_and_names_both_fixes(self) -> None:
        code, _, err, download_style, signed_in = self._run(style=[self.REJECTED], stored=None)
        self.assertEqual(code, 2)
        signed_in.assert_not_called()
        self.assertEqual(download_style.call_count, 1)
        self.assertIn("The saved DAM session was rejected. No stored login to sign in with", err)
        self.assertIn("dam_auth.py capture", err)
        self.assertIn("DAM_LOGIN_ID and DAM_PASSWORD", err)

    def test_a_fresh_session_rejected_again_is_not_retried(self) -> None:
        code, _, err, download_style, signed_in = self._run(
            style=[self.REJECTED, self.REJECTED], stored=self.STORED
        )
        self.assertEqual(code, 2)
        signed_in.assert_called_once()
        self.assertEqual(download_style.call_count, 2)
        self.assertIn("DAM download failed: The saved DAM session was rejected.", err)

    def test_a_refused_login_is_reported(self) -> None:
        code, _, err, download_style, signed_in = self._run(
            style=[self.REJECTED],
            stored=self.STORED,
            sign_in=AuthenticationExpired(
                "Gap SSO rejected the credentials or restarted sign-in."
            ),
        )
        self.assertEqual(code, 2)
        signed_in.assert_called_once()
        self.assertEqual(download_style.call_count, 1)
        self.assertIn("DAM download failed: Gap SSO rejected the credentials", err)


class ItemDetailsTests(unittest.TestCase):
    """--item-details on its own searches the text; next to a style it is the
    fallback for the one failure a style search can have, no laydown assets."""

    def _run_main(self, argv: list[str], **patches: object) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch("dam_scrape.load_storage_state"))
            for name, value in patches.items():
                stack.enter_context(patch(f"dam_scrape.{name}", **value))
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_neither_search_is_a_usage_error(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main([])
        self.assertEqual(caught.exception.code, 2)

    def test_text_alone_downloads_first_results_and_names_its_manifest(self) -> None:
        manifest = Path("/tmp/downloads/item-details/blue-hoodie/manifest.json")
        style = {"side_effect": AssertionError("the style search must not run")}
        details = {"return_value": manifest}
        code, out, _ = self._run_main(
            ["--item-details", "blue hoodie"],
            download_style=style, download_item_details=details,
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[-1], f"manifest {manifest}")

    def test_nothing_to_download_is_its_own_exit_status(self) -> None:
        miss = NoLaydownAssetsError("The Gap DAM has no laydown assets for style 440760.")
        code, _, err = self._run_main(["440760022"], download_style={"side_effect": miss})
        self.assertEqual(code, EXIT_NOTHING_TO_DOWNLOAD)
        self.assertIn("DAM has nothing to download: The Gap DAM has no laydown assets", err)

        text_miss = NoLaydownAssetsError('The Gap DAM has no laydown assets for "blue hoodie".')
        code, _, err = self._run_main(
            ["440760022", "--item-details", "blue hoodie"],
            download_style={"side_effect": miss}, download_item_details={"side_effect": text_miss},
        )
        self.assertEqual(code, EXIT_NOTHING_TO_DOWNLOAD)
        self.assertIn("Falling back to --item-details", err)
        self.assertIn('no laydown assets for "blue hoodie"', err)

    def test_style_with_no_assets_falls_back_to_the_text(self) -> None:
        manifest = Path("/tmp/downloads/item-details/blue-hoodie/manifest.json")
        style = {
            "side_effect": NoLaydownAssetsError(
                "The Gap DAM has no laydown assets for style 440760."
            )
        }
        details = {"return_value": manifest}
        code, out, err = self._run_main(
            ["440760022", "--item-details", "blue hoodie"],
            download_style=style, download_item_details=details,
        )
        self.assertEqual(code, 0)
        self.assertIn("no laydown assets for style 440760. Falling back to --item-details 'blue hoodie'.", err)
        self.assertEqual(out.splitlines()[-1], f"manifest {manifest}")

    def test_style_with_no_final_image_falls_back_to_the_text(self) -> None:
        manifest = Path("/tmp/downloads/item-details/blue-hoodie/manifest.json")
        style = {
            "side_effect": NoFinalImageError(
                "The Gap DAM has laydown shots for style 523570, but none of them is FINAL."
            )
        }
        details = {"return_value": manifest}
        code, out, err = self._run_main(
            ["523570022", "--item-details", "blue hoodie"],
            download_style=style, download_item_details=details,
        )
        self.assertEqual(code, 0)
        self.assertIn("none of them is FINAL. Falling back to --item-details 'blue hoodie'.", err)
        self.assertEqual(out.splitlines()[-1], f"manifest {manifest}")

    def test_style_with_no_final_image_and_no_text_is_nothing_to_download(self) -> None:
        style = {
            "side_effect": NoFinalImageError(
                "The Gap DAM has laydown shots for style 523570, but none of them is FINAL."
            )
        }
        code, _, err = self._run_main(["523570022"], download_style=style)
        self.assertEqual(code, EXIT_NOTHING_TO_DOWNLOAD)
        self.assertEqual(
            err.strip(),
            "DAM has nothing to download: The Gap DAM has laydown shots for style 523570, "
            "but none of them is FINAL.",
        )

    def test_style_that_works_never_touches_the_text(self) -> None:
        manifest = Path("/tmp/downloads/440760/manifest.json")
        style = {"return_value": manifest}
        details = {"side_effect": AssertionError("the text search must not run")}
        code, out, _ = self._run_main(
            ["440760022", "--item-details", "blue hoodie"],
            download_style=style, download_item_details=details,
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[-1], f"manifest {manifest}")

    def test_style_with_no_assets_and_no_text_still_fails_plainly(self) -> None:
        style = {
            "side_effect": NoLaydownAssetsError(
                "The Gap DAM has no laydown assets for style 440760."
            )
        }
        details = {"side_effect": AssertionError("there is no text to fall back to")}
        code, out, err = self._run_main(
            ["440760022"], download_style=style, download_item_details=details
        )
        self.assertEqual(code, EXIT_NOTHING_TO_DOWNLOAD)
        self.assertEqual(
            err.strip(),
            "DAM has nothing to download: The Gap DAM has no laydown assets for style 440760.",
        )
        self.assertEqual(out, "")

    def test_other_style_failures_do_not_fall_back(self) -> None:
        style = {"side_effect": ScrapeError("No FINAL image is available for this style.")}
        details = {"side_effect": AssertionError("only an empty search falls back")}
        code, _, err = self._run_main(
            ["440760022", "--item-details", "blue hoodie"],
            download_style=style, download_item_details=details,
        )
        self.assertEqual(code, 2)
        self.assertIn("No FINAL image is available for this style.", err)

    def test_text_manifest_is_reused_only_under_the_current_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            (output_directory / "assets.zip").touch()
            manifest = {
                "status": "complete",
                "filters": dict(REQUIRED_FILTERS),
                "archives": [{"filename": "assets.zip"}],
                "shot_request_policy": {
                    "mode": "first_results",
                    "first_results_limit": ITEM_DETAILS_LIMIT,
                    "selected_batches": [
                        {"shot_request_id": None, "available": 900, "selected": 50}
                    ],
                },
            }
            self.assertTrue(is_complete_manifest_reusable(manifest, output_directory))

            # Written under a smaller page size, with more left to take: refetch.
            manifest["shot_request_policy"]["first_results_limit"] = 20
            manifest["shot_request_policy"]["selected_batches"][0]["selected"] = 20
            self.assertFalse(is_complete_manifest_reusable(manifest, output_directory))

            # Smaller page size, but the search only ever had 12: nothing to gain.
            manifest["shot_request_policy"]["selected_batches"] = [
                {"shot_request_id": None, "available": 12, "selected": 12}
            ]
            self.assertTrue(is_complete_manifest_reusable(manifest, output_directory))

    def test_text_manifest_pulled_under_a_leftover_filter_is_refetched(self) -> None:
        # The search ran with P01 still checked from an earlier run, so its
        # "everything" was P01's everything. A rerun may get the filter off.
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            (output_directory / "assets.zip").touch()
            manifest = {
                "status": "complete",
                "filters": dict(REQUIRED_FILTERS),
                "archives": [{"filename": "assets.zip"}],
                "shot_request_policy": {
                    "mode": "first_results",
                    "first_results_limit": ITEM_DETAILS_LIMIT,
                    "leftover_shot_request_ids": ["P01"],
                    "selected_batches": [
                        {"shot_request_id": None, "available": 2, "selected": 2}
                    ],
                },
            }
            self.assertFalse(is_complete_manifest_reusable(manifest, output_directory))

            manifest["shot_request_policy"]["leftover_shot_request_ids"] = []
            self.assertTrue(is_complete_manifest_reusable(manifest, output_directory))


class ArchiveTests(unittest.TestCase):
    def test_jpg_archive_is_inspected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "assets.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("first.jpg", b"one")
                archive.writestr("second.jpeg", b"two")

            self.assertEqual(
                inspect_jpg_archive(archive_path, expected_count=2),
                [
                    {"filename": "first.jpg", "bytes": 3},
                    {"filename": "second.jpeg", "bytes": 3},
                ],
            )

    def test_archive_count_must_match_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "assets.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("only.jpg", b"one")

            with self.assertRaisesRegex(ScrapeError, "Expected 2"):
                inspect_jpg_archive(archive_path, expected_count=2)

    def test_archive_members_are_named_by_bare_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "assets.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("853417/first.jpg", b"one")
                archive.writestr("853417/nested/second.jpg", b"two")

            self.assertEqual(
                inspect_jpg_archive(archive_path, expected_count=2),
                [
                    {"filename": "first.jpg", "bytes": 3},
                    {"filename": "second.jpg", "bytes": 3},
                ],
            )

    def test_archive_rejects_two_members_with_the_same_bare_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "assets.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("a/same.jpg", b"one")
                archive.writestr("b/same.jpg", b"two")

            with self.assertRaisesRegex(ScrapeError, "two files named same.jpg"):
                inspect_jpg_archive(archive_path, expected_count=2)

    def test_single_asset_sent_bare_is_wrapped_into_an_archive(self) -> None:
        # The DAM sends one selected asset as the JPG itself, not a ZIP of one.
        with tempfile.TemporaryDirectory() as temporary_directory:
            download_path = Path(temporary_directory) / "assets.zip.part"
            jpg_bytes = b"\xff\xd8\xff\xe0" + b"laydown"
            download_path.write_bytes(jpg_bytes)

            wrapped = wrap_bare_jpg_as_archive(download_path, "PB_gp_1_RAV5_2.jpg")

            self.assertEqual(wrapped, "PB_gp_1_RAV5_2.jpg")
            with zipfile.ZipFile(download_path) as archive:
                self.assertEqual(archive.namelist(), ["PB_gp_1_RAV5_2.jpg"])
                self.assertEqual(archive.read("PB_gp_1_RAV5_2.jpg"), jpg_bytes)
            self.assertEqual(
                inspect_jpg_archive(download_path, expected_count=1),
                [{"filename": "PB_gp_1_RAV5_2.jpg", "bytes": len(jpg_bytes)}],
            )

    def test_archive_sent_by_the_dam_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            download_path = Path(temporary_directory) / "assets.zip.part"
            with zipfile.ZipFile(download_path, "w") as archive:
                archive.writestr("first.jpg", b"one")
                archive.writestr("second.jpg", b"two")
            sent = download_path.read_bytes()

            self.assertIsNone(wrap_bare_jpg_as_archive(download_path, "assets.zip"))

            self.assertEqual(download_path.read_bytes(), sent)

    def test_bare_download_that_is_not_a_jpg_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            download_path = Path(temporary_directory) / "assets.zip.part"
            download_path.write_bytes(b"<html><body>Sign in</body></html>")

            with self.assertRaisesRegex(
                ScrapeError, "neither a ZIP archive nor a JPG .*'login.html'"
            ):
                wrap_bare_jpg_as_archive(download_path, "login.html")

    def test_jpg_bytes_under_a_non_image_name_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            download_path = Path(temporary_directory) / "assets.zip.part"
            download_path.write_bytes(b"\xff\xd8\xff\xe0laydown")

            with self.assertRaisesRegex(ScrapeError, "'download.bin'"):
                wrap_bare_jpg_as_archive(download_path, "download.bin")

    def test_extract_archive_writes_members_flat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "assets.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("853417/", b"")
                archive.writestr("853417/first.jpg", b"one")
                archive.writestr("853417/nested/second.jpg", b"two")
            destination = Path(temporary_directory) / "library"

            extract_archive(archive_path, destination)

            self.assertEqual(
                sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*")),
                ["first.jpg", "second.jpg"],
            )
            self.assertEqual((destination / "first.jpg").read_bytes(), b"one")
            self.assertEqual((destination / "second.jpg").read_bytes(), b"two")

    def test_archive_rejects_unsafe_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "assets.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.jpg", b"one")

            with self.assertRaisesRegex(ScrapeError, "Unsafe path"):
                inspect_jpg_archive(archive_path, expected_count=1)


class ManifestTests(unittest.TestCase):
    def test_complete_manifest_requires_final_filter_for_cache_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            (output_directory / "assets.zip").touch()
            manifest = {
                "status": "complete",
                "filters": {"Shot Type": "L"},
                "archives": [{"filename": "assets.zip"}],
                "shot_request_policy": {"maximum_per_code": MAX_PER_CODE},
            }

            self.assertFalse(
                is_complete_manifest_reusable(manifest, output_directory)
            )

            manifest["filters"][ASSET_PRODUCTION_TYPE] = FINAL_ASSET_VALUE
            self.assertTrue(
                is_complete_manifest_reusable(manifest, output_directory)
            )

    def test_manifest_from_a_smaller_cap_is_refetched_unless_it_took_everything(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            (output_directory / "assets.zip").touch()
            manifest = {
                "status": "complete",
                "filters": dict(REQUIRED_FILTERS),
                "archives": [{"filename": "assets.zip"}],
                "shot_request_policy": {
                    "maximum_per_code": 3,
                    "selected_batches": [
                        {"shot_request_id": "AV5", "available": 35, "selected": 3}
                    ],
                },
            }
            # 3 of 35 under the old cap: a rerun would now take 10, so refetch.
            self.assertFalse(
                is_complete_manifest_reusable(manifest, output_directory)
            )

            # 2 of 2: no cap would change the selection, so keep it.
            manifest["shot_request_policy"]["selected_batches"] = [
                {"shot_request_id": "AV5", "available": 2, "selected": 2}
            ]
            self.assertTrue(
                is_complete_manifest_reusable(manifest, output_directory)
            )

            # A manifest with no policy record cannot be judged; refetch.
            del manifest["shot_request_policy"]
            self.assertFalse(
                is_complete_manifest_reusable(manifest, output_directory)
            )

    def test_json_write_replaces_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            write_json_atomic(path, {"status": "complete"})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "complete"},
            )


if __name__ == "__main__":
    unittest.main()


class CanceledDownloadTests(unittest.TestCase):
    def test_canceled_transfer_is_recognised(self) -> None:
        self.assertTrue(is_canceled_download(PlaywrightError("Download.save_as: canceled")))
        self.assertFalse(is_canceled_download(PlaywrightError("Download.save_as: no such file")))
        self.assertFalse(is_canceled_download(ScrapeError("canceled")))

    def test_browser_error_is_a_clean_failure_not_a_traceback(self) -> None:
        # Before this, "Download.save_as: canceled" escaped main as a traceback
        # with exit 1; a caller could not tell it from a crash.
        stderr = io.StringIO()
        with (
            patch(
                "dam_scrape.download_with_session",
                side_effect=PlaywrightError("Download.save_as: canceled"),
            ),
            redirect_stderr(stderr),
            redirect_stdout(io.StringIO()),
        ):
            code = main(["--item-details", "blue hoodie"])
        self.assertEqual(code, 2)
        self.assertIn("DAM download failed: Download.save_as: canceled", stderr.getvalue())
