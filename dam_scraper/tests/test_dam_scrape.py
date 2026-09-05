from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from dam_scrape import (
    ASSET_PRODUCTION_TYPE,
    FINAL_ASSET_VALUE,
    FacetOption,
    FacetUnavailableError,
    ITEM_DETAILS_LIMIT,
    MAX_PER_CODE,
    NoLaydownAssetsError,
    REQUIRED_FILTERS,
    ScrapeError,
    SHOT_REQUEST_ID,
    ShotBatch,
    apply_exclusive_facet,
    choose_shot_batches,
    dismiss_popups,
    extract_archive,
    find_facet_containers,
    inspect_jpg_archive,
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
    def __init__(self) -> None:
        self.clicks = 0

    def is_checked(self) -> bool:
        return False

    def evaluate(self, _: str) -> None:
        self.clicks += 1


class _FacetCheckboxCandidates:
    def __init__(self, checkbox: _FacetCheckbox) -> None:
        self.first = checkbox

    def count(self) -> int:
        return 1


class _FacetContainers:
    def __init__(self) -> None:
        self.checkbox = _FacetCheckbox()
        self.selector = ""

    def locator(self, selector: str) -> _FacetCheckboxCandidates:
        self.selector = selector
        return _FacetCheckboxCandidates(self.checkbox)


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


class EmptySearchTests(unittest.TestCase):
    """An empty search must read as "no laydown assets for this style", not as
    the Shot Request ID facet going missing - which is what the sidebar does
    when there is nothing left to facet."""

    def _patched_search(
        self, *, clear_side_effect: object, total: int
    ) -> tuple[ExitStack, _SearchContext, list[tuple[str, str]]]:
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
        self.assertEqual(code, 2)
        self.assertEqual(
            err.strip(),
            "DAM download failed: The Gap DAM has no laydown assets for style 440760.",
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
