from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from dam_scrape import (
    ASSET_PRODUCTION_TYPE,
    FINAL_ASSET_VALUE,
    FacetOption,
    FacetUnavailableError,
    ScrapeError,
    SHOT_REQUEST_ID,
    ShotBatch,
    apply_exclusive_facet,
    choose_shot_batches,
    find_facet_containers,
    inspect_jpg_archive,
    is_complete_manifest_reusable,
    normalize_style_number,
    open_search_page,
    parse_selected_asset_count,
    parse_total_result_count,
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
    def test_p01_takes_priority_and_is_capped_at_three(self) -> None:
        self.assertEqual(
            choose_shot_batches({"AV5": 8, "P01": 5, "P02": 4}),
            (ShotBatch("P01", 3),),
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
            choose_shot_batches({"AV2": 5, "P02": 2, "AV1": 0}),
            (ShotBatch("AV2", 3), ShotBatch("P02", 2)),
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
            }

            self.assertFalse(
                is_complete_manifest_reusable(manifest, output_directory)
            )

            manifest["filters"][ASSET_PRODUCTION_TYPE] = FINAL_ASSET_VALUE
            self.assertTrue(
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
