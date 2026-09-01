from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from contextlib import nullcontext
from pathlib import Path

from dam_scrape import (
    ScrapeError,
    ShotBatch,
    choose_shot_batches,
    inspect_jpg_archive,
    normalize_style_number,
    parse_selected_asset_count,
    parse_total_result_count,
    safe_query_directory,
    select_asset_limit,
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
