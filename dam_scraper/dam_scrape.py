"""Search the Gap DAM and bulk-download matching assets as full-resolution JPGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

from auth_session import AuthStateError, load_storage_state
from dam_auth import DEFAULT_AUTH_STATE, _find_authenticated_page


DEFAULT_SEARCH_URL = (
    "https://digitalassets.gapinc.com/asset-management/270HRGZOLFO1H"
    "?WS=270H397TAWK&TP=default&Flat=FP&FR_=1&W=858&H=1016"
    "#/DamView&TP=default&VBID=270HZOLSY3NGJ&PN=1&WS=270H397TAWK"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


class ScrapeError(RuntimeError):
    """The requested DAM job could not be completed safely."""


@dataclass(frozen=True)
class SearchResults:
    total: int
    source_filenames: tuple[str, ...]


@dataclass(frozen=True)
class ShotBatch:
    code: str
    limit: int


def choose_shot_batches(counts: dict[str, int]) -> tuple[ShotBatch, ...]:
    available = {code: count for code, count in counts.items() if count > 0}
    for preferred_code in ("P01", "AV5"):
        if preferred_code in available:
            return (ShotBatch(preferred_code, min(3, available[preferred_code])),)
    batches = tuple(
        ShotBatch(code, min(3, count)) for code, count in sorted(available.items())
    )
    if not batches:
        raise ScrapeError("No Shot Request ID values are available for this style.")
    return batches


def normalize_style_number(value: str) -> str:
    style_number = value.strip()
    if not re.fullmatch(r"\d{6}(?:\d{3})?", style_number):
        raise ScrapeError("The style number must contain exactly 6 or 9 digits.")
    return style_number[:6]


def safe_query_directory(query: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", query.strip()).strip("-.")
    if not normalized:
        raise ScrapeError("The query must contain a letter or number.")
    return normalized


def parse_total_result_count(text: str) -> int | None:
    matches = re.findall(r"\b\d+\s*-\s*\d+\s+of\s+(\d+)\b", text)
    return max((int(match) for match in matches), default=None)


def parse_selected_asset_count(text: str) -> int | None:
    match = re.search(r"\b(\d+)\s+assets?\b", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_jpg_archive(path: Path, expected_count: int) -> list[dict[str, object]]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = []
            for info in archive.infolist():
                member_path = Path(info.filename)
                if info.is_dir():
                    continue
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ScrapeError(f"Unsafe path in DAM archive: {info.filename}")
                if member_path.suffix.casefold() not in IMAGE_SUFFIXES:
                    raise ScrapeError(
                        f"DAM archive contained a non-JPG file: {info.filename}"
                    )
                members.append({"filename": info.filename, "bytes": info.file_size})
    except zipfile.BadZipFile as exc:
        raise ScrapeError("The DAM response was not a valid ZIP archive.") from exc

    if len(members) != expected_count:
        raise ScrapeError(
            f"Expected {expected_count} JPG files, but the archive contained {len(members)}."
        )
    return members


def extract_archive(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        json.dump(value, temporary_file, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def find_visible(page: Any, selectors: Iterable[str], timeout_ms: int) -> Any:
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        for frame in page.frames:
            for selector in selectors:
                candidates = frame.locator(selector)
                for index in range(candidates.count()):
                    candidate = candidates.nth(index)
                    if candidate.is_visible():
                        return candidate
        if time.monotonic() >= deadline:
            raise ScrapeError(f"Required DAM control was not visible: {tuple(selectors)}")
        time.sleep(0.25)


def find_existing(page: Any, selectors: Iterable[str], timeout_ms: int) -> Any:
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        for frame in page.frames:
            for selector in selectors:
                candidate = frame.locator(selector)
                if candidate.count() > 0:
                    return candidate.first
        if time.monotonic() >= deadline:
            raise ScrapeError(f"Required DAM control was not found: {tuple(selectors)}")
        time.sleep(0.25)


def read_search_results(page: Any, timeout_ms: int) -> SearchResults:
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        filenames: set[str] = set()
        total: int | None = None
        for frame in page.frames:
            try:
                body_text = frame.locator("body").inner_text(timeout=1_000)
                frame_total = parse_total_result_count(body_text)
                if frame_total is not None:
                    total = frame_total
                for alt in frame.locator("img[alt]").evaluate_all(
                    "images => images.map(image => image.alt)"
                ):
                    if Path(alt).suffix.casefold() in {".psd", ".jpg", ".jpeg", ".png"}:
                        filenames.add(alt)
            except PlaywrightTimeoutError:
                continue
        if total == 0:
            raise ScrapeError("The DAM search returned no assets.")
        if total is not None and filenames:
            return SearchResults(total=total, source_filenames=tuple(sorted(filenames)))
        if time.monotonic() >= deadline:
            raise ScrapeError(
                "The DAM results did not finish loading before the timeout."
            )
        time.sleep(0.5)


def wait_for_post(page: Any, action: Any, timeout_ms: int) -> None:
    with page.expect_response(
        lambda response: response.request.method == "POST"
        and response.url.split("?", 1)[0].endswith("/CS.aspx"),
        timeout=timeout_ms,
    ):
        action()


def open_search_page(context: Any, url: str, query: str, timeout_ms: int) -> Any:
    page = context.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    authenticated_page = _find_authenticated_page(context, url)
    if authenticated_page is None:
        raise ScrapeError(
            "The saved DAM session was rejected. Run dam_auth.py capture again."
        )
    search = find_visible(
        authenticated_page,
        (
            "input[placeholder='Type your search here']",
            "[role='textbox'][aria-label='Type your search here']",
        ),
        timeout_ms,
    )
    search.fill(query)
    authenticated_page.wait_for_timeout(500)
    wait_for_post(authenticated_page, lambda: search.press("Enter"), timeout_ms)
    authenticated_page.wait_for_timeout(2_000)
    return authenticated_page


def read_shot_request_facet(
    page: Any, timeout_ms: int
) -> tuple[dict[str, int], dict[str, bool]]:
    sidebar_candidates = [
        frame.locator("aside[aria-label='Photo Studio Filters']")
        for frame in page.frames
    ]
    sidebar = next(
        (
            candidate.first
            for candidate in sidebar_candidates
            if candidate.count() and candidate.first.is_visible()
        ),
        None,
    )
    if sidebar is None:
        find_visible(page, ("a[aria-label='Filter']",), timeout_ms).click()
        sidebar = find_visible(
            page, ("aside[aria-label='Photo Studio Filters']",), timeout_ms
        )
    sidebar.evaluate("element => { element.scrollTop = element.scrollHeight; }")
    page.wait_for_timeout(500)

    heading = sidebar.locator("[original-title='Shot Request ID']").first
    if heading.count() == 0:
        raise ScrapeError("The Shot Request ID filter was not available.")
    header = heading.locator("xpath=ancestor::*[contains(@id, ':HeaderPnl')][1]")
    header_id = header.get_attribute("id") or ""
    prefix = header_id.removesuffix(":HeaderPnl")
    if not prefix:
        raise ScrapeError("The Shot Request ID filter could not be read.")
    containers = sidebar.locator(
        f"[id^='{prefix}:FacetContainer'][data-opn='FacetContainer']"
    )
    values = containers.evaluate_all(
        """items => items.map(item => ({
            code: item.querySelector('input[type=checkbox]')?.getAttribute('aria-label') || '',
            count: Number((item.querySelector('[data-opn=OccurrencesLbl]')?.innerText || '0').trim()),
            checked: Boolean(item.querySelector('input[type=checkbox]')?.checked)
        })).filter(item => item.code)"""
    )
    return (
        {item["code"]: item["count"] for item in values},
        {item["code"]: item["checked"] for item in values},
    )


def toggle_shot_request_checkbox(page: Any, code: str, timeout_ms: int) -> None:
    checkbox = find_existing(page, (f"input[aria-label='{code}']",), timeout_ms)
    expected_checked = not checkbox.is_checked()
    wait_for_post(
        page,
        lambda: checkbox.evaluate("element => element.click()"),
        timeout_ms,
    )
    page.wait_for_timeout(1_500)
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        _, checked = read_shot_request_facet(page, timeout_ms)
        if checked.get(code, False) == expected_checked:
            return
        if time.monotonic() >= deadline:
            raise ScrapeError(f"Shot Request ID {code} did not update.")
        page.wait_for_timeout(500)


def clear_shot_request_filters(page: Any, timeout_ms: int) -> dict[str, int]:
    _, checked = read_shot_request_facet(page, timeout_ms)
    for code, is_checked in checked.items():
        if is_checked:
            toggle_shot_request_checkbox(page, code, timeout_ms)
    counts, _ = read_shot_request_facet(page, timeout_ms)
    return counts


def set_shot_request_filter(page: Any, code: str, timeout_ms: int) -> None:
    counts = clear_shot_request_filters(page, timeout_ms)
    if counts.get(code, 0) <= 0:
        raise ScrapeError(f"Shot Request ID {code} is not available for this style.")
    toggle_shot_request_checkbox(page, code, timeout_ms)

    sidebar = find_visible(
        page, ("aside[aria-label='Photo Studio Filters']",), timeout_ms
    )
    if sidebar.is_visible():
        find_visible(page, ("a[aria-label='Filter']",), timeout_ms).click()
        page.wait_for_timeout(500)


def select_asset_limit(page: Any, limit: int, timeout_ms: int) -> tuple[str, ...]:
    deselect_all = None
    for frame in page.frames:
        candidate = frame.locator("a[aria-label^='Deselect all']")
        if candidate.count() and candidate.first.is_visible():
            deselect_all = candidate.first
            break
    if deselect_all is not None:
        wait_for_post(page, deselect_all.click, timeout_ms)
        page.wait_for_timeout(500)

    deadline = time.monotonic() + timeout_ms / 1000
    assets: list[tuple[Any, str]] = []
    while time.monotonic() < deadline:
        for frame in page.frames:
            regions = frame.locator("[role='region'][aria-label^='Gap Image:']")
            visible_assets = []
            for index in range(regions.count()):
                region = regions.nth(index)
                if region.is_visible():
                    visible_assets.append(
                        (region, region.get_attribute("aria-label") or "")
                    )
            if len(visible_assets) >= limit:
                assets = visible_assets[:limit]
                break
        if assets:
            break
        time.sleep(0.25)
    if len(assets) < limit:
        raise ScrapeError(f"Only {len(assets)} assets loaded; expected {limit}.")

    for region, label in assets:
        box = region.bounding_box()
        if box is None:
            raise ScrapeError(f"Asset card was not visible: {label}")
        position = {"x": max(1, box["width"] - 14), "y": min(20, box["height"] / 2)}
        wait_for_post(
            page,
            lambda region=region, position=position: region.click(position=position),
            timeout_ms,
        )
        page.wait_for_timeout(300)
    return tuple(label.removeprefix("Gap Image: ") for _, label in assets)


def discover_shot_counts(
    *, query: str, auth_state: Path, url: str, headed: bool, timeout_ms: int
) -> dict[str, int]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed)
        context = browser.new_context(storage_state=auth_state)
        page = open_search_page(context, url, query, timeout_ms)
        counts = clear_shot_request_filters(page, timeout_ms)
        browser.close()
        return counts


def download_batch_from_dam(
    *,
    query: str,
    batch: ShotBatch,
    auth_state: Path,
    url: str,
    destination: Path,
    headed: bool,
    timeout_ms: int,
) -> SearchResults:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed)
        context = browser.new_context(storage_state=auth_state, accept_downloads=True)
        authenticated_page = open_search_page(context, url, query, timeout_ms)
        set_shot_request_filter(authenticated_page, batch.code, timeout_ms)
        authenticated_page.wait_for_timeout(1_000)
        results = read_search_results(authenticated_page, timeout_ms)
        if results.total < batch.limit:
            raise ScrapeError(
                f"Shot Request ID {batch.code} returned {results.total} assets; "
                f"the selection plan expected {batch.limit}."
            )
        selected_filenames = select_asset_limit(
            authenticated_page, batch.limit, timeout_ms
        )
        first_asset = find_visible(
            authenticated_page,
            (f"[role='region'][aria-label='Gap Image: {selected_filenames[0]}']",),
            timeout_ms,
        )
        first_asset.locator("xpath=ancestor::*[@data-pv][1]").hover()
        bulk_download = find_visible(
            authenticated_page,
            (
                "button[aria-label='Download']",
                "a[aria-label='Download']",
                "button:has-text('Download')",
            ),
            timeout_ms,
        )
        bulk_download.click()

        original = find_visible(
            authenticated_page, ("input[aria-label='Original']",), timeout_ms
        )
        jpg_original = find_visible(
            authenticated_page,
            ("input[aria-label='JPG Original Resolution']",),
            timeout_ms,
        )
        download_dialog = original.locator(
            "xpath=ancestor::*[@data-vf='DownloadMultipleFormatSelector_VForm']"
        )
        selected_count = parse_selected_asset_count(download_dialog.inner_text())
        if selected_count != batch.limit:
            browser.close()
            raise ScrapeError(
                f"Selected {batch.limit} {batch.code} assets, but the download "
                f"dialog contained {selected_count or 0}."
            )
        if original.is_checked():
            wait_for_post(authenticated_page, original.uncheck, timeout_ms)
        if not jpg_original.is_checked():
            wait_for_post(authenticated_page, jpg_original.check, timeout_ms)

        standard_download = find_visible(
            authenticated_page,
            ("a[aria-label='Standard download']",),
            timeout_ms,
        ).locator("xpath=parent::*")
        with authenticated_page.expect_download(timeout=timeout_ms) as download_info:
            standard_download.click()
        download = download_info.value
        download.save_as(destination)
        browser.close()
        return SearchResults(
            total=batch.limit,
            source_filenames=tuple(selected_filenames),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search the current Gap DAM view, apply the Shot Request ID fallback "
            "policy, and download up to three assets per selected code as JPGs."
        )
    )
    parser.add_argument(
        "style_number",
        help="A 6- or 9-digit style number; 853417012 is searched as 853417.",
    )
    parser.add_argument("--auth-state", type=Path, default=DEFAULT_AUTH_STATE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("DAM_OUTPUT_ROOT", "downloads")),
    )
    parser.add_argument("--url", default=DEFAULT_SEARCH_URL)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        query = normalize_style_number(args.style_number)
        query_directory = safe_query_directory(query)
        auth_state = args.auth_state.expanduser().resolve()
        load_storage_state(auth_state)
        output_directory = args.output_root.expanduser().resolve() / query_directory
        manifest_path = output_directory / "manifest.json"
        if manifest_path.exists() and not args.force:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            archives = manifest.get("archives", [])
            archives_exist = isinstance(archives, list) and archives and all(
                isinstance(archive, dict)
                and isinstance(archive.get("filename"), str)
                and (output_directory / archive["filename"]).is_file()
                for archive in archives
            )
            if manifest.get("status") == "complete" and archives_exist:
                print(f"Already complete: {output_directory}")
                return 0

        output_directory.mkdir(parents=True, exist_ok=True)
        shot_counts = discover_shot_counts(
            query=query,
            auth_state=auth_state,
            url=args.url,
            headed=args.headed,
            timeout_ms=args.timeout_ms,
        )
        plan = choose_shot_batches(shot_counts)
        images_directory = output_directory / "images"
        archives: list[dict[str, object]] = []
        files: list[dict[str, object]] = []
        selected_sources: list[dict[str, object]] = []
        for batch in plan:
            safe_code = safe_query_directory(batch.code)
            archive_path = (
                output_directory
                / f"gap-{query_directory}-{safe_code}-jpg-original.zip"
            )
            partial_path = archive_path.with_suffix(".zip.part")
            results = download_batch_from_dam(
                query=query,
                batch=batch,
                auth_state=auth_state,
                url=args.url,
                destination=partial_path,
                headed=args.headed,
                timeout_ms=args.timeout_ms,
            )
            members = inspect_jpg_archive(partial_path, batch.limit)
            partial_path.replace(archive_path)
            extract_archive(archive_path, images_directory)
            archives.append(
                {
                    "shot_request_id": batch.code,
                    "filename": archive_path.name,
                    "bytes": archive_path.stat().st_size,
                    "sha256": sha256_file(archive_path),
                }
            )
            files.extend(
                {**member, "shot_request_id": batch.code} for member in members
            )
            selected_sources.extend(
                {
                    "filename": filename,
                    "shot_request_id": batch.code,
                }
                for filename in results.source_filenames
            )

        manifest = {
            "status": "complete",
            "input_style_number": args.style_number,
            "query": query,
            "brand_scope": "Gap",
            "filters": {"Shot Type": "L"},
            "shot_request_policy": {
                "preferred_order": ["P01", "AV5"],
                "maximum_per_code": 3,
                "available_counts": shot_counts,
                "selected_batches": [
                    {
                        "shot_request_id": batch.code,
                        "available": shot_counts[batch.code],
                        "selected": batch.limit,
                    }
                    for batch in plan
                ],
            },
            "format": "JPG Original Resolution",
            "download_method": "Standard download",
            "result_count": sum(batch.limit for batch in plan),
            "selected_sources": selected_sources,
            "archives": archives,
            "files": files,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(manifest_path, manifest)
        print(
            f"Downloaded {sum(batch.limit for batch in plan)} JPG files "
            f"to {output_directory}"
        )
        return 0
    except (AuthStateError, ScrapeError, PlaywrightTimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"DAM download failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
