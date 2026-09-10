"""Search the Gap DAM and bulk-download matching assets as full-resolution JPGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from auth_session import AuthStateError, AuthenticationExpired, load_storage_state
from dam_auth import (
    DEFAULT_AUTH_STATE,
    LOGIN_ID_ENV,
    PASSWORD_ENV,
    _find_authenticated_page,
    sign_in,
    stored_credentials,
)


DEFAULT_SEARCH_URL = (
    "https://digitalassets.gapinc.com/asset-management/270HRGZOLFO1H"
    "?WS=270H397TAWK&TP=default&Flat=FP&FR_=1&W=858&H=1016"
    "#/DamView&TP=default&VBID=270HZOLSY3NGJ&PN=1&WS=270H397TAWK"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
# The first three bytes of every JPEG file: the SOI marker and the start of
# the marker after it.
JPEG_MAGIC = b"\xff\xd8\xff"
# The DAM's facet titles and the values the scraper insists on. These four were
# dropped by the merge of agen-ivn into agentic-gen-iteration (ad27d62) while
# their uses survived, which is how a run died on NameError at the first
# search; restored from 1ea1b13.
ASSET_PRODUCTION_TYPE = "Asset Production Type"
FINAL_ASSET_VALUE = "FINAL"
SHOT_REQUEST_ID = "Shot Request ID"
REQUIRED_FILTERS = {
    "Shot Type": "L",
    ASSET_PRODUCTION_TYPE: FINAL_ASSET_VALUE,
}
# How many laydown shots a style contributes to the reference library, per
# selected Shot Request ID. The harness picks one reference out of the library,
# so this is the width of its choice for the style, not a number of outputs.
# Recorded in each manifest as maximum_per_code, and a manifest written under
# a smaller cap is not reused - see is_complete_manifest_reusable.
MAX_PER_CODE = 10
# What --item-details takes: the first page of laydown results for the text,
# whatever their Shot Request ID. The DAM pages results 50 at a time, so this
# is exactly the set of cards a search renders before anyone scrolls to page 2.
ITEM_DETAILS_LIMIT = 50
# The exit status for "the DAM was searched and holds nothing for this style
# or text". A miss rather than a failure - the session was good and the search
# ran - and distinct so a caller can go on without the shots. run.sh does:
# it warns and lets the harness search whatever the library already holds.
EXIT_NOTHING_TO_DOWNLOAD = 4
# Text searches keep their ZIP and manifest apart from the per-style folders,
# so a text that happens to look like a style number cannot collide with one.
ITEM_DETAILS_DIRECTORY = "item-details"
DEFAULT_IMAGE_ROOT = Path(__file__).resolve().parent.parent / "inputs" / "reference_library"


class ScrapeError(RuntimeError):
    """The requested DAM job could not be completed safely."""


class SessionRejectedError(ScrapeError):
    """The DAM sent the saved session back to its login page."""


class FacetUnavailableError(ScrapeError):
    """A named DAM facet is not present in the current result set."""


class NoLaydownAssetsError(ScrapeError):
    """The search holds nothing the scraper can download: a miss, not a
    failure. This is what --item-details stands in for, and what exits with
    EXIT_NOTHING_TO_DOWNLOAD when there is no text to fall back on."""


class NoFinalImageError(NoLaydownAssetsError):
    """The search has laydown shots, but none of them is a FINAL asset. The
    scraper insists on FINAL, so this is as much a miss as an empty search."""


@dataclass(frozen=True)
class SearchResults:
    total: int
    source_filenames: tuple[str, ...]


@dataclass(frozen=True)
class BatchDownload:
    """What one bulk download left on disk: the cards that were selected for
    it, and the name the DAM gave the file it sent back. `total` is the count
    the results pane showed when they were selected, and `shot_request_ids`
    the Shot Request ID values checked at that moment: the one code of a style
    batch, or whatever leftover a text search could not clear - () when it
    ran, as intended, under no Shot Request ID at all."""

    selected_filenames: tuple[str, ...]
    suggested_filename: str
    total: int = 0
    shot_request_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShotBatch:
    code: str
    limit: int


@dataclass(frozen=True)
class FacetOption:
    value: str
    count: int
    checked: bool


def choose_shot_batches(counts: dict[str, int]) -> tuple[ShotBatch, ...]:
    available = {code: count for code, count in counts.items() if count > 0}
    for preferred_code in ("P01", "AV5"):
        if preferred_code in available:
            return (
                ShotBatch(preferred_code, min(MAX_PER_CODE, available[preferred_code])),
            )
    batches = tuple(
        ShotBatch(code, min(MAX_PER_CODE, count))
        for code, count in sorted(available.items())
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
    """The JPGs the archive holds, named as they will sit in the reference
    library: by bare filename, whatever folder the DAM wrapped them in. The
    library is one flat folder - the harness picks a reference from everything
    in it - so a folder inside the ZIP is dropped, not reproduced."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = []
            seen: set[str] = set()
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
                if member_path.name in seen:
                    raise ScrapeError(
                        f"DAM archive holds two files named {member_path.name}; "
                        "they cannot both land in the flat reference library."
                    )
                seen.add(member_path.name)
                members.append({"filename": member_path.name, "bytes": info.file_size})
    except zipfile.BadZipFile as exc:
        raise ScrapeError("The DAM response was not a valid ZIP archive.") from exc

    if len(members) != expected_count:
        raise ScrapeError(
            f"Expected {expected_count} JPG files, but the archive contained {len(members)}."
        )
    return members


def wrap_bare_jpg_as_archive(path: Path, downloaded_name: str) -> str | None:
    """The DAM's bulk download is a ZIP when two or more assets are selected
    and the bare JPG when one is, so a style whose only laydown shot is one
    file comes back as that file. Rewrite such a download in place as a ZIP
    holding the JPG under the name the DAM gave it, so the manifest and the
    downloads folder look the same however many shots a style has. Returns
    the JPG's name when one was wrapped, None when `path` already is a ZIP
    and was left alone. Anything else - an error page, say - is neither and
    is reported under the name the DAM sent it as."""
    if zipfile.is_zipfile(path):
        return None
    member_name = Path(downloaded_name).name
    with path.open("rb") as file_handle:
        head = file_handle.read(len(JPEG_MAGIC))
    if head != JPEG_MAGIC or Path(member_name).suffix.casefold() not in IMAGE_SUFFIXES:
        raise ScrapeError(
            "The DAM response was neither a ZIP archive nor a JPG "
            f"(the DAM sent it as {downloaded_name!r})."
        )
    wrapped_path = path.with_name(path.name + ".wrap")
    with zipfile.ZipFile(wrapped_path, "w", zipfile.ZIP_STORED) as archive:
        archive.write(path, arcname=member_name)
    wrapped_path.replace(path)
    return member_name


def extract_archive(path: Path, destination: Path) -> None:
    """Write every file in the archive straight into destination, flat. Folders
    inside the ZIP are not recreated; see inspect_jpg_archive."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = destination / Path(info.filename).name
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        json.dump(value, temporary_file, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def is_complete_manifest_reusable(
    manifest: dict[str, object], output_directory: Path
) -> bool:
    """A finished download stands in for a new one only when a new one would
    fetch the same thing: the same filters, its archives still on disk, and a
    selection the current cap would not enlarge - either it was made under
    this cap, or every batch already took all the shots the DAM had."""
    filters = manifest.get("filters")
    if not isinstance(filters, dict) or any(
        filters.get(title) != value for title, value in REQUIRED_FILTERS.items()
    ):
        return False
    archives = manifest.get("archives")
    archives_exist = isinstance(archives, list) and bool(archives) and all(
        isinstance(archive, dict)
        and isinstance(archive.get("filename"), str)
        and (output_directory / archive["filename"]).is_file()
        for archive in archives
    )
    if manifest.get("status") != "complete" or not archives_exist:
        return False
    policy = manifest.get("shot_request_policy")
    if not isinstance(policy, dict):
        return False
    # A text search that ran under a leftover Shot Request ID took its first
    # page from a narrowed pool. A rerun may get the filter off and take the
    # real first page, so that pull is not the same thing as a fresh one.
    if policy.get("leftover_shot_request_ids"):
        return False
    if policy.get("maximum_per_code") == MAX_PER_CODE:
        return True
    if policy.get("first_results_limit") == ITEM_DETAILS_LIMIT:
        return True
    batches = policy.get("selected_batches")
    return isinstance(batches, list) and bool(batches) and all(
        isinstance(batch, dict) and batch.get("selected") == batch.get("available")
        for batch in batches
    )


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


def read_result_total(page: Any, timeout_ms: int) -> int:
    """The total the results pane reports for the current search - "1 - 50 of
    14507", or "0 - 0 of 0" when nothing matched - polled until the pane has
    rendered it. Unlike read_search_results this needs no asset card, so it is
    the one reader that can report an empty search."""
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        total: int | None = None
        for frame in page.frames:
            try:
                body_text = frame.locator("body").inner_text(timeout=1_000)
            except PlaywrightTimeoutError:
                continue
            frame_total = parse_total_result_count(body_text)
            if frame_total is not None:
                total = frame_total if total is None else max(total, frame_total)
        if total is not None:
            return total
        if time.monotonic() >= deadline:
            raise ScrapeError(
                "The DAM results did not finish loading before the timeout."
            )
        time.sleep(0.5)


def no_laydown_assets_error(subject: str) -> NoLaydownAssetsError:
    """The search runs inside the Gap folder under the view's standing
    Shot Type = L filter, so an empty search means no laydown shot for the
    subject - which includes a style number the DAM has never heard of."""
    return NoLaydownAssetsError(f"The Gap DAM has no laydown assets for {subject}.")


def no_final_image_error(subject: str) -> NoFinalImageError:
    """apply_exclusive_facet knows only that FINAL is not on offer; this names
    the search it was not on offer for."""
    return NoFinalImageError(
        f"The Gap DAM has laydown shots for {subject}, but none of them is FINAL."
    )


def describe_batch(subject: str, shot_code: str | None) -> str:
    if shot_code is None:
        return f"The first-results batch for {subject}"
    return f"Shot Request ID {shot_code} for {subject}"


def wait_for_post(page: Any, action: Any, timeout_ms: int) -> None:
    with page.expect_response(
        lambda response: response.request.method == "POST"
        and response.url.split("?", 1)[0].endswith("/CS.aspx"),
        timeout=timeout_ms,
    ):
        action()


def open_search_page(
    context: Any,
    url: str,
    query: str,
    timeout_ms: int,
    subject: str | None = None,
    shot_request_clear_required: bool = True,
) -> Any:
    """The DAM searched for `query`, with the leftover Shot Request ID filter
    cleared and FINAL applied. `subject` is how the search is named in errors:
    "style 440760" by default, or whatever --item-details is searching for.

    A leftover Shot Request ID that will not come off ends a style search: its
    plan is built from the facet's per-code counts, and the code it then picks
    has to be applied on its own. The text search needs none of that - its
    requirements are FINAL and Shot Type L, and it takes the first page
    whatever the Shot Request ID - so with `shot_request_clear_required` False
    it goes on under the leftover filter, after saying so on stderr. What
    stayed applied is read back by the download and recorded in the manifest,
    and such a pull is not reused (see is_complete_manifest_reusable)."""
    subject = subject or f"style {query}"
    page = context.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    authenticated_page = _find_authenticated_page(context, url)
    if authenticated_page is None:
        raise SessionRejectedError("The saved DAM session was rejected.")
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
    # The previous run leaves its Shot Request ID (P01, say) checked in the
    # DAM's saved view, so a new style is first searched under that leftover
    # filter; clear it before reading anything. When the search is empty the
    # sidebar drops the Shot Request ID section altogether - before the clear
    # if nothing was checked, right after the uncheck if something was - so a
    # missing facet here is how a style with no laydown assets first shows up.
    # The result count is what tells that apart from a facet that genuinely
    # failed to render.
    try:
        clear_facet_filters(authenticated_page, SHOT_REQUEST_ID, timeout_ms)
    except FacetUnavailableError:
        if read_result_total(authenticated_page, timeout_ms) == 0:
            raise no_laydown_assets_error(subject) from None
        raise
    except ScrapeError as exc:
        if shot_request_clear_required:
            raise
        print(
            f"warning: {exc} The search for {subject} goes on under the leftover "
            "Shot Request ID filter; FINAL and Shot Type L still apply.",
            file=sys.stderr,
        )
    if read_result_total(authenticated_page, timeout_ms) == 0:
        raise no_laydown_assets_error(subject)
    # Laydown shots with none of them FINAL are the other way a search can
    # hold nothing to download; named for the subject, like the empty search.
    try:
        apply_exclusive_facet(
            authenticated_page,
            ASSET_PRODUCTION_TYPE,
            FINAL_ASSET_VALUE,
            timeout_ms,
        )
    except NoFinalImageError:
        raise no_final_image_error(subject) from None
    return authenticated_page


def find_filter_sidebar(page: Any, timeout_ms: int) -> Any:
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
    return sidebar


def find_facet_containers(page: Any, title: str, timeout_ms: int) -> Any:
    sidebar = find_filter_sidebar(page, timeout_ms)
    metrics = sidebar.evaluate(
        """element => ({
            clientHeight: element.clientHeight,
            scrollHeight: element.scrollHeight
        })"""
    )
    client_height = max(int(metrics.get("clientHeight", 0)), 1)
    max_scroll = max(int(metrics.get("scrollHeight", 0)) - client_height, 0)
    step = max(client_height * 3 // 4, 1)
    scroll_positions = list(range(0, max_scroll + 1, step))
    if scroll_positions[-1] != max_scroll:
        scroll_positions.append(max_scroll)

    heading = None
    for scroll_top in scroll_positions:
        sidebar.evaluate(
            "(element, value) => { element.scrollTop = value; }", scroll_top
        )
        page.wait_for_timeout(150)
        rendered_headings = sidebar.locator("[id$=':FacetNameLbl_Lbl']")
        for index in range(rendered_headings.count()):
            candidate = rendered_headings.nth(index)
            candidate_title = candidate.inner_text().replace(
                "\N{NO-BREAK SPACE}", " "
            ).strip()
            if candidate_title == title:
                heading = candidate
                break
        if heading is None:
            titled_headings = sidebar.locator(
                f"[original-title={json.dumps(title)}]"
            )
            if titled_headings.count() > 0:
                heading = titled_headings.first
        if heading is not None:
            break

    if heading is None:
        raise FacetUnavailableError(f"The {title} filter was not available.")
    header = heading.locator("xpath=ancestor::*[contains(@id, ':HeaderPnl')][1]")
    header_id = header.get_attribute("id") or ""
    prefix = header_id.removesuffix(":HeaderPnl")
    if not prefix:
        raise ScrapeError(f"The {title} filter could not be read.")
    containers = sidebar.locator(
        f"[id^='{prefix}:FacetContainer'][data-opn='FacetContainer']"
    )
    if containers.count() == 0:
        raise ScrapeError(f"The {title} filter could not be read.")
    return containers


def read_facet_options(
    page: Any, title: str, timeout_ms: int
) -> tuple[FacetOption, ...]:
    containers = find_facet_containers(page, title, timeout_ms)
    values = containers.evaluate_all(
        """items => items.map(item => ({
            value: item.querySelector('input[type=checkbox]')?.getAttribute('aria-label') || '',
            count: Number((item.querySelector('[data-opn=OccurrencesLbl]')?.innerText || '0').trim()),
            checked: Boolean(item.querySelector('input[type=checkbox]')?.checked)
        })).filter(item => item.value)"""
    )
    return tuple(
        FacetOption(
            value=item["value"],
            count=item["count"],
            checked=item["checked"],
        )
        for item in values
    )


# How long a facet gets to settle after a click before its state is judged.
# The DAM redraws the sidebar in more than one pass, so the read straight after
# a click can still show the old state, or no options at all, for a moment.
FACET_SETTLE_MS = 5_000
# A click the DAM drops on the floor gets this many more tries before the run
# gives up on the facet.
FACET_TOGGLE_ATTEMPTS = 3


def find_facet_checkbox(containers: Any, value: str) -> Any | None:
    """The checkbox for `value` among a facet's containers, or None when the
    facet does not list the value."""
    candidates = containers.locator(
        f"input[type='checkbox'][aria-label={json.dumps(value)}]"
    )
    if candidates.count() == 0:
        return None
    return candidates.first


def toggle_facet_checkbox(
    page: Any, title: str, value: str, timeout_ms: int
) -> None:
    """Click the facet's checkbox for `value` and wait for the sidebar to show
    it the other way. The DAM drops a click now and then - one run's leftover
    P01 stayed checked for the whole timeout, and came off in three seconds on
    the next - so a click that has not shown after FACET_SETTLE_MS is made
    again, FACET_TOGGLE_ATTEMPTS times in all; "did not update" is raised only
    after the last. A value the facet stops listing while it is being unchecked
    is off the filter, which is what the uncheck was for."""
    containers = find_facet_containers(page, title, timeout_ms)
    checkbox = find_facet_checkbox(containers, value)
    if checkbox is None:
        raise ScrapeError(f"The {title} value {value} could not be read.")
    expected_checked = not checkbox.is_checked()
    deadline = time.monotonic() + timeout_ms / 1000
    for attempt in range(FACET_TOGGLE_ATTEMPTS):
        if attempt:
            # The sidebar has been redrawn since the last click; find the
            # checkbox afresh rather than clicking a stale one.
            containers = find_facet_containers(page, title, timeout_ms)
            checkbox = find_facet_checkbox(containers, value)
            if checkbox is None:
                if not expected_checked:
                    return
                raise ScrapeError(f"The {title} value {value} could not be read.")
        wait_for_post(
            page,
            lambda checkbox=checkbox: checkbox.evaluate("element => element.click()"),
            timeout_ms,
        )
        page.wait_for_timeout(1_500)
        settle_deadline = min(deadline, time.monotonic() + FACET_SETTLE_MS / 1000)
        while True:
            options = read_facet_options(page, title, timeout_ms)
            checked = next(
                (option.checked for option in options if option.value == value),
                False,
            )
            if checked == expected_checked:
                return
            if time.monotonic() >= settle_deadline:
                break
            page.wait_for_timeout(500)
    raise ScrapeError(f"{title} {value} did not update.")


def clear_facet_filters(page: Any, title: str, timeout_ms: int) -> tuple[FacetOption, ...]:
    options = read_facet_options(page, title, timeout_ms)
    for option in options:
        if option.checked:
            toggle_facet_checkbox(page, title, option.value, timeout_ms)
    return read_facet_options(page, title, timeout_ms)


def checked_facet_values(page: Any, title: str, timeout_ms: int) -> tuple[str, ...]:
    """The facet's checked values in sidebar order; () when the sidebar does
    not show the facet, which is what it does with nothing to offer."""
    try:
        options = read_facet_options(page, title, timeout_ms)
    except FacetUnavailableError:
        return ()
    return tuple(option.value for option in options if option.checked)


def is_exclusive_selection(options: Iterable[FacetOption], value: str) -> bool:
    options = tuple(options)
    return any(option.value == value and option.checked for option in options) and not any(
        option.value != value and option.checked for option in options
    )


def apply_exclusive_facet(
    page: Any, title: str, value: str, timeout_ms: int
) -> tuple[FacetOption, ...]:
    """Leave `value` as the only checked option of the facet. Each round
    unchecks the others, checks the value, then waits up to FACET_SETTLE_MS
    for the sidebar to show exactly that; a round whose click did not stick is
    repeated, FACET_TOGGLE_ATTEMPTS times in all."""
    try:
        options = read_facet_options(page, title, timeout_ms)
    except FacetUnavailableError:
        if title == ASSET_PRODUCTION_TYPE and value == FINAL_ASSET_VALUE:
            raise NoFinalImageError("No FINAL image is available for this style.") from None
        raise
    target = next((option for option in options if option.value == value), None)
    if target is not None and target.count > 0 and is_exclusive_selection(options, value):
        return options

    deadline = time.monotonic() + timeout_ms / 1000
    for _ in range(FACET_TOGGLE_ATTEMPTS):
        for option in options:
            if option.checked and option.value != value:
                toggle_facet_checkbox(page, title, option.value, timeout_ms)

        options = read_facet_options(page, title, timeout_ms)
        target = next((option for option in options if option.value == value), None)
        if target is None or target.count <= 0:
            if title == ASSET_PRODUCTION_TYPE and value == FINAL_ASSET_VALUE:
                raise NoFinalImageError("No FINAL image is available for this style.")
            raise ScrapeError(f"{title} {value} is not available for this style.")
        if not target.checked:
            toggle_facet_checkbox(page, title, value, timeout_ms)

        settle_deadline = min(deadline, time.monotonic() + FACET_SETTLE_MS / 1000)
        while True:
            options = read_facet_options(page, title, timeout_ms)
            if is_exclusive_selection(options, value):
                return options
            if time.monotonic() >= settle_deadline:
                break
            page.wait_for_timeout(500)
    raise ScrapeError(f"{title} {value} did not update.")


def shot_request_counts(options: Iterable[FacetOption]) -> dict[str, int]:
    return {option.value: option.count for option in options}


def dismiss_popups(page: Any) -> None:
    """Close whatever the DAM has floated over the results. The funnel button,
    for one, opens an "Applied Filters" popup when the filter aside is pinned
    open, and it lands exactly over the first asset card's checkbox - every
    click on that card is then intercepted until the popup goes away. The
    layer's direct children are zero-height wrappers, and the popup's content
    arrives about a second after the button is pressed, so what is checked is
    any visible descendant, not the wrappers."""
    for _ in range(3):
        if not any(
            frame.locator("#PopupLayer *:visible").count() for frame in page.frames
        ):
            return
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)


def set_shot_request_filter(page: Any, code: str, timeout_ms: int) -> None:
    apply_exclusive_facet(page, SHOT_REQUEST_ID, code, timeout_ms)

    # The funnel folds the filter aside when the DAM shows it as a drawer.
    # When the aside is pinned open (a per-user setting the DAM remembers),
    # the same button opens the "Applied Filters" popup instead. Either way,
    # nothing may be left floating over the results before cards are clicked.
    sidebar = find_visible(
        page, ("aside[aria-label='Photo Studio Filters']",), timeout_ms
    )
    if sidebar.is_visible():
        find_visible(page, ("a[aria-label='Filter']",), timeout_ms).click()
        page.wait_for_timeout(1_500)
    dismiss_popups(page)


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

    dismiss_popups(page)
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
    *,
    query: str,
    subject: str,
    auth_state: Path,
    url: str,
    headed: bool,
    timeout_ms: int,
) -> dict[str, int]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed)
        context = browser.new_context(storage_state=auth_state)
        page = open_search_page(context, url, query, timeout_ms, subject=subject)
        counts = shot_request_counts(
            clear_facet_filters(page, SHOT_REQUEST_ID, timeout_ms)
        )
        browser.close()
        return counts


def batch_limit(subject: str, shot_code: str | None, planned: int, total: int) -> int:
    """How many cards the download takes. A style batch was planned from the
    facet's own count for its code, so a page with fewer results than planned
    is an error. The text search is planned as "the first page", so it takes
    what the page holds, up to that."""
    if total >= planned:
        return planned
    if shot_code is not None:
        raise ScrapeError(
            f"{describe_batch(subject, shot_code)} returned {total} "
            f"assets; the selection plan expected {planned}."
        )
    return total


# A bulk download the browser reports as canceled - the DAM, or the link in
# between, dropped the transfer - is started again from the same selection,
# this many times in all.
DOWNLOAD_ATTEMPTS = 3


def is_canceled_download(exc: BaseException) -> bool:
    """Playwright's "Download.save_as: canceled": the transfer was cut off,
    which is how a 238 MB ZIP ended after six minutes on a slow link when the
    same ZIP had taken two on a good one."""
    return isinstance(exc, PlaywrightError) and "canceled" in str(exc)


def save_bulk_download(
    page: Any,
    selected_filenames: tuple[str, ...],
    *,
    subject: str,
    shot_code: str | None,
    destination: Path,
    timeout_ms: int,
) -> Any:
    """Open the DAM's download dialog for the selected cards, choose JPG
    Original Resolution and save its Standard download to `destination`.
    Returns the finished download. A transfer the browser reports canceled is
    started again from the same selection, DOWNLOAD_ATTEMPTS times in all; the
    dialog is opened afresh each time, so the selection is re-counted too."""
    last_error: BaseException | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        if attempt:
            print(
                f"The download for {describe_batch(subject, shot_code)} was "
                f"canceled in transfer; starting it again "
                f"({attempt + 1} of {DOWNLOAD_ATTEMPTS}).",
                file=sys.stderr,
            )
            dismiss_popups(page)
        first_asset = find_visible(
            page,
            (f"[role='region'][aria-label='Gap Image: {selected_filenames[0]}']",),
            timeout_ms,
        )
        first_asset.locator("xpath=ancestor::*[@data-pv][1]").hover()
        bulk_download = find_visible(
            page,
            (
                "button[aria-label='Download']",
                "a[aria-label='Download']",
                "button:has-text('Download')",
            ),
            timeout_ms,
        )
        bulk_download.click()

        original = find_visible(page, ("input[aria-label='Original']",), timeout_ms)
        jpg_original = find_visible(
            page, ("input[aria-label='JPG Original Resolution']",), timeout_ms
        )
        download_dialog = original.locator(
            "xpath=ancestor::*[@data-vf='DownloadMultipleFormatSelector_VForm']"
        )
        selected_count = parse_selected_asset_count(download_dialog.inner_text())
        if selected_count != len(selected_filenames):
            raise ScrapeError(
                f"Selected {len(selected_filenames)} assets for "
                f"{describe_batch(subject, shot_code)}, but the download dialog "
                f"contained {selected_count or 0}."
            )
        if original.is_checked():
            wait_for_post(page, original.uncheck, timeout_ms)
        if not jpg_original.is_checked():
            wait_for_post(page, jpg_original.check, timeout_ms)

        standard_download = find_visible(
            page, ("a[aria-label='Standard download']",), timeout_ms
        ).locator("xpath=parent::*")
        with page.expect_download(timeout=timeout_ms) as download_info:
            standard_download.click()
        download = download_info.value
        try:
            download.save_as(destination)
        except PlaywrightError as exc:
            if not is_canceled_download(exc):
                raise
            last_error = exc
            continue
        return download
    raise ScrapeError(
        f"The download for {describe_batch(subject, shot_code)} was canceled in "
        f"transfer {DOWNLOAD_ATTEMPTS} times ({last_error})."
    )


def download_batch_from_dam(
    *,
    query: str,
    subject: str,
    shot_code: str | None,
    limit: int,
    auth_state: Path,
    url: str,
    destination: Path,
    headed: bool,
    timeout_ms: int,
) -> BatchDownload:
    """One download of up to `limit` JPGs - a ZIP, or the bare JPG when one
    card is taken (see wrap_bare_jpg_as_archive): the first cards the search
    shows, under Shot Request ID `shot_code` when one is given and in the DAM's
    own order when it is None (the --item-details mode, which also settles for
    a leftover Shot Request ID it could not clear - see open_search_page)."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed)
        context = browser.new_context(storage_state=auth_state, accept_downloads=True)
        authenticated_page = open_search_page(
            context,
            url,
            query,
            timeout_ms,
            subject=subject,
            shot_request_clear_required=shot_code is not None,
        )
        if shot_code is not None:
            set_shot_request_filter(authenticated_page, shot_code, timeout_ms)
            shot_request_ids: tuple[str, ...] = (shot_code,)
        else:
            # Read before the cards are touched: the facet read may unfold the
            # filter aside, which set_shot_request_filter folds away on purpose.
            shot_request_ids = checked_facet_values(
                authenticated_page, SHOT_REQUEST_ID, timeout_ms
            )
        authenticated_page.wait_for_timeout(1_000)
        results = read_search_results(authenticated_page, timeout_ms)
        limit = batch_limit(subject, shot_code, limit, results.total)
        selected_filenames = select_asset_limit(authenticated_page, limit, timeout_ms)
        download = save_bulk_download(
            authenticated_page,
            selected_filenames,
            subject=subject,
            shot_code=shot_code,
            destination=destination,
            timeout_ms=timeout_ms,
        )
        browser.close()
        return BatchDownload(
            selected_filenames=tuple(selected_filenames),
            suggested_filename=download.suggested_filename,
            total=results.total,
            shot_request_ids=shot_request_ids,
        )


@dataclass(frozen=True)
class RunSettings:
    """Everything a download needs besides what it is searching for."""

    auth_state: Path
    url: str
    headed: bool
    timeout_ms: int
    force: bool
    output_root: Path
    images_directory: Path


def reuse_complete_download(settings: RunSettings, manifest_path: Path) -> bool:
    """Serve a finished download from disk when a fresh one would fetch the
    same thing (see is_complete_manifest_reusable), putting back any JPG the
    library has since lost. True means the caller is done."""
    output_directory = manifest_path.parent
    if settings.force or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not is_complete_manifest_reusable(manifest, output_directory):
        return False
    missing = [
        entry["filename"]
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("filename"), str)
        and not (settings.images_directory / entry["filename"]).is_file()
    ]
    if missing:
        for archive in manifest["archives"]:
            extract_archive(
                output_directory / archive["filename"], settings.images_directory
            )
        print(f"Restored {len(missing)} JPG files to {settings.images_directory}")
    else:
        print(f"Already complete: {settings.images_directory}")
    return True


def fetch_batch(
    settings: RunSettings,
    *,
    query: str,
    subject: str,
    shot_code: str | None,
    limit: int,
    archive_path: Path,
) -> tuple[
    dict[str, object], list[dict[str, object]], list[dict[str, object]], BatchDownload
]:
    """Download one batch to `archive_path` and unpack it into the library.
    Returns the manifest's archive record, its file records, the DAM-side
    source filenames that were selected, and the download itself."""
    partial_path = archive_path.with_suffix(".zip.part")
    download = download_batch_from_dam(
        query=query,
        subject=subject,
        shot_code=shot_code,
        limit=limit,
        auth_state=settings.auth_state,
        url=settings.url,
        destination=partial_path,
        headed=settings.headed,
        timeout_ms=settings.timeout_ms,
    )
    wrapped_name = wrap_bare_jpg_as_archive(partial_path, download.suggested_filename)
    members = inspect_jpg_archive(partial_path, len(download.selected_filenames))
    partial_path.replace(archive_path)
    extract_archive(archive_path, settings.images_directory)
    archive: dict[str, object] = {
        "shot_request_id": shot_code,
        "filename": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }
    if wrapped_name is not None:
        # This ZIP is the scraper's, not the DAM's: the DAM sent the one JPG
        # bare, and this records which file that was.
        archive["wrapped_bare_file"] = wrapped_name
    files = [{**member, "shot_request_id": shot_code} for member in members]
    sources = [
        {"filename": filename, "shot_request_id": shot_code}
        for filename in download.selected_filenames
    ]
    return archive, files, sources, download


def download_style(settings: RunSettings, style_number: str) -> Path:
    """The style's laydown shots under the Shot Request ID policy - up to
    MAX_PER_CODE of P01, else of AV5, else of every code. Returns the manifest."""
    query = normalize_style_number(style_number)
    query_directory = safe_query_directory(query)
    output_directory = settings.output_root / query_directory
    manifest_path = output_directory / "manifest.json"
    if reuse_complete_download(settings, manifest_path):
        return manifest_path

    output_directory.mkdir(parents=True, exist_ok=True)
    subject = f"style {query}"
    shot_counts = discover_shot_counts(
        query=query,
        subject=subject,
        auth_state=settings.auth_state,
        url=settings.url,
        headed=settings.headed,
        timeout_ms=settings.timeout_ms,
    )
    plan = choose_shot_batches(shot_counts)
    archives: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    selected_sources: list[dict[str, object]] = []
    for batch in plan:
        archive, batch_files, sources, _ = fetch_batch(
            settings,
            query=query,
            subject=subject,
            shot_code=batch.code,
            limit=batch.limit,
            archive_path=output_directory
            / f"gap-{query_directory}-{safe_query_directory(batch.code)}-jpg-original.zip",
        )
        archives.append(archive)
        files.extend(batch_files)
        selected_sources.extend(sources)

    manifest = {
        "status": "complete",
        "input_style_number": style_number,
        "query": query,
        "brand_scope": "Gap",
        "filters": REQUIRED_FILTERS,
        "shot_request_policy": {
            "preferred_order": ["P01", "AV5"],
            "maximum_per_code": MAX_PER_CODE,
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
        "image_root": str(settings.images_directory),
        "archives": archives,
        "files": files,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(manifest_path, manifest)
    print(
        f"Downloaded {sum(batch.limit for batch in plan)} JPG files "
        f"to {settings.images_directory}"
    )
    return manifest_path


def download_item_details(settings: RunSettings, details: str) -> Path:
    """The first ITEM_DETAILS_LIMIT laydown shots the DAM returns for free
    text, in its own order and whatever their Shot Request ID. Returns the
    manifest."""
    query = details.strip()
    if not query:
        raise ScrapeError("--item-details needs some text to search with.")
    query_directory = safe_query_directory(query)
    output_directory = settings.output_root / ITEM_DETAILS_DIRECTORY / query_directory
    manifest_path = output_directory / "manifest.json"
    if reuse_complete_download(settings, manifest_path):
        return manifest_path

    output_directory.mkdir(parents=True, exist_ok=True)
    subject = f"the search {query!r}"
    archive, files, selected_sources, download = fetch_batch(
        settings,
        query=query,
        subject=subject,
        shot_code=None,
        limit=ITEM_DETAILS_LIMIT,
        archive_path=output_directory
        / f"gap-{query_directory}-first-results-jpg-original.zip",
    )
    total = download.total
    taken = len(selected_sources)

    manifest = {
        "status": "complete",
        "input_style_number": None,
        "item_details": details,
        "query": query,
        "brand_scope": "Gap",
        "filters": REQUIRED_FILTERS,
        "shot_request_policy": {
            "mode": "first_results",
            "first_results_limit": ITEM_DETAILS_LIMIT,
            # The Shot Request ID values the search ran under. [] is the
            # intended state; anything else is a leftover from an earlier run
            # that would not come off, and the pull is not reused.
            "leftover_shot_request_ids": list(download.shot_request_ids),
            "selected_batches": [
                {"shot_request_id": None, "available": total, "selected": taken}
            ],
        },
        "format": "JPG Original Resolution",
        "download_method": "Standard download",
        "result_count": taken,
        "selected_sources": selected_sources,
        "image_root": str(settings.images_directory),
        "archives": [archive],
        "files": files,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(manifest_path, manifest)
    print(
        f"Downloaded {taken} of {total} JPG files for {subject} "
        f"to {settings.images_directory}"
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search FINAL laydown assets in the current Gap DAM view and download "
            f"them as JPGs: by style number, up to {MAX_PER_CODE} per selected Shot "
            f"Request ID; or by --item-details text, the first {ITEM_DETAILS_LIMIT} "
            "results. Given both, the text is the fallback for a style the DAM has "
            "no FINAL laydown assets for."
        )
    )
    parser.add_argument(
        "style_number",
        nargs="?",
        help="A 6- or 9-digit style number; 853417012 is searched as 853417.",
    )
    parser.add_argument(
        "--item-details",
        metavar="TEXT",
        help=(
            f"Free text to search the DAM with; the first {ITEM_DETAILS_LIMIT} "
            "laydown results are taken, whatever their Shot Request ID. On its own "
            "it is the whole job; next to a style number it is used only when the "
            "style has no laydown assets, or none of them FINAL."
        ),
    )
    parser.add_argument("--auth-state", type=Path, default=DEFAULT_AUTH_STATE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("DAM_OUTPUT_ROOT", "downloads")),
        help=(
            "Where the source ZIP and manifest are kept: one folder per style, "
            f"and one per text under {ITEM_DETAILS_DIRECTORY}/."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(os.environ.get("DAM_IMAGE_ROOT", DEFAULT_IMAGE_ROOT)),
        help=(
            "Where the extracted JPGs are written, flat, with no per-style folder; "
            "defaults to the reference library."
        ),
    )
    parser.add_argument("--url", default=DEFAULT_SEARCH_URL)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    return parser


def download(settings: RunSettings, args: argparse.Namespace) -> Path:
    """The manifest of this pull: the style's, else the text's - the text
    stands in when there is no style, or when the style has no laydown
    assets (none at all, or none of them FINAL)."""
    if args.style_number is not None:
        try:
            return download_style(settings, args.style_number)
        except NoLaydownAssetsError as exc:
            if args.item_details is None:
                raise
            print(
                f"{exc} Falling back to --item-details {args.item_details!r}.",
                file=sys.stderr,
            )
    return download_item_details(settings, args.item_details)


def download_with_session(settings: RunSettings, args: argparse.Namespace) -> Path:
    """download(), signing in first when there is no saved session or the DAM
    sends the saved one back to login.

    Only a login that needs no terminal is used (dam_auth.stored_credentials:
    the environment, else the macOS Keychain), so a container run never hangs
    on a prompt; with none stored the error names both ways to provide one.
    One sign-in, then one more try: a fresh session the DAM rejects again is
    an error, not a loop.
    """
    try:
        load_storage_state(settings.auth_state)
        return download(settings, args)
    except (AuthStateError, SessionRejectedError) as exc:
        stored = stored_credentials()
        if stored is None:
            raise ScrapeError(
                f"{exc} No stored login to sign in with: run dam_auth.py capture, "
                f"or set {LOGIN_ID_ENV} and {PASSWORD_ENV}."
            ) from exc
        credentials, source = stored
        print(f"{exc} Signing in with the login from {source}.", file=sys.stderr)
    sign_in(
        settings.auth_state,
        credentials=credentials,
        timeout_ms=settings.timeout_ms,
        headed=settings.headed,
    )
    return download(settings, args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.style_number is None and args.item_details is None:
        parser.error("a style number, --item-details TEXT, or both is required")
    try:
        settings = RunSettings(
            auth_state=args.auth_state.expanduser().resolve(),
            url=args.url,
            headed=args.headed,
            timeout_ms=args.timeout_ms,
            force=args.force,
            output_root=args.output_root.expanduser().resolve(),
            images_directory=args.image_root.expanduser().resolve(),
        )
        manifest_path = download_with_session(settings, args)
        # The last line of stdout names the manifest, so a caller that cannot
        # know in advance which of the two searches produced it can find it.
        print(f"manifest {manifest_path}")
        return 0
    except NoLaydownAssetsError as exc:
        print(f"DAM has nothing to download: {exc}", file=sys.stderr)
        return EXIT_NOTHING_TO_DOWNLOAD
    except (
        AuthStateError,
        AuthenticationExpired,
        RuntimeError,  # ScrapeError and dam_auth's configuration errors
        PlaywrightError,  # its TimeoutError included; "Download.save_as: canceled" was one
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"DAM download failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
