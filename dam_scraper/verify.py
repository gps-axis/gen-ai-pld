"""Run the repeatable local checks for the isolated DAM auth component."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*arguments: str) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def verify_playwright_versions_match() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    requirement = next(
        (dependency for dependency in dependencies if dependency.startswith("playwright==")),
        None,
    )
    if requirement is None:
        raise RuntimeError("pyproject.toml must pin one Playwright version.")

    version = requirement.removeprefix("playwright==")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    expected_image = f"mcr.microsoft.com/playwright/python:v{version}-noble"
    if expected_image not in dockerfile:
        raise RuntimeError(
            f"Docker image and Python package differ; expected {expected_image}."
        )


def main() -> int:
    verify_playwright_versions_match()
    run("uv", "lock", "--check")
    run(
        sys.executable,
        "-m",
        "compileall",
        "-q",
        "auth_session.py",
        "dam_auth.py",
        "dam_scrape.py",
    )
    run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")
    run(sys.executable, "dam_auth.py", "--help")
    run(sys.executable, "dam_scrape.py", "--help")
    print("DAM scraper component verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
