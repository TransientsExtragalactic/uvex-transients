"""
Pin this repository to the latest uvex-scheduler release.

The package config (``uvex_transients/config.yaml``) registers every released schedule by name
(``uvex_v0.5.0``, ...), and the release run (``configs/full_run.yaml``) and the package defaults
point at the newest one. This script looks up the latest release of ``m4opt/uvex-scheduler`` and
makes the following edits, all in place and all idempotent:

- registers ``uvex_v<latest>`` in ``schedules.schedule_urls`` (older entries are left alone, so
  they remain available as references to the previous schedules),
- points ``schedules.default_schedule`` at it,
- points ``observatories.uvex.repo_url`` (the survey-footprint files) at the same tag,
- points ``schedule.name`` in ``configs/full_run.yaml`` at it.

Edits are plain text substitutions rather than a YAML round trip, so the diff is exactly the
lines above and no comments or formatting are disturbed. If the repository is already pinned to
the latest release, nothing changes.

Usage::

    python scripts/sync_scheduler.py            # look up the latest release on GitHub
    python scripts/sync_scheduler.py --tag v0.6.0

When run inside GitHub Actions, ``tag`` and ``changed`` are also written to ``$GITHUB_OUTPUT``.
"""

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

UPSTREAM = "m4opt/uvex-scheduler"
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_CONFIG = ROOT / "uvex_transients" / "config.yaml"
RUN_CONFIG = ROOT / "configs" / "full_run.yaml"

_RAW = f"https://raw.githubusercontent.com/{UPSTREAM}/refs/tags"
_TAG = re.compile(r"v\d+\.\d+\.\d+")


def latest_release_tag() -> str:
    """
    Look up the tag of the latest ``m4opt/uvex-scheduler`` GitHub release.

    Returns
    -------
    str
        The release tag, e.g. ``"v0.5.0"``.
    """
    request = urllib.request.Request(
        f"https://api.github.com/repos/{UPSTREAM}/releases/latest",
        headers={"Accept": "application/vnd.github+json"},
    )
    if token := os.environ.get("GITHUB_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["tag_name"]


def _sub_once(pattern: str, replacement: str, text: str, what: str) -> str:
    """Apply a multiline regex substitution that must match exactly once."""
    new, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match for {what}, found {count}.")
    return new


def pin_package_config(text: str, tag: str) -> str:
    """
    Pin the package config text (``uvex_transients/config.yaml``) to the release `tag`.

    Parameters
    ----------
    text : str
        The current contents of the package config.
    tag : str
        The release tag, e.g. ``"v0.6.0"``.

    Returns
    -------
    str
        The updated contents. Identical to `text` if it is already pinned to `tag`.
    """
    name = f"uvex_{tag}"
    entry = f"    {name}: {_RAW}/{tag}/tables/plan.ecsv"
    if not re.search(rf"^    {re.escape(name)}:", text, flags=re.MULTILINE):
        text = _sub_once(r"^(    uvex_dev: .*)$", rf"\1\n{entry}", text, "the 'uvex_dev' schedule entry")
    text = _sub_once(r"^(  default_schedule: ).*$", rf"\g<1>{name}", text, "'default_schedule'")
    return _sub_once(r"^(    repo_url: ).*$", rf"\g<1>{_RAW}/{tag}", text, "the uvex 'repo_url'")


def pin_run_config(text: str, tag: str) -> str:
    """
    Pin the run-config text (``configs/full_run.yaml``) to the release `tag`.

    Parameters
    ----------
    text : str
        The current contents of the run-config.
    tag : str
        The release tag, e.g. ``"v0.6.0"``.

    Returns
    -------
    str
        The updated contents. Identical to `text` if it is already pinned to `tag`.
    """
    return _sub_once(r"^(schedule:\n  name: )\S+", rf"\g<1>uvex_{tag}", text, "'schedule.name'")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", help="Pin to this release tag instead of looking up the latest one.")
    args = parser.parse_args()

    tag = args.tag or latest_release_tag()
    if not _TAG.fullmatch(tag):
        raise SystemExit(f"Refusing to pin to {tag!r}: expected a tag like v1.2.3.")

    changed = False
    for path, pin in ((PACKAGE_CONFIG, pin_package_config), (RUN_CONFIG, pin_run_config)):
        old = path.read_text()
        new = pin(old, tag)
        if new != old:
            path.write_text(new)
            changed = True
            print(f"updated {path.relative_to(ROOT)}")

    print(f"uvex-scheduler {tag}: {'pinned' if changed else 'already up to date'}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as f:
            f.write(f"tag={tag}\nchanged={str(changed).lower()}\n")


if __name__ == "__main__":
    main()
