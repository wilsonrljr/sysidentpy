"""MkDocs build hooks for SysIdentPy documentation."""

from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict

import requests

_PEPY_URL = "https://api.pepy.tech/api/v2/projects/sysidentpy"


def _format_compact(value: float | None) -> str:
    """Return a short human-readable representation for download counts."""
    if value is None or value < 0:
        return "56K+"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M+"
    if value >= 1_000:
        return f"{value / 1_000:.1f}".rstrip("0").rstrip(".") + "K+"
    return f"{int(value)}"


def _fetch_pepy_stats(api_key: str | None) -> Dict[str, Any]:
    """Fetch download statistics from the PePy API."""
    stats: Dict[str, Any] = {
        "total_downloads": None,
        "downloads_month": None,
        "last_updated": None,
        "latest_version": None,
    }

    if not api_key:
        logging.info("PEPY_API_KEY not provided; skipping PePy stats fetch.")
        return stats

    headers = {"X-API-Key": api_key}

    try:
        response = requests.get(_PEPY_URL, headers=headers, timeout=10)
        if not response.ok:
            logging.warning(
                "PePy API responded with status %s: %s",
                response.status_code,
                response.text,
            )
            return stats
        data = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network error
        logging.warning("Unable to fetch PePy stats: %s", exc)
        return stats

    stats["total_downloads"] = data.get("total_downloads")
    downloads = data.get("downloads") or {}
    stats["downloads_month"] = (
        downloads.get("last_month")
        or downloads.get("last_week")
        or downloads.get("last_day")
    )
    versions = data.get("versions") or []
    if isinstance(versions, list) and versions:
        stats["latest_version"] = str(versions[-1])
    stats["last_updated"] = _dt.datetime.utcnow().isoformat() + "Z"
    return stats


def on_config(config):
    """MkDocs hook executed once the configuration is loaded."""
    api_key = os.getenv("PEPY_API_KEY", "").strip()
    pepy_stats = _fetch_pepy_stats(api_key)
    pepy_stats["formatted_total"] = _format_compact(pepy_stats["total_downloads"])

    config.extra.setdefault("pepy_stats", pepy_stats)
    latest_version = pepy_stats.get("latest_version")
    if latest_version:
        config.extra["pypi_version"] = latest_version
    return config
