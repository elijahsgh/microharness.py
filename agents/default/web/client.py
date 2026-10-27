"""User-extensible httpx2 web interaction hooks for Microharness WebFetch (`agents/default/web/client.py`).

Customize this file to inject Bearer tokens, API keys, custom headers, cookies,
or custom `httpx2.AsyncClient` settings per domain. Changes are hot-reloaded on every run.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx2
from pydantic_ai._http import create_async_httpx2_client

DEFAULT_USER_AGENT = "Microharness-WebFetch/1.0 (+httpx2)"


def prepare_headers(url: str, headers: dict[str, str]) -> dict[str, str]:
    """Customize or inject authentication headers before `httpx2` sends a request.

    Example: automatically attach a GitHub token or internal API key for specific hosts.
    """
    merged = {"User-Agent": DEFAULT_USER_AGENT}
    merged.update(headers)
    host = (urlparse(url).hostname or "").lower()

    if host in ("api.github.com", "github.com") and (
        gh_token := os.environ.get("GITHUB_TOKEN")
    ):
        merged.setdefault("Authorization", f"Bearer {gh_token}")

    return merged


def create_client(*, timeout: float = 30.0) -> httpx2.AsyncClient:
    """Create the `httpx2.AsyncClient` used for native URL fetching.

    Override this to configure custom proxies, TLS certificates, or connection limits.
    """
    return create_async_httpx2_client(timeout=timeout)


def transform_response(url: str, result: dict[str, Any]) -> dict[str, Any]:
    """Optional post-processing hook called on the fetched result dict before returning to the agent."""
    return result
