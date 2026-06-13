"""Security helpers for ref_audio / speaker_audio / prompt_audio path resolution.

Prevents:
  - Path traversal (any ``..`` segment, absolute paths outside allowed_dir, symlinks that escape)
  - SSRF (HTTP(S) URL fetches that resolve to private/loopback/link-local IPs)

Only stdlib is used: pathlib, urllib.parse, socket, ipaddress.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("realtime-server")

# Default allowed directory for local ref audio files, resolved from cwd.
_DEFAULT_REF_AUDIO_DIR = Path("./ref_audio").resolve()


@dataclass
class RefAudioConfig:
    """Configuration for safe ref-audio resolution."""

    allowed_dir: Path = field(default_factory=lambda: _DEFAULT_REF_AUDIO_DIR)
    allowed_hosts: list[str] = field(default_factory=list)
    max_bytes: int = 50 * 1024 * 1024  # 50 MiB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_ip_private_or_reserved(ip_str: str) -> bool:
    """Return True if *ip_str* is private, loopback, link-local, reserved, or multicast."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        # Not a valid IP — treat as unsafe
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )


def _hostname_matches_allowlist(hostname: str, allowed_hosts: list[str]) -> bool:
    """Case-insensitive exact or suffix match against *allowed_hosts*.

    Suffix match: if an allowlist entry starts with ``.``, it matches any
    subdomain.  E.g. ``.example.com`` matches ``foo.example.com``.
    """
    hn = hostname.lower()
    for entry in allowed_hosts:
        e = entry.lower()
        if e == hn:
            return True
        # suffix match: ".example.com" matches "foo.example.com"
        if e.startswith(".") and hn.endswith(e):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_local_ref(ref: str, cfg: RefAudioConfig) -> Path:
    """Resolve a local file path for ref audio, enforcing path-traversal guards.

    Args:
        ref: A relative path string (no scheme).
        cfg: Safety configuration.

    Returns:
        Resolved absolute ``Path`` inside *cfg.allowed_dir*.

    Raises:
        ValueError: If the path attempts traversal, is absolute and outside
            the allowed dir, or is a symlink that escapes.
    """
    # 1. Reject any ".." segment
    parts = Path(ref).parts
    if ".." in parts:
        raise ValueError(
            f"ref_audio path traversal rejected (contains '..'): {ref}"
        )

    # 2. Build candidate path
    allowed_dir = cfg.allowed_dir.resolve()
    candidate = (allowed_dir / ref).resolve(strict=False)

    # 3. If the file exists, resolve symlinks and check again
    if candidate.exists():
        try:
            real = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"ref_audio symlink resolution failed: {ref} — {exc}"
            ) from exc
        if not real.is_relative_to(allowed_dir):
            raise ValueError(
                f"ref_audio symlink escape rejected: {ref} resolves to {real} "
                f"which is outside {allowed_dir}"
            )
        return real

    # 4. For non-existent paths, still ensure the logical path stays inside
    if not candidate.is_relative_to(allowed_dir):
        raise ValueError(
            f"ref_audio path escape rejected: {ref} resolves to {candidate} "
            f"which is outside {allowed_dir}"
        )
    return candidate


def validate_remote_url(url: str, cfg: RefAudioConfig) -> str:
    """Validate an HTTP(S) URL against SSRF and allowlist rules.

    Args:
        url: A URL string with ``http://`` or ``https://`` scheme.
        cfg: Safety configuration.

    Returns:
        The validated URL string (unchanged).

    Raises:
        ValueError: If the scheme is not http/https, hostname is not
            allowlisted, or any resolved IP is private/loopback/link-local.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"ref_audio URL scheme rejected (only http/https allowed): {url}"
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError(f"ref_audio URL has no hostname: {url}")

    # 1. Hostname allowlist check
    if not _hostname_matches_allowlist(hostname, cfg.allowed_hosts):
        raise ValueError(
            f"ref_audio URL hostname not in allowlist: {hostname} "
            f"(allowed: {cfg.allowed_hosts})"
        )

    # 2. DNS resolution + private-IP rejection
    try:
        addr_infos = socket.getaddrinfo(hostname, parsed.port or 80)
    except socket.gaierror as exc:
        raise ValueError(
            f"ref_audio URL hostname resolution failed: {hostname} — {exc}"
        ) from exc

    for _family, _type, _proto, _canon, sockaddr in addr_infos:
        ip_str: str = sockaddr[0]
        if _is_ip_private_or_reserved(ip_str):
            raise ValueError(
                f"ref_audio URL rejected (resolved IP {ip_str} is "
                f"private/loopback/link-local/reserved): {url}"
            )

    return url


def read_ref_audio_safely(ref: str | None, cfg: RefAudioConfig) -> bytes:
    """Read ref audio bytes safely, dispatching to local or remote path.

    For local paths: validates via *resolve_local_ref* and reads the file.
    For HTTP(S) URLs: validates via *validate_remote_url* and fetches.
    For data: URLs: returns the payload as-is (already inline).

    Args:
        ref: A file path, HTTP(S) URL, or ``data:`` URL.  ``None`` raises ValueError.
        cfg: Safety configuration.

    Returns:
        Raw audio bytes.

    Raises:
        ValueError: On any safety violation or if the file is too large.
        FileNotFoundError: If a local file does not exist.
    """
    if not ref:
        raise ValueError("ref_audio is empty or None")

    # data: URLs are inline — no I/O risk
    if ref.startswith("data:"):
        # Extract payload after base64, (we return the full data URL bytes
        # since the caller will base64-encode again for the API).
        # Actually, for data URLs we should just return the full string —
        # but the contract says return bytes.  The caller handles encoding.
        # Return the raw data URL as bytes for passthrough.
        return ref.encode("utf-8")

    # HTTP(S) URL
    if "://" in ref and ref.startswith(("http://", "https://")):
        validate_remote_url(ref, cfg)
        # Fetch with size guard using only stdlib
        return _fetch_url(ref, cfg)

    # Local file path
    resolved = resolve_local_ref(ref, cfg)
    if not resolved.exists():
        raise FileNotFoundError(f"ref_audio file not found: {resolved}")

    data = resolved.read_bytes()
    if len(data) > cfg.max_bytes:
        raise ValueError(
            f"ref_audio file too large: {len(data)} bytes "
            f"(max {cfg.max_bytes})"
        )
    return data


def _fetch_url(url: str, cfg: RefAudioConfig) -> bytes:
    """Fetch a remote URL with a size guard using stdlib only.

    This is a synchronous helper.  For the async pipelines the URL is
    already validated by *validate_remote_url* and the actual fetch is done
    via aiohttp in the pipeline code — this function is provided as a
    convenience for callers that need bytes immediately.
    """
    from urllib.request import urlopen, Request

    req = Request(url)
    total = 0
    chunks: list[bytes] = []
    with urlopen(req, timeout=15) as resp:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > cfg.max_bytes:
                raise ValueError(
                    f"ref_audio URL response too large: >{cfg.max_bytes} bytes"
                )
            chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Convenience: build a RefAudioConfig from the app's existing config object
# ---------------------------------------------------------------------------

_config_warning_logged = False


def ref_audio_config_from_app_config(app_cfg: object | None) -> RefAudioConfig:
    """Build a :class:`RefAudioConfig` from the application config object.

    Reads ``app_cfg.tts.ref_audio_dir`` and ``app_cfg.tts.allowed_ref_audio_hosts``
    if present; otherwise falls back to defaults and logs a one-time INFO warning.
    """
    global _config_warning_logged

    allowed_dir = _DEFAULT_REF_AUDIO_DIR
    allowed_hosts: list[str] = []

    # The app config is a ServerConfig with a .get() method
    get_method = getattr(app_cfg, "get", None) if app_cfg is not None else None
    if callable(get_method):
        ref_audio_dir_val = get_method("services", "tts", "ref_audio_dir", default=None)
        if ref_audio_dir_val is not None:
            allowed_dir = Path(ref_audio_dir_val).resolve()
        allowed_hosts_val = get_method("services", "tts", "allowed_ref_audio_hosts", default=None)
        if allowed_hosts_val is not None:
            allowed_hosts = list(allowed_hosts_val)
    elif app_cfg is not None and hasattr(app_cfg, "tts"):
        tts_cfg = getattr(app_cfg, "tts", None)
        if tts_cfg is not None:
            if hasattr(tts_cfg, "ref_audio_dir"):
                allowed_dir = Path(getattr(tts_cfg, "ref_audio_dir")).resolve()
            if hasattr(tts_cfg, "allowed_ref_audio_hosts"):
                allowed_hosts = list(getattr(tts_cfg, "allowed_ref_audio_hosts"))

    if not allowed_hosts and not _config_warning_logged:
        logger.info(
            "ref_audio safety: no allowed_ref_audio_hosts configured — "
            "remote HTTP(S) ref_audio URLs will be rejected"
        )
        _config_warning_logged = True

    return RefAudioConfig(allowed_dir=allowed_dir, allowed_hosts=allowed_hosts)
