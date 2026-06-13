"""Smoke tests for server._ref_audio_safety — path-traversal and SSRF hardening."""

import ipaddress
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from server._ref_audio_safety import (
    RefAudioConfig,
    resolve_local_ref,
    validate_remote_url,
    read_ref_audio_safely,
    _is_ip_private_or_reserved,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_allowed_dir(tmp_path: Path) -> Path:
    """Return a temporary allowed directory with one dummy audio file."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return tmp_path


@pytest.fixture()
def cfg(tmp_allowed_dir: Path) -> RefAudioConfig:
    """RefAudioConfig pointing at the temp allowed dir, no remote hosts."""
    return RefAudioConfig(allowed_dir=tmp_allowed_dir, allowed_hosts=[])


@pytest.fixture()
def cfg_with_hosts(tmp_allowed_dir: Path) -> RefAudioConfig:
    """RefAudioConfig that allows a specific remote host."""
    return RefAudioConfig(
        allowed_dir=tmp_allowed_dir,
        allowed_hosts=["cdn.example.com"],
    )


# ---------------------------------------------------------------------------
# 1. Path traversal — rejects ../../etc/passwd
# ---------------------------------------------------------------------------

def test_rejects_path_traversal_dotdot(cfg: RefAudioConfig):
    """Any '..' segment in the path must be rejected."""
    with pytest.raises(ValueError, match="path traversal"):
        resolve_local_ref("../../etc/passwd", cfg)


def test_rejects_path_traversal_dotdot_middle(cfg: RefAudioConfig):
    """Path like foo/../../etc/passwd is also rejected."""
    with pytest.raises(ValueError, match="path traversal"):
        resolve_local_ref("foo/../../etc/passwd", cfg)


# ---------------------------------------------------------------------------
# 2. Absolute path — rejects /etc/passwd
# ---------------------------------------------------------------------------

def test_rejects_absolute_path_outside_allowed(cfg: RefAudioConfig, tmp_allowed_dir: Path):
    """An absolute path that resolves outside the allowed dir is rejected."""
    with pytest.raises(ValueError, match="path escape|outside"):
        resolve_local_ref("/etc/passwd", cfg)


# ---------------------------------------------------------------------------
# 3. SSRF — rejects http://127.0.0.1
# ---------------------------------------------------------------------------

def test_rejects_loopback_url(tmp_allowed_dir: Path):
    """HTTP URL pointing to 127.0.0.1 must be rejected (loopback SSRF)."""
    # Put 127.0.0.1 in the allowlist so we reach the IP check
    cfg = RefAudioConfig(allowed_dir=tmp_allowed_dir, allowed_hosts=["127.0.0.1"])
    with pytest.raises(ValueError, match="private/loopback"):
        validate_remote_url("http://127.0.0.1/secret", cfg)


# ---------------------------------------------------------------------------
# 4. SSRF — rejects http://169.254.169.254 (cloud metadata)
# ---------------------------------------------------------------------------

def test_rejects_cloud_metadata_url(cfg_with_hosts: RefAudioConfig):
    """HTTP URL pointing to 169.254.169.254 must be rejected (cloud metadata SSRF)."""
    # 169.254.x.x is link-local — the allowlist should pass but IP check should reject
    cfg = RefAudioConfig(
        allowed_dir=cfg_with_hosts.allowed_dir,
        allowed_hosts=["169.254.169.254"],
    )
    with pytest.raises(ValueError, match="private/loopback|link-local"):
        validate_remote_url("http://169.254.169.254/latest/meta-data/", cfg)


# ---------------------------------------------------------------------------
# 5. Allowlisted relative path with dummy file
# ---------------------------------------------------------------------------

def test_allows_relative_path_under_allowed_dir(cfg: RefAudioConfig, tmp_allowed_dir: Path):
    """A relative path that stays inside allowed_dir should succeed."""
    result = resolve_local_ref("sample.wav", cfg)
    assert result.is_relative_to(tmp_allowed_dir.resolve())
    assert result.exists()


# ---------------------------------------------------------------------------
# 6. read_ref_audio_safely — local file
# ---------------------------------------------------------------------------

def test_read_local_file_safely(cfg: RefAudioConfig):
    """Reading a valid local file under allowed_dir should return its bytes."""
    data = read_ref_audio_safely("sample.wav", cfg)
    assert data.startswith(b"RIFF")


# ---------------------------------------------------------------------------
# 7. Hostname not in allowlist
# ---------------------------------------------------------------------------

def test_rejects_url_hostname_not_in_allowlist(cfg: RefAudioConfig):
    """When allowed_hosts is empty, any HTTP URL hostname is rejected."""
    with pytest.raises(ValueError, match="not in allowlist"):
        validate_remote_url("http://evil.com/audio.wav", cfg)


# ---------------------------------------------------------------------------
# 8. Hostname in allowlist but resolves to private IP
# ---------------------------------------------------------------------------

def test_rejects_allowlisted_host_with_private_ip(cfg_with_hosts: RefAudioConfig):
    """Even if hostname is allowlisted, resolved private IP must be rejected."""
    # Mock DNS to return a private IP for cdn.example.com
    with patch("server._ref_audio_safety.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.1", 80))
        ]
        with pytest.raises(ValueError, match="private/loopback"):
            validate_remote_url("http://cdn.example.com/audio.wav", cfg_with_hosts)


# ---------------------------------------------------------------------------
# 9. _is_ip_private_or_reserved helper
# ---------------------------------------------------------------------------

def test_private_ips_detected():
    assert _is_ip_private_or_reserved("10.0.0.1") is True
    assert _is_ip_private_or_reserved("172.16.0.1") is True
    assert _is_ip_private_or_reserved("192.168.1.1") is True
    assert _is_ip_private_or_reserved("127.0.0.1") is True
    assert _is_ip_private_or_reserved("169.254.169.254") is True
    assert _is_ip_private_or_reserved("224.0.0.1") is True  # multicast
    assert _is_ip_private_or_reserved("8.8.8.8") is False
    assert _is_ip_private_or_reserved("1.1.1.1") is False


# ---------------------------------------------------------------------------
# 10. data: URL passthrough
# ---------------------------------------------------------------------------

def test_data_url_passthrough(cfg: RefAudioConfig):
    """data: URLs should be returned as-is by read_ref_audio_safely."""
    result = read_ref_audio_safely("data:audio/wav;base64,UklGRiQ=", cfg)
    assert result.startswith(b"data:audio/wav")


# ---------------------------------------------------------------------------
# 11. Scheme rejection
# ---------------------------------------------------------------------------

def test_rejects_file_scheme(cfg: RefAudioConfig):
    """file:// URLs must be rejected."""
    with pytest.raises(ValueError, match="scheme rejected"):
        validate_remote_url("file:///etc/passwd", cfg)


def test_rejects_ftp_scheme(cfg: RefAudioConfig):
    """ftp:// URLs must be rejected."""
    with pytest.raises(ValueError, match="scheme rejected"):
        validate_remote_url("ftp://evil.com/audio.wav", cfg)
