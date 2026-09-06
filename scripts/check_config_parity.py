#!/usr/bin/env python3
"""Check parity between _DEFAULT_CONFIG in server/config.py and config.yaml.example.

Recursively walks both structures and asserts every key path exists in both
with matching values. Exits 0 on success, 1 on mismatch.
"""
import os
import re
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from server.config import _DEFAULT_CONFIG


def _flatten(d, prefix=""):
    """Flatten a nested dict into {path: value} pairs."""
    items = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, path))
        else:
            items[path] = v
    return items


def _values_match(a, b):
    """Check if two values match, allowing string vs path equivalence."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        # None vs empty string/list is acceptable for optional fields
        if a is None and b == "":
            return True
        if b is None and a == "":
            return True
        if a is None and b == []:
            return True
        if b is None and a == []:
            return True
        return False
    if type(a) != type(b):
        # Allow int vs float comparison
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) == float(b)
        # Allow bool vs string "true"/"false" (YAML parsing)
        if isinstance(a, bool) and isinstance(b, str):
            return str(a).lower() == b.lower()
        if isinstance(b, bool) and isinstance(a, str):
            return str(b).lower() == a.lower()
        return False
    return a == b


def check_consumer_security_keys(root: str) -> list[str]:
    """扫描代码中消费的 security.* 配置键，确保都在 _DEFAULT_CONFIG 中定义。

    防止键名不一致（如 max_audio_frame_b64 vs max_audio_frame_b64_bytes）
    导致 yaml 配置静默失效。
    """
    defaults_sec = _DEFAULT_CONFIG.get("security", {})
    patterns = [
        # config.get("security", "<key>", ...)
        re.compile(r'\.get\(\s*"security"\s*,\s*"([A-Za-z0-9_]+)"'),
        # _sec = config.get("security", ...) 后 _sec.get("<key>", ...)
        re.compile(r'_sec\.get\(\s*"([A-Za-z0-9_]+)"'),
    ]
    files = [os.path.join(root, "main.py")]
    server_dir = os.path.join(root, "server")
    if os.path.isdir(server_dir):
        files += sorted(
            os.path.join(server_dir, f)
            for f in os.listdir(server_dir)
            if f.endswith(".py")
        )

    errors = []
    for path in files:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            src = f.read()
        fname = os.path.relpath(path, root)
        for pat in patterns:
            for key in sorted(set(pat.findall(src))):
                if key not in defaults_sec:
                    errors.append(
                        f"CONSUMER KEY security.{key!r} used in {fname} "
                        f"but missing in _DEFAULT_CONFIG"
                    )
    return errors


def main():
    # Load config.yaml.example
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml.example",
    )
    with open(example_path, "r", encoding="utf-8") as f:
        example_config = yaml.safe_load(f) or {}

    # Flatten both
    default_flat = _flatten(_DEFAULT_CONFIG)
    example_flat = _flatten(example_config)

    errors = []

    # Check all keys in _DEFAULT_CONFIG exist in example
    for path, value in default_flat.items():
        if path not in example_flat:
            errors.append(f"MISSING in config.yaml.example: {path} (default: {value!r})")
        elif not _values_match(value, example_flat[path]):
            errors.append(
                f"MISMATCH at {path}: _DEFAULT={value!r} vs example={example_flat[path]!r}"
            )

    # Check all keys in example exist in _DEFAULT_CONFIG
    for path, value in example_flat.items():
        if path not in default_flat:
            errors.append(f"MISSING in _DEFAULT_CONFIG: {path} (example: {value!r})")
        elif not _values_match(default_flat[path], value):
            # Already reported above
            pass

    errors += check_consumer_security_keys(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    if errors:
        print("PARITY CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
