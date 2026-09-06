"""以子进程方式运行 scripts/check_config_parity.py。

使配置漂移（_DEFAULT_CONFIG vs config.yaml.example vs 代码消费的
security.* 键名）在 pytest 中直接失败，而不依赖手动执行脚本。
P1 回归：曾因 session.py 读取 max_audio_frame_b64（缺少 _bytes 后缀）
导致 yaml 限额配置静默失效。
"""
from __future__ import annotations

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_parity_script_passes() -> None:
    script = os.path.join(_REPO_ROOT, "scripts", "check_config_parity.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"check_config_parity.py failed:\n{result.stdout}\n{result.stderr}"
    )
