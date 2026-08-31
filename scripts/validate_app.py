"""Deployment preflight checks for the standalone Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ["app.py", "kingmomentum_core.py", "requirements.txt", "README.md"]
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
DATA_FILES = sorted((APP_DIR / "data").glob("*.parquet"))


def check_source_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (APP_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"缺少项目文件：{', '.join(missing)}")
    for name in ("app.py", "kingmomentum_core.py"):
        source = (APP_DIR / name).read_text(encoding="utf-8")
        compile(source, name, "exec")


def check_secret_boundary() -> None:
    forbidden = [APP_DIR / ".streamlit" / "secrets.toml", APP_DIR / ".env"]
    present = [str(path.relative_to(APP_DIR)) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"请勿将本地密钥文件放入待上传目录：{', '.join(present)}")


def check_data() -> None:
    if len(DATA_FILES) != 9:
        raise RuntimeError(f"预期9个数据文件，实际发现 {len(DATA_FILES)} 个")
    for path in DATA_FILES:
        frame = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path.name} 缺少字段：{sorted(missing)}")
        if frame.empty or frame.index.isna().any():
            raise RuntimeError(f"{path.name} 为空或存在无效日期索引")
        if not frame.index.is_monotonic_increasing:
            raise RuntimeError(f"{path.name} 日期未按升序排列")
        if (frame["close"] <= 0).any():
            raise RuntimeError(f"{path.name} 存在非正收盘价")
        print(f"{path.name}: {frame.index.min().date()} 至 {frame.index.max().date()}，{len(frame)} 行")


def main() -> int:
    try:
        check_source_files()
        check_secret_boundary()
        check_data()
    except Exception as exc:
        print(f"预检失败：{exc}", file=sys.stderr)
        return 1
    print("预检通过：代码、依赖边界、密钥边界和9个数据快照均正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
