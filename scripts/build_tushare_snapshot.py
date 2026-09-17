"""Build an independent Tushare-adjusted snapshot for deployment."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kingmomentum_core import ASSETS, fetch_symbol_with_tushare, load_data


def parse_date(value: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"无法识别日期：{value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="将Tushare原始日线+复权因子重建为独立部署快照")
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help="统一起始日期；默认沿用PandaData快照中每个标的的最早日期",
    )
    parser.add_argument("--end-date", type=parse_date, default=date.today(), help="结束日期，默认今天")
    args = parser.parse_args()

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("缺少环境变量 TUSHARE_TOKEN；请在本机配置后重试。")
    if args.start_date and args.start_date >= args.end_date:
        raise SystemExit("起始日期必须早于结束日期。")

    target_dir = ROOT / "data_tushare"
    target_dir.mkdir(parents=True, exist_ok=True)
    panda_data = load_data(ROOT / "data")

    for symbol in ASSETS:
        start = args.start_date or panda_data[symbol].index.min().date()
        if start >= args.end_date:
            raise SystemExit(f"{symbol} 的起始日期不早于结束日期。")
        print(f"正在下载 {symbol}：{start} 至 {args.end_date}")
        frame = fetch_symbol_with_tushare(symbol, token, start, args.end_date)
        output = target_dir / f"{symbol}.parquet"
        frame.to_parquet(output)
        print(f"已保存 {output.relative_to(ROOT)}：{frame.index.min().date()} 至 {frame.index.max().date()}，{len(frame)} 行")

    print("Tushare 独立快照构建完成；PandaData 的 data/ 未被修改。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
