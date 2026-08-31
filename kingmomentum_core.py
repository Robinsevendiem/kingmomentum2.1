"""Standalone KingMomentum calculation and event-driven backtest engine."""

from __future__ import annotations

from datetime import date
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = {
    "SHSE.518880": "黄金ETF",
    "SHSE.513520": "日经ETF",
    "SHSE.513100": "纳指100",
    "SHSE.513020": "港股科技",
    "SHSE.510180": "上证180",
    "SHSE.511090": "30年国债",
    "SHSE.588120": "科创板",
    "SZSE.159915": "创业板",
    "SHSE.501018": "南方原油LOF",
}


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol in ASSETS:
        path = data_dir / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path).copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame.sort_index()
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{symbol}缺少字段：{sorted(required - set(frame.columns))}")
        data[symbol] = frame
    return data


def kingmomentum_score(frame: pd.DataFrame, window: int = 25) -> pd.Series:
    """Quadratic-weighted WLS log-price trend score used by the project."""
    close = frame["close"].astype(float)
    log_close = np.log(close.where(close > 0)).to_numpy()
    output = np.full(len(log_close), np.nan)
    x = np.arange(window, dtype=float)
    weights = 1.0 + np.square(np.linspace(0.0, 1.0, window))
    fit_weights = weights**2
    weight_sum = fit_weights.sum()
    weighted_x = np.dot(fit_weights, x)
    weighted_x2 = np.dot(fit_weights, x * x)
    denominator = weight_sum * weighted_x2 - weighted_x**2
    if len(log_close) < window or denominator <= 0:
        return pd.Series(output, index=frame.index, name="score")
    windows = np.lib.stride_tricks.sliding_window_view(log_close, window)
    valid = np.isfinite(windows).all(axis=1)
    weighted_y = windows @ fit_weights
    weighted_xy = windows @ (fit_weights * x)
    slope = (weight_sum * weighted_xy - weighted_x * weighted_y) / denominator
    intercept = (weighted_y - slope * weighted_x) / weight_sum
    fitted = intercept[:, None] + slope[:, None] * x[None, :]
    residual = windows - fitted
    mean = (windows @ weights) / weights.sum()
    sse = np.sum(weights[None, :] * residual**2, axis=1)
    sst = np.sum(weights[None, :] * (windows - mean[:, None])**2, axis=1)
    r_squared = np.clip(np.where(sst > 0, 1.0 - sse / sst, 0.0), 0.0, 1.0)
    scores = (np.exp(slope * 252.0) - 1.0) * r_squared * 100.0
    scores[~valid] = np.nan
    output[window - 1 :] = scores
    return pd.Series(output, index=frame.index, name="score")


def score_panel(data: dict[str, pd.DataFrame], window: int = 25) -> pd.DataFrame:
    return pd.DataFrame({symbol: kingmomentum_score(frame, window) for symbol, frame in data.items()}).sort_index()


def coverage_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, frame in data.items():
        rows.append(
            {
                "代码": symbol,
                "名称": ASSETS[symbol],
                "数据起始日": frame.index.min().date(),
                "数据结束日": frame.index.max().date(),
                "交易日数量": len(frame),
            }
        )
    return pd.DataFrame(rows)


def _price(data: dict[str, pd.DataFrame], symbol: str, day: pd.Timestamp, field: str) -> float | None:
    frame = data[symbol]
    if day in frame.index:
        value = frame.at[day, field]
        return float(value) if np.isfinite(value) and value > 0 else None
    return None


def _prior_vol(data: dict[str, pd.DataFrame], symbol: str, day: pd.Timestamp, lookback: int = 20) -> float:
    frame = data[symbol]
    close = frame.loc[frame.index < day, "close"]
    value = close.pct_change().tail(lookback).std(ddof=1) * np.sqrt(252.0)
    return float(value) if np.isfinite(value) and value > 0 else np.nan


def portfolio_volatility(data: dict[str, pd.DataFrame], symbols: list[str], day: pd.Timestamp, lookback: int = 20) -> float:
    """Estimate annualized volatility for an equal-weight portfolio before ``day``."""
    if not symbols:
        return np.nan
    returns = pd.concat([data[symbol]["close"].pct_change().rename(symbol) for symbol in symbols], axis=1)
    returns = returns.loc[returns.index < day].tail(lookback)
    covariance = returns.cov() * 252.0
    weights = np.full(len(symbols), 1.0 / len(symbols))
    covariance_array = covariance.reindex(index=symbols, columns=symbols).fillna(0.0).to_numpy()
    variance = float(weights @ covariance_array @ weights)
    volatility = np.sqrt(max(variance, 0.0))
    return float(volatility) if np.isfinite(volatility) and volatility > 0 else np.nan


def position_management_snapshot(
    data: dict[str, pd.DataFrame],
    signal_day: pd.Timestamp,
    selected: list[str],
    current_holdings: list[str],
    current_exposure: float,
    *,
    target_volatility: float | None = 0.20,
    rebalance_band: float = 0.10,
    volatility_day: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return the target-volatility position decision for the next execution.

    When a holding snapshot is viewed after a market close, ``volatility_day``
    should be the next trading day so the estimate includes the latest close,
    matching the backtest's next-open execution calculation.
    """
    estimated_volatility = portfolio_volatility(data, selected, volatility_day or signal_day)
    if not selected:
        target_exposure = 0.0
        status = "转入现金" if current_holdings else "持有现金"
        basis = "没有有效正分标的"
    elif target_volatility is None:
        target_exposure = 1.0
        status = "需要按信号换仓" if set(current_holdings) != set(selected) else "满仓持有"
        basis = "未启用目标波动率仓位管理"
    else:
        target_exposure = 1.0 if not np.isfinite(estimated_volatility) else min(1.0, target_volatility / estimated_volatility)
        if set(current_holdings) != set(selected):
            status = "需要按信号换仓"
            basis = "推荐标的与当前策略持仓不一致"
        elif abs(current_exposure - target_exposure) <= rebalance_band:
            status = "无需调整（调整带内）"
            basis = f"仓位偏离 {abs(current_exposure - target_exposure):.2%}，未超过 {rebalance_band:.0%} 调整带"
        elif current_exposure < target_exposure:
            status = "需要增仓"
            basis = f"当前仓位低于目标仓位，偏离 {target_exposure - current_exposure:.2%}"
        else:
            status = "需要减仓"
            basis = f"当前仓位高于目标仓位，偏离 {current_exposure - target_exposure:.2%}"
    return pd.DataFrame(
        [
            {
                "信号日期": signal_day.date(),
                "推荐持仓": "|".join(selected) if selected else "现金",
                "推荐持仓名称": "|".join(ASSETS[x] for x in selected) if selected else "现金",
                "当前策略持仓": "|".join(current_holdings) if current_holdings else "现金",
                "当前策略持仓名称": "|".join(ASSETS[x] for x in current_holdings) if current_holdings else "现金",
                "组合估计年化波动率": estimated_volatility,
                "目标波动率": target_volatility,
                "目标总仓位": target_exposure,
                "目标现金比例": 1.0 - target_exposure,
                "当前策略总仓位": current_exposure,
                "仓位偏离": current_exposure - target_exposure,
                "仓位调整带": rebalance_band,
                "仓位判断": status,
                "判断依据": basis,
            }
        ]
    )


def _select_targets(
    scores: pd.DataFrame,
    day: pd.Timestamp,
    current: list[str],
    *,
    top_n: int,
    buffer: float,
    cutoff: float,
) -> tuple[list[str], str, dict[str, float]]:
    today = scores.loc[day].dropna() if day in scores.index else pd.Series(dtype=float)
    positive = today[today > 0]
    if positive.empty:
        return [], "所有标的分数≤0，持有现金", {}
    valid = positive[positive <= cutoff]
    if valid.empty:
        return current, "所有正分标的过热，保留当前持仓", {}
    ranked = list(valid.sort_values(ascending=False).index)
    n = min(top_n, len(ranked))
    if not current:
        chosen = ranked[:n]
        return chosen, "建立新仓位", {x: float(today[x]) for x in chosen}
    survivors = [x for x in current if x in valid.index]
    if len(survivors) != len(current):
        candidates = [x for x in ranked if x not in survivors]
        chosen = (survivors + candidates)[:n]
        return chosen, "当前持仓过热或失效，进行替换", {x: float(today[x]) for x in chosen}
    if top_n == 1 and ranked[0] in current:
        return current, "持有当前第一名", {x: float(today[x]) for x in current}
    low, high = float(valid.min()), float(valid.max())
    normalized = (valid - low) / (high - low) * 100.0 if high != low else pd.Series(50.0, index=valid.index)
    if top_n == 1:
        leader = ranked[0]
        if leader in current:
            return current, "持有当前第一名", {x: float(today[x]) for x in current}
        weakest = current[0]
        if float(normalized[leader] - normalized[weakest]) > buffer:
            return [leader], "新第一名超过换仓缓冲", {leader: float(today[leader])}
        return current, "新第一名未超过换仓缓冲", {x: float(today[x]) for x in current}
    chosen = survivors[:n]
    for candidate in ranked:
        if candidate in chosen:
            continue
        weakest = min(chosen, key=lambda x: float(normalized.get(x, -np.inf))) if chosen else None
        if weakest is None or float(normalized[candidate] - normalized[weakest]) > buffer:
            if weakest is not None:
                chosen.remove(weakest)
            chosen.append(candidate)
    reason = "新标的超过换仓缓冲，调整组合" if set(chosen) != set(current) else "组合未超过换仓缓冲"
    return chosen, reason, {x: float(today[x]) for x in chosen}


def _metrics(nav: pd.Series, trades: pd.DataFrame) -> dict[str, float | str]:
    returns = nav.pct_change().fillna(0.0)
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1 / 365.25)
    annual = float((1.0 + total) ** (1.0 / years) - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else np.nan
    drawdown = nav / nav.cummax() - 1.0
    return {
        "累计收益": total,
        "年化收益": annual,
        "年化波动": vol,
        "夏普": annual / vol if vol > 0 else np.nan,
        "最大回撤": float(drawdown.min()),
        "最大回撤日期": str(drawdown.idxmin()),
        "交易次数": int(len(trades)),
    }


def backtest(
    data: dict[str, pd.DataFrame],
    scores: pd.DataFrame,
    *,
    start: date,
    end: date,
    top_n: int = 1,
    buffer: float = 5.0,
    cutoff: float = 500.0,
    fee: float = 0.0005,
    target_volatility: float | None = 0.15,
    rebalance_band: float = 0.10,
    initial: float = 100_000.0,
) -> tuple[dict[str, float | str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(sorted(set().union(*[set(frame.index) for frame in data.values()])))
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    cash = initial
    holdings: dict[str, float] = {}
    pending: list[str] = []
    values: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []
    pending_signal_day: date | None = None
    pending_reason = "建立新仓位"
    pending_scores: dict[str, float] = {}
    last_rebalance_nav: float | None = None

    for day in dates:
        open_values = {x: holdings[x] * (_price(data, x, day, "open") or 0.0) for x in holdings}
        nav_open = cash + sum(open_values.values())
        pre_holdings = dict(holdings)
        pre_exposure = sum(open_values.values()) / nav_open if nav_open else 0.0
        trade_start = len(trades)
        target_exposure = 1.0
        target_weights = pd.Series(1.0 / len(pending), index=pending, dtype=float) if pending else pd.Series(dtype=float)
        if pending and target_volatility is not None:
            returns = pd.concat([data[x]["close"].pct_change().rename(x) for x in pending], axis=1)
            returns = returns.loc[returns.index < day].tail(20)
            covariance = returns.cov() * 252.0
            weight_array = target_weights.reindex(covariance.index).fillna(0).to_numpy()
            variance = float(weight_array @ covariance.fillna(0).to_numpy() @ weight_array)
            current_vol = np.sqrt(max(variance, 0.0))
            if np.isfinite(current_vol) and current_vol > 0:
                target_exposure = min(1.0, target_volatility / current_vol)
        target_weights *= target_exposure
        desired_values = nav_open * target_weights
        if pending and holdings and set(pending) == set(holdings) and rebalance_band > 0:
            current_weights = pd.Series(open_values) / nav_open if nav_open else pd.Series(dtype=float)
            union = current_weights.index.union(target_weights.index)
            gap = (current_weights.reindex(union).fillna(0) - target_weights.reindex(union).fillna(0)).abs().max()
            cash_gap = abs(cash / nav_open - (1.0 - target_weights.sum())) if nav_open else 0.0
            if max(float(gap), float(cash_gap)) <= rebalance_band:
                desired_values = pd.Series(open_values).reindex(pending).fillna(0.0)

        def sell(symbol: str, shares: float, reason: str, trade_day: pd.Timestamp = day) -> None:
            nonlocal cash
            price = _price(data, symbol, trade_day, "open")
            if price is None or shares <= 0:
                return
            amount = shares * price
            if amount <= 1e-6:
                return
            cash += amount * (1.0 - fee)
            holdings[symbol] = holdings.get(symbol, 0.0) - shares
            if holdings[symbol] <= 1e-10:
                holdings.pop(symbol, None)
            trades.append({"日期": trade_day.date(), "动作": "卖出", "代码": symbol, "名称": ASSETS[symbol], "成交价格": price, "金额": amount, "原因": reason})

        def buy(symbol: str, shares: float, reason: str, trade_day: pd.Timestamp = day) -> None:
            nonlocal cash
            price = _price(data, symbol, trade_day, "open")
            if price is None or shares <= 0:
                return
            shares = min(shares, cash / (price * (1.0 + fee)))
            amount = shares * price
            if amount <= 1e-6:
                return
            cash -= amount * (1.0 + fee)
            holdings[symbol] = holdings.get(symbol, 0.0) + shares
            trades.append({"日期": trade_day.date(), "动作": "买入", "代码": symbol, "名称": ASSETS[symbol], "成交价格": price, "金额": amount, "原因": reason})

        if not pending:
            for symbol in list(holdings):
                sell(symbol, holdings[symbol], "转入现金")
        elif all(_price(data, symbol, day, "open") is not None for symbol in pending):
            for symbol in list(holdings):
                if symbol not in pending:
                    sell(symbol, holdings[symbol], "调仓替换")
            for symbol in list(holdings):
                price = _price(data, symbol, day, "open")
                target = float(desired_values.get(symbol, 0.0))
                if price and holdings[symbol] * price > target:
                    sell(symbol, holdings[symbol] - target / price, "风险再平衡")
            for symbol in pending:
                price = _price(data, symbol, day, "open")
                target = float(desired_values.get(symbol, 0.0))
                current_value = holdings.get(symbol, 0.0) * (price or 0.0)
                if price and current_value < target:
                    buy(symbol, (target - current_value) / price, "目标仓位")

        day_trades = trades[trade_start:]
        event_row: dict[str, object] | None = None
        if day_trades:
            post_open_values = {x: holdings[x] * (_price(data, x, day, "open") or 0.0) for x in holdings}
            nav_after_trade = cash + sum(post_open_values.values())
            post_exposure = sum(post_open_values.values()) / nav_after_trade if nav_after_trade else 0.0
            trade_reasons = {str(trade["原因"]) for trade in day_trades}
            signal_rebalance = set(pre_holdings) != set(pending)
            if signal_rebalance:
                rebalance_type = "信号调仓"
                event_reason = pending_reason
            elif trade_reasons & {"风险再平衡", "目标仓位"}:
                rebalance_type = "风险仓位再平衡"
                event_reason = "目标波动率仓位调整"
            else:
                rebalance_type = "组合调仓"
                event_reason = "调整组合仓位"
            event_row = {
                "信号日期": pending_signal_day or day.date(),
                "执行日期": day.date(),
                "调仓类型": rebalance_type,
                "当前持仓": "|".join(sorted(pre_holdings)) if pre_holdings else "现金",
                "当前持仓名称": "|".join(ASSETS[x] for x in sorted(pre_holdings)) if pre_holdings else "现金",
                "目标持仓": "|".join(sorted(pending)) if pending else "现金",
                "目标持仓名称": "|".join(ASSETS[x] for x in sorted(pending)) if pending else "现金",
                "原因": event_reason,
                "最高分": max(pending_scores.values()) if pending_scores else np.nan,
                "目标总仓位": float(target_weights.sum()),
                "调仓前总仓位": pre_exposure,
                "调仓后总仓位": post_exposure,
                "仓位变化": post_exposure - pre_exposure,
                "调仓前净值": nav_open,
                "调仓后净值": nav_after_trade,
                "调仓净值影响": nav_after_trade / nav_open - 1.0 if nav_open else np.nan,
                "手续费估算": sum(float(trade["金额"]) for trade in day_trades) * fee,
                "成交笔数": len(day_trades),
                "上次调仓至本次调仓前收益率": (nav_open / last_rebalance_nav - 1.0) if last_rebalance_nav else np.nan,
            }
            rebalance_rows.append(event_row)

        close_values = {x: holdings[x] * (_price(data, x, day, "close") or 0.0) for x in holdings}
        nav_close = cash + sum(close_values.values())
        exposure = sum(close_values.values()) / nav_close if nav_close else 0.0
        values.append({"日期": day.date(), "净值": nav_close, "持仓": "|".join(sorted(holdings)) if holdings else "现金", "总仓位": exposure})
        for symbol, value in close_values.items():
            holdings_rows.append({"日期": day.date(), "代码": symbol, "名称": ASSETS[symbol], "份额": holdings[symbol], "市值": value, "组合权重": value / nav_close if nav_close else 0.0, "总仓位": exposure, "现金": cash})

        signal_day = scores.index[scores.index <= day][-1] if len(scores.index[scores.index <= day]) else None
        if signal_day is not None:
            selected, reason, score_values = _select_targets(scores, signal_day, list(holdings), top_n=top_n, buffer=buffer, cutoff=cutoff)
            if selected != pending:
                pending_signal_day = signal_day.date()
                pending_reason = reason
                pending_scores = score_values
            pending = selected
        if event_row is not None:
            event_row["调仓后收盘净值"] = nav_close
            last_rebalance_nav = nav_close

    value_frame = pd.DataFrame(values).set_index("日期")
    trade_frame = pd.DataFrame(trades)
    metrics = _metrics(value_frame["净值"], trade_frame)
    return metrics, value_frame.reset_index(), pd.DataFrame(holdings_rows), pd.DataFrame(rebalance_rows), trade_frame


def latest_signal(
    data: dict[str, pd.DataFrame],
    scores: pd.DataFrame,
    *,
    top_n: int = 1,
    buffer: float = 5.0,
    cutoff: float = 500.0,
    as_of: date | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame, list[str], str]:
    latest = min(frame.index.max() for frame in data.values())
    if as_of is not None:
        latest = min(latest, pd.Timestamp(as_of))
    available_scores = scores.index[scores.index <= latest]
    if len(available_scores) == 0:
        raise ValueError(f"{latest.date()} 之前没有可用的动量分数")
    score_day = available_scores[-1]
    row = scores.loc[score_day].rename("动量分数").to_frame()
    row["名称"] = row.index.map(ASSETS)
    row["状态"] = np.select([row["动量分数"] <= 0, row["动量分数"] > cutoff], ["非正分", "过热"], default="有效候选")
    row = row[["名称", "动量分数", "状态"]].sort_values("动量分数", ascending=False)
    positive = row[(row["动量分数"] > 0) & (row["动量分数"] <= cutoff)]
    selected = positive.head(top_n).index.tolist()
    reason = "所有分数≤0，持有现金" if positive.empty else "按25日动量排名选择有效候选"
    return score_day, row, selected, reason


def _prepare_pandadata_runtime() -> None:
    """让 PandaData 的认证缓存落在部署环境可写的临时目录。

    panda_data 0.0.12 的 init_token() 会把加密认证状态写入 SDK 安装目录，
    但 Streamlit Community Cloud 的虚拟环境目录是只读的。将其内部缓存目录
    指向临时目录后，认证仍只在当前运行容器内保存，不会写入 GitHub 仓库。
    """
    import panda_data.auth_manager as auth_manager

    runtime_dir = Path(tempfile.gettempdir()) / "kingmomentum_pandadata"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # 0.0.12 没有公开的 user.json 路径配置；该变量由 SDK 的认证管理器使用。
    auth_manager._user_json_dir = str(runtime_dir)


def refresh_with_pandadata(data_dir: Path, username: str, password: str, start: date, end: date) -> tuple[int, str]:
    """Optional updater; requires PandaData credentials and SDK in deployment secrets."""
    try:
        import panda_data
    except ImportError as exc:
        raise RuntimeError("未安装 panda_data，请在 requirements.txt 中安装后重试") from exc
    if not username or not password:
        raise RuntimeError("请先配置 PANDA_DATA_USERNAME 和 PANDA_DATA_PASSWORD")
    _prepare_pandadata_runtime()
    panda_data.init_token(username=username, password=password)
    updated = 0
    for symbol in ASSETS:
        prefix, code = symbol.split(".")
        exchange = "SH" if prefix == "SHSE" else "SZ"
        native = f"{code}.{exchange}"
        existing_path = data_dir / f"{symbol}.parquet"
        if existing_path.exists():
            existing = pd.read_parquet(existing_path)
            existing.index = pd.to_datetime(existing.index).tz_localize(None)
            fetch_start = existing.index.max().date() + pd.Timedelta(days=1).to_pytimedelta()
        else:
            existing = pd.DataFrame()
            fetch_start = start
        if fetch_start > end:
            continue
        frames = []
        chunk_start = fetch_start
        while chunk_start <= end:
            chunk_end = min(chunk_start + pd.Timedelta(days=364).to_pytimedelta(), end)
            frame = panda_data.get_fund_daily_pre(start_date=chunk_start.strftime("%Y%m%d"), end_date=chunk_end.strftime("%Y%m%d"), symbol=native, fields=[])
            if frame is not None and not frame.empty:
                frame = frame.copy()
                date_column = "date" if "date" in frame.columns else "eob"
                frame[date_column] = pd.to_datetime(frame[date_column])
                frames.append(frame.set_index(date_column))
            chunk_start = chunk_end + pd.Timedelta(days=1).to_pytimedelta()
        if not frames:
            continue
        frame = pd.concat([existing, *frames], axis=0)
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        columns = [x for x in ["open", "high", "low", "close", "volume"] if x in frame.columns]
        if len(columns) < 5:
            continue
        frame[columns].to_parquet(existing_path)
        updated += 1
    return updated, end.isoformat()
