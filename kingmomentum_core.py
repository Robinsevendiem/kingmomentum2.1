"""Standalone KingMomentum calculation and event-driven backtest engine."""

from __future__ import annotations

from collections.abc import Sequence
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
DEFAULT_SYMBOLS = tuple(ASSETS)
REQUIRED_MARKET_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_symbol_input(value: str) -> str | None:
    """Normalize common A-share exchange/code formats to ``SHSE.123456``."""
    text = str(value or "").strip().upper().replace("-", ".")
    if not text:
        return None
    parts = [part for part in text.split(".") if part]
    exchange: str | None = None
    code: str | None = None
    if len(parts) == 1 and parts[0].isdigit():
        code = parts[0].zfill(6)
    elif len(parts) == 2:
        left, right = parts
        exchange_aliases = {"SH": "SHSE", "SHSE": "SHSE", "SSE": "SHSE", "SZ": "SZSE", "SZSE": "SZSE"}
        if left in exchange_aliases and right.isdigit():
            exchange, code = exchange_aliases[left], right.zfill(6)
        elif right in exchange_aliases and left.isdigit():
            exchange, code = exchange_aliases[right], left.zfill(6)
    if code is None or len(code) != 6:
        return None
    if exchange is None:
        # ETF/LOF codes in the project's universe are unambiguous under this
        # convention: 5/6/9-series are Shanghai, 0/1/2/3-series Shenzhen.
        if code.startswith(("0", "1", "2", "3")):
            exchange = "SZSE"
        elif code.startswith(("5", "6", "9")):
            exchange = "SHSE"
        else:
            return None
    return f"{exchange}.{code}"


def _normalize_market_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize a local/API daily frame to the app's OHLCV contract."""
    if frame is None or frame.empty:
        raise ValueError(f"{symbol}没有可用日线数据")
    normalized = frame.copy()
    date_column = next((column for column in ("date", "eob", "trade_date") if column in normalized.columns), None)
    if date_column is not None:
        normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.pop(date_column))).tz_localize(None)
    else:
        normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index)).tz_localize(None)
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    missing = set(REQUIRED_MARKET_COLUMNS) - set(normalized.columns)
    if missing:
        raise ValueError(f"{symbol}缺少字段：{sorted(missing)}")
    normalized = normalized.loc[:, list(REQUIRED_MARKET_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if normalized.empty or (normalized["close"] <= 0).all():
        raise ValueError(f"{symbol}没有有效收盘价")
    return normalized


def load_data(data_dir: Path, symbols: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    requested = tuple(DEFAULT_SYMBOLS if symbols is None else symbols)
    for symbol in requested:
        path = data_dir / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        data[symbol] = _normalize_market_frame(pd.read_parquet(path), symbol)
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


EXECUTION_PRICE_MODES = {
    "open": "执行日开盘价",
    "typical": "执行日OHLC典型价（四价均值）",
}


def _execution_base_price(
    data: dict[str, pd.DataFrame], symbol: str, day: pd.Timestamp, mode: str
) -> float | None:
    """Return the observable daily price used as the execution reference.

    ``open`` is the current project baseline. ``typical`` is an OHLC proxy for
    sensitivity analysis; it is not a true intraday VWAP.
    """
    if mode not in EXECUTION_PRICE_MODES:
        raise ValueError(f"未知成交价格模式：{mode}")
    if mode == "open":
        return _price(data, symbol, day, mode)
    frame = data[symbol]
    if day not in frame.index:
        return None
    row = frame.loc[day, ["open", "high", "low", "close"]]
    values = pd.to_numeric(row, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= 0).any():
        return None
    return float(values.mean())


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


TREND_FILTER_LABELS = {
    "price_above_sma60": "价格高于60日均线",
    "sma60_up": "价格高于60日均线且均线向上",
    "multi_horizon_vote": "20/60/120日收益多数为正",
}


def single_asset_trend_symbols(
    data: dict[str, pd.DataFrame], day: pd.Timestamp, method: str
) -> set[str]:
    """Return symbols passing a candidate-level, close-only trend filter."""
    if method not in TREND_FILTER_LABELS:
        raise ValueError(f"未知单标的趋势过滤器：{method}")
    passed: set[str] = set()
    for symbol, frame in data.items():
        close = frame["close"].astype(float).loc[lambda series: series.index <= day]
        if close.empty:
            continue
        latest = float(close.iloc[-1])
        sma60 = close.rolling(60, min_periods=60).mean()
        if method == "price_above_sma60":
            if len(close) >= 60 and np.isfinite(sma60.iloc[-1]) and latest > float(sma60.iloc[-1]):
                passed.add(symbol)
        elif method == "sma60_up":
            if len(close) < 80:
                continue
            current_sma = sma60.iloc[-1]
            prior_sma = sma60.iloc[-21]
            if np.isfinite(current_sma) and np.isfinite(prior_sma) and latest > current_sma and current_sma > prior_sma:
                passed.add(symbol)
        elif method == "multi_horizon_vote":
            if len(close) < 121:
                continue
            returns = [latest / float(close.iloc[-lookback - 1]) - 1.0 for lookback in (20, 60, 120)]
            if sum(value > 0 for value in returns) >= 2:
                passed.add(symbol)
    return passed


def single_asset_trend_scale(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    day: pd.Timestamp,
    method: str,
    weak_multiplier: float,
) -> float:
    """Return a soft exposure multiplier using information before ``day``.

    The multiplier is evaluated for the currently intended holdings before
    the execution-day open. For the current Top-1 strategy, ``symbols`` has
    one member. Multiple holdings use the mean of their individual factors.
    """
    if not symbols:
        return 0.0
    if not 0.0 <= weak_multiplier <= 1.0:
        raise ValueError("趋势弱势仓位系数必须位于0到1之间")
    factors: list[float] = []
    for symbol in symbols:
        frame = data[symbol]
        close = frame["close"].astype(float).loc[lambda series: series.index < day]
        if close.empty:
            factors.append(weak_multiplier)
            continue
        latest = float(close.iloc[-1])
        if method == "price_above_sma60":
            sma60 = close.rolling(60, min_periods=60).mean().iloc[-1]
            passed = len(close) >= 60 and np.isfinite(sma60) and latest > float(sma60)
        elif method == "sma60_up":
            if len(close) < 80:
                passed = False
            else:
                sma60 = close.rolling(60, min_periods=60).mean()
                current_sma = sma60.iloc[-1]
                prior_sma = sma60.iloc[-21]
                passed = np.isfinite(current_sma) and np.isfinite(prior_sma) and latest > float(current_sma) and current_sma > float(prior_sma)
        elif method == "multi_horizon_vote":
            if len(close) < 121:
                passed = False
            else:
                returns = [latest / float(close.iloc[-lookback - 1]) - 1.0 for lookback in (20, 60, 120)]
                passed = sum(value > 0 for value in returns) >= 2
        else:
            raise ValueError(f"未知趋势缩放器：{method}")
        factors.append(1.0 if passed else weak_multiplier)
    return float(np.mean(factors))


def _annualized_covariance(returns: pd.DataFrame, method: str, ewma_half_life: int) -> pd.DataFrame:
    """Estimate an annualized covariance matrix for position sizing."""
    if method == "rolling":
        return returns.cov() * 252.0
    if method == "downside":
        clean = returns.dropna(how="any")
        if clean.empty:
            return clean.cov()
        downside = np.minimum(clean.to_numpy(dtype=float), 0.0)
        covariance = downside.T @ downside / len(clean)
        return pd.DataFrame(covariance * 252.0, index=clean.columns, columns=clean.columns)
    if method != "ewma":
        raise ValueError(f"未知波动率估计方法：{method}")
    if ewma_half_life < 1:
        raise ValueError("EWMA半衰期必须为正整数")
    clean = returns.dropna(how="any")
    if clean.empty:
        return clean.cov()
    decay = 0.5 ** (1.0 / ewma_half_life)
    weights = decay ** np.arange(len(clean) - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    values = clean.to_numpy(dtype=float)
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean[None, :]
    covariance = (centered * weights[:, None]).T @ centered
    return pd.DataFrame(covariance * 252.0, index=clean.columns, columns=clean.columns)


def _select_targets(
    scores: pd.DataFrame,
    day: pd.Timestamp,
    current: list[str],
    *,
    top_n: int,
    buffer: float,
    cutoff: float,
    allowed_symbols: set[str] | None = None,
) -> tuple[list[str], str, dict[str, float]]:
    today = scores.loc[day].dropna() if day in scores.index else pd.Series(dtype=float)
    positive = today[today > 0]
    if positive.empty:
        return [], "所有标的分数≤0，持有现金", {}
    valid = positive[positive <= cutoff]
    if allowed_symbols is not None:
        valid = valid[valid.index.isin(allowed_symbols)]
    if valid.empty:
        if allowed_symbols is not None:
            return [], "没有通过单标的趋势确认的有效候选，持有现金", {}
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
    execution_price_mode: str = "open",
    target_volatility: float | None = 0.15,
    rebalance_band: float = 0.10,
    volatility_lookback: int = 20,
    volatility_method: str = "rolling",
    ewma_half_life: int = 10,
    volatility_shock_enabled: bool = False,
    shock_short_window: int = 5,
    shock_long_window: int = 40,
    shock_trigger_ratio: float = 1.50,
    shock_recovery_ratio: float = 1.20,
    shock_recovery_days: int = 3,
    shock_multiplier: float = 0.50,
    trend_filter: str | None = None,
    trend_scaling_method: str | None = None,
    trend_weak_multiplier: float = 0.50,
    max_target_exposure: float | None = None,
    min_target_exposure: float = 0.0,
    max_exposure_increase: float | None = None,
    recovery_uses_raw_target_for_band: bool = False,
    initial: float = 100_000.0,
) -> tuple[dict[str, float | str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if execution_price_mode not in EXECUTION_PRICE_MODES:
        raise ValueError(f"未知成交价格模式：{execution_price_mode}")
    if volatility_lookback < 2:
        raise ValueError("波动率回看窗口至少需要2个交易日")
    if shock_short_window < 2 or shock_long_window <= shock_short_window:
        raise ValueError("波动率突增窗口必须满足长期窗口大于短期窗口，且短期窗口至少为2日")
    if shock_trigger_ratio <= 0 or shock_recovery_ratio <= 0 or shock_recovery_ratio >= shock_trigger_ratio:
        raise ValueError("波动率突增的恢复阈值必须小于触发阈值且均为正数")
    if shock_recovery_days < 1 or not 0.0 <= shock_multiplier <= 1.0:
        raise ValueError("波动率突增恢复天数必须为正数，降仓系数必须位于0到1之间")
    if max_target_exposure is not None and not 0.0 <= max_target_exposure <= 1.0:
        raise ValueError("仓位上限必须位于0到1之间")
    if not 0.0 <= min_target_exposure <= 1.0:
        raise ValueError("仓位下限必须位于0到1之间")
    if max_target_exposure is not None and min_target_exposure > max_target_exposure:
        raise ValueError("仓位下限不能高于仓位上限")
    if max_exposure_increase is not None and not 0.0 < max_exposure_increase <= 1.0:
        raise ValueError("每日最大增仓比例必须大于0且不超过1")
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
    # 上一次实际调仓完成时的净值，按当次成交价格重估；用于计算完整的调仓间收益。
    last_rebalance_nav: float | None = None
    shock_active = False
    shock_recovery_count = 0

    for day in dates:
        execution_values = {x: holdings[x] * (_execution_base_price(data, x, day, execution_price_mode) or 0.0) for x in holdings}
        nav_open = cash + sum(execution_values.values())
        pre_holdings = dict(holdings)
        pre_exposure = sum(execution_values.values()) / nav_open if nav_open else 0.0
        trade_start = len(trades)
        target_exposure = 1.0
        target_weights = pd.Series(1.0 / len(pending), index=pending, dtype=float) if pending else pd.Series(dtype=float)
        if pending and target_volatility is not None:
            returns = pd.concat([data[x]["close"].pct_change().rename(x) for x in pending], axis=1)
            returns = returns.loc[returns.index < day].tail(volatility_lookback)
            covariance = _annualized_covariance(returns, volatility_method, ewma_half_life)
            weight_array = target_weights.reindex(covariance.index).fillna(0).to_numpy()
            variance = float(weight_array @ covariance.fillna(0).to_numpy() @ weight_array)
            current_vol = np.sqrt(max(variance, 0.0))
            if np.isfinite(current_vol) and current_vol > 0:
                target_exposure = min(1.0, target_volatility / current_vol)
        if pending and volatility_shock_enabled:
            short_vol = portfolio_volatility(data, pending, day, lookback=shock_short_window)
            long_vol = portfolio_volatility(data, pending, day, lookback=shock_long_window)
            shock_ratio = short_vol / long_vol if np.isfinite(short_vol) and np.isfinite(long_vol) and long_vol > 0 else np.nan
            if np.isfinite(shock_ratio):
                if shock_active:
                    if shock_ratio <= shock_recovery_ratio:
                        shock_recovery_count += 1
                        if shock_recovery_count >= shock_recovery_days:
                            shock_active = False
                            shock_recovery_count = 0
                    else:
                        shock_recovery_count = 0
                elif shock_ratio >= shock_trigger_ratio:
                    shock_active = True
                    shock_recovery_count = 0
            if shock_active:
                target_exposure *= shock_multiplier
        if pending and trend_scaling_method:
            target_exposure *= single_asset_trend_scale(
                data,
                pending,
                day,
                trend_scaling_method,
                trend_weak_multiplier,
            )
        raw_target_exposure_for_recovery = target_exposure
        if pending and target_exposure > 0:
            if max_target_exposure is not None:
                target_exposure = min(target_exposure, max_target_exposure)
            target_exposure = max(target_exposure, min_target_exposure)
        # 只限制同一持仓下的恢复速度；风险上升时仍允许立即减仓。
        if (
            pending
            and holdings
            and set(pending) == set(holdings)
            and max_exposure_increase is not None
            and target_exposure > pre_exposure
        ):
            target_exposure = min(target_exposure, pre_exposure + max_exposure_increase)
        target_weights *= target_exposure
        desired_values = nav_open * target_weights
        if pending and holdings and set(pending) == set(holdings) and rebalance_band > 0:
            current_weights = pd.Series(execution_values) / nav_open if nav_open else pd.Series(dtype=float)
            union = current_weights.index.union(target_weights.index)
            gap = (current_weights.reindex(union).fillna(0) - target_weights.reindex(union).fillna(0)).abs().max()
            cash_gap = abs(cash / nav_open - (1.0 - target_weights.sum())) if nav_open else 0.0
            recovery_triggered = (
                recovery_uses_raw_target_for_band
                and max_exposure_increase is not None
                and raw_target_exposure_for_recovery > pre_exposure
                and raw_target_exposure_for_recovery - pre_exposure > rebalance_band
            )
            if max(float(gap), float(cash_gap)) <= rebalance_band and not recovery_triggered:
                desired_values = pd.Series(execution_values).reindex(pending).fillna(0.0)

        def sell(symbol: str, shares: float, reason: str, trade_day: pd.Timestamp = day) -> None:
            nonlocal cash
            price = _execution_base_price(data, symbol, trade_day, execution_price_mode)
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
            price = _execution_base_price(data, symbol, trade_day, execution_price_mode)
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
        elif all(_execution_base_price(data, symbol, day, execution_price_mode) is not None for symbol in pending):
            for symbol in list(holdings):
                if symbol not in pending:
                    sell(symbol, holdings[symbol], "调仓替换")
            for symbol in list(holdings):
                price = _execution_base_price(data, symbol, day, execution_price_mode)
                target = float(desired_values.get(symbol, 0.0))
                if price and holdings[symbol] * price > target:
                    sell(symbol, holdings[symbol] - target / price, "风险再平衡")
            for symbol in pending:
                price = _execution_base_price(data, symbol, day, execution_price_mode)
                target = float(desired_values.get(symbol, 0.0))
                current_value = holdings.get(symbol, 0.0) * (price or 0.0)
                if price and current_value < target:
                    buy(symbol, (target - current_value) / price, "目标仓位")

        day_trades = trades[trade_start:]
        event_row: dict[str, object] | None = None
        if day_trades:
            post_execution_values = {x: holdings[x] * (_execution_base_price(data, x, day, execution_price_mode) or 0.0) for x in holdings}
            nav_after_trade = cash + sum(post_execution_values.values())
            post_exposure = sum(post_execution_values.values()) / nav_after_trade if nav_after_trade else 0.0
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
            allowed_symbols = single_asset_trend_symbols(data, signal_day, trend_filter) if trend_filter else None
            selected, reason, score_values = _select_targets(
                scores,
                signal_day,
                list(holdings),
                top_n=top_n,
                buffer=buffer,
                cutoff=cutoff,
                allowed_symbols=allowed_symbols,
            )
            if selected != pending:
                pending_signal_day = signal_day.date()
                pending_reason = reason
                pending_scores = score_values
            pending = selected
        if event_row is not None:
            event_row["调仓后收盘净值"] = nav_close
            # 收益区间的起点应为上一次调仓完成后的执行价净值，而不是
            # 上一次调仓日收盘后的净值。这样会纳入调仓完成后至当日收盘的持仓收益。
            last_rebalance_nav = float(event_row["调仓后净值"])

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


def fetch_symbol_with_pandadata(symbol: str, username: str, password: str, start: date, end: date) -> pd.DataFrame:
    """Fetch one additional fund/ETF symbol for the current Streamlit session."""
    try:
        import panda_data
    except ImportError as exc:
        raise RuntimeError("未安装 panda_data，请在 requirements.txt 中安装后重试") from exc
    if not username or not password:
        raise RuntimeError("请先配置 PANDA_DATA_USERNAME 和 PANDA_DATA_PASSWORD")
    normalized_symbol = normalize_symbol_input(symbol)
    if normalized_symbol is None:
        raise ValueError("无法识别代码，请输入6位代码，或使用 SHSE.518880 / SZSE.159915 格式")
    _prepare_pandadata_runtime()
    panda_data.init_token(username=username, password=password)
    prefix, code = normalized_symbol.split(".")
    native = f"{code}.{'SH' if prefix == 'SHSE' else 'SZ'}"
    frames: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.Timedelta(days=364).to_pytimedelta(), end)
        frame = panda_data.get_fund_daily_pre(
            start_date=chunk_start.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
            symbol=native,
            fields=[],
        )
        if frame is not None and not frame.empty:
            frames.append(frame)
        chunk_start = chunk_end + pd.Timedelta(days=1).to_pytimedelta()
    if not frames:
        raise RuntimeError(f"PandaData 未返回 {normalized_symbol} 的日线数据，请检查代码、权限或上市日期")
    return _normalize_market_frame(pd.concat(frames, axis=0), normalized_symbol)


def refresh_with_pandadata(
    data_dir: Path,
    username: str,
    password: str,
    start: date,
    end: date,
    symbols: Sequence[str] | None = None,
) -> tuple[int, str]:
    """Update the built-in snapshot, optionally restricted to given symbols."""
    requested = tuple(DEFAULT_SYMBOLS if symbols is None else symbols)
    updated = 0
    for symbol in requested:
        existing_path = data_dir / f"{symbol}.parquet"
        existing = load_data(data_dir, symbols=[symbol])[symbol] if existing_path.exists() else pd.DataFrame()
        fetch_start = existing.index.max().date() + pd.Timedelta(days=1).to_pytimedelta() if not existing.empty else start
        if fetch_start > end:
            continue
        frame = fetch_symbol_with_pandadata(symbol, username, password, fetch_start, end)
        combined = pd.concat([existing, frame], axis=0) if not existing.empty else frame
        _normalize_market_frame(combined, symbol).to_parquet(existing_path)
        updated += 1
    return updated, end.isoformat()
