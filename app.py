"""Streamlit interface for the standalone KingMomentum ETF/LOF project."""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kingmomentum_core import ASSETS, coverage_table, latest_signal, load_data, portfolio_volatility, position_management_snapshot, refresh_with_pandadata, score_panel, backtest


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
EARLIEST_BACKTEST_START = date(2017, 8, 1)
POSITION_MODES = ["均衡仓位（目标波动率20%）", "防守仓位（目标波动率15%）", "原始满仓/现金"]
POSITION_MODE_OPTIONS = [*POSITION_MODES, "自定义"]


@st.cache_data(show_spinner=False)
def cached_data() -> dict:
    return load_data(DATA_DIR)


@st.cache_data(show_spinner=False)
def cached_scores(data: dict) -> object:
    return score_panel(data, window=25)


def pct(value: float | None) -> str:
    return "—" if value is None or value != value else f"{value:.2%}"


def format_percentage_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(pct)
    return formatted


def trade_detail_text(frame: pd.DataFrame, action: str) -> str:
    rows = frame[frame["动作"] == action]
    return "；".join(
        f"{row['名称']}（{row['代码']}） @ {float(row['成交价格']):.4f}，金额 {float(row['金额']):,.2f}"
        for _, row in rows.iterrows()
    )


def build_rebalance_summary(rebalances: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Build one row per rebalance event, with buy/sell details aggregated by date."""
    summary = rebalances.copy()
    if trades.empty or summary.empty:
        summary["卖出明细"] = ""
        summary["买入明细"] = ""
        return summary
    detail_rows = []
    for trade_date, group in trades.groupby("日期", sort=False):
        detail_rows.append(
            {
                "执行日期": trade_date,
                "卖出明细": trade_detail_text(group, "卖出"),
                "买入明细": trade_detail_text(group, "买入"),
            }
        )
    return summary.merge(pd.DataFrame(detail_rows), on="执行日期", how="left").fillna({"卖出明细": "", "买入明细": ""})


def resolve_symbol_input(value: str) -> str | None:
    normalized = value.strip().upper()
    if not normalized:
        return None
    if "." in normalized:
        return normalized if normalized in ASSETS else None
    code = normalized.zfill(6)
    matches = [symbol for symbol in ASSETS if symbol.endswith(f".{code}")]
    return matches[0] if len(matches) == 1 else None


def parse_date_text(value: str) -> date | None:
    """Parse a keyboard-entered date in common formats."""
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def next_trading_day(data: dict, day: pd.Timestamp) -> date | None:
    """Return the next available market day after a signal day."""
    dates = pd.DatetimeIndex(sorted(set().union(*[set(frame.index) for frame in data.values()])))
    future = dates[dates > pd.Timestamp(day)]
    return future[0].date() if len(future) else None


def mode_settings(mode: str) -> dict[str, float | None]:
    return {
        "原始满仓/现金": {"target_volatility": None, "rebalance_band": 0.0},
        "防守仓位（目标波动率15%）": {"target_volatility": 0.15, "rebalance_band": 0.10},
        "均衡仓位（目标波动率20%）": {"target_volatility": 0.20, "rebalance_band": 0.10},
    }[mode]


def position_mode_controls(container, label: str, key_prefix: str) -> tuple[str, dict[str, float | None], str]:
    mode = container.selectbox(label, POSITION_MODE_OPTIONS, index=0, key=f"{key_prefix}_mode")
    if mode != "自定义":
        return mode, mode_settings(mode), mode
    target_volatility_pct = container.number_input(
        "自定义目标波动率（%）",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=0.5,
        format="%.1f",
        key=f"{key_prefix}_target_volatility",
    )
    rebalance_band_pct = container.number_input(
        "自定义仓位调整带（%）",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        format="%.1f",
        key=f"{key_prefix}_rebalance_band",
    )
    settings = {
        "target_volatility": target_volatility_pct / 100.0,
        "rebalance_band": rebalance_band_pct / 100.0,
    }
    label = f"自定义（目标波动率{target_volatility_pct:.1f}%，调整带{rebalance_band_pct:.1f}%）"
    return mode, settings, label


def run_strategy(data: dict, start: date, end: date, mode: str, settings: dict[str, float | None] | None = None):
    settings = mode_settings(mode) if settings is None else settings
    return backtest(data, cached_scores(data), start=start, end=end, **settings)


def nav_chart(nav):
    frame = nav.copy()
    frame["日期"] = frame["日期"].astype(str)
    base = float(frame["净值"].iloc[0])
    frame["归一化净值"] = frame["净值"] / base
    frame["回撤"] = frame["净值"] / frame["净值"].cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["日期"], y=frame["归一化净值"], name="策略净值", mode="lines", line={"color": "#1769aa", "width": 2}))
    fig.update_layout(title="策略净值曲线", yaxis_title="净值（起点=1）", hovermode="x unified", height=430, margin={"l": 40, "r": 20, "t": 55, "b": 35})
    return fig


def drawdown_chart(nav):
    frame = nav.copy()
    frame["日期"] = frame["日期"].astype(str)
    frame["回撤"] = frame["净值"] / frame["净值"].cummax() - 1.0
    fig = go.Figure(go.Scatter(x=frame["日期"], y=frame["回撤"], name="回撤", mode="lines", fill="tozeroy", line={"color": "#d62728"}))
    fig.update_layout(title="回撤曲线（回撤越深位置越高）", yaxis={"autorange": "reversed", "tickformat": ".0%"}, hovermode="x unified", height=300, margin={"l": 40, "r": 20, "t": 55, "b": 35})
    return fig


def period_returns(nav: pd.DataFrame, frequency: str) -> pd.Series:
    """Calculate period returns, including the partial first period in the backtest."""
    daily = pd.Series(nav["净值"].to_numpy(dtype=float), index=pd.to_datetime(nav["日期"])).sort_index()
    period_end = daily.resample(frequency).last().dropna()
    returns = period_end / period_end.shift(1) - 1.0
    if not period_end.empty:
        returns.iloc[0] = period_end.iloc[0] / daily.iloc[0] - 1.0
    return returns


def return_limit(series_list: list[pd.Series]) -> float:
    values = [abs(float(value)) for series in series_list for value in series.dropna()]
    return max(max(values, default=0.01), 0.01)


def return_color(value: float, limit: float) -> str:
    if pd.isna(value):
        return "#e5e7eb"
    ratio = min(abs(float(value)) / limit, 1.0)
    base = (198, 40, 40) if value >= 0 else (0, 105, 92)
    rgb = tuple(round(255 * (1.0 - ratio) + channel * ratio) for channel in base)
    return f"rgb{rgb}"


def monthly_heatmap(nav: pd.DataFrame, monthly: pd.Series) -> go.Figure:
    table = monthly.to_frame("收益率")
    table["年份"] = table.index.year
    table["月份"] = table.index.month
    matrix = table.pivot(index="年份", columns="月份", values="收益率").reindex(columns=range(1, 13))
    matrix = matrix.sort_index()
    month_labels = [f"{month}月" for month in range(1, 13)]
    text = [[pct(value) for value in row] for row in matrix.to_numpy()]
    limit = return_limit([monthly])
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=month_labels,
            y=[str(year) for year in matrix.index],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale=[[0.0, "#00695c"], [0.5, "#ffffff"], [1.0, "#c62828"]],
            zmin=-limit,
            zmax=limit,
            zmid=0.0,
            colorbar={"title": "收益率", "tickformat": ".0%"},
            hovertemplate="%{y}年%{x}：%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title="月度收益热力图", xaxis_title="月份", yaxis_title="年份", height=430, margin={"l": 45, "r": 20, "t": 55, "b": 35})
    return fig


def return_bar_chart(series: pd.Series, title: str, labels: list[str]) -> go.Figure:
    limit = return_limit([series])
    values = [float(value) for value in series.to_numpy()]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[return_color(value, limit) for value in values],
            text=[pct(value) for value in values],
            textposition="outside",
            hovertemplate="%{x}：%{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title=title, yaxis_title="收益率", yaxis_tickformat=".0%", height=360, margin={"l": 45, "r": 20, "t": 55, "b": 45})
    fig.add_hline(y=0.0, line_width=1, line_color="#6b7280")
    return fig


def cycle_return_section(nav: pd.DataFrame) -> None:
    monthly = period_returns(nav, "ME")
    annual = period_returns(nav, "YE")
    quarterly = period_returns(nav, "QE")
    st.subheader("周期收益分析")
    st.caption("颜色说明：盈利越高越红，亏损越严重越绿；首个周期从本次回测起始净值计算，可能是不完整周期。")
    monthly_tab, quarterly_tab, annual_tab = st.tabs(["月度", "季度", "年度"])
    with monthly_tab:
        st.plotly_chart(monthly_heatmap(nav, monthly), use_container_width=True)
    with quarterly_tab:
        quarter_labels = [str(index.to_period("Q")) for index in quarterly.index]
        st.plotly_chart(return_bar_chart(quarterly, "季度收益", quarter_labels), use_container_width=True)
    with annual_tab:
        annual_labels = [str(index.year) for index in annual.index]
        st.plotly_chart(return_bar_chart(annual, "年度收益", annual_labels), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="KingMomentum ETF轮动", page_icon="📈", layout="wide")
    st.title("KingMomentum ETF / LOF 轮动策略")
    st.caption("25个交易日对数价格加权线性回归 · 收盘计算信号 · 下一交易日开盘执行")
    try:
        all_data = cached_data()
    except Exception as exc:
        st.error(f"数据读取失败：{exc}")
        st.stop()

    with st.sidebar:
        st.header("策略与数据")
        page = st.radio("页面", ["回测", "最新持仓", "策略说明"], index=0)
        if "selected_symbols" not in st.session_state:
            st.session_state.selected_symbols = list(ASSETS)
        selected = st.multiselect(
            "标的池（代码 · 名称）",
            list(ASSETS),
            default=st.session_state.selected_symbols,
            format_func=lambda x: f"{x} · {ASSETS[x]}",
        )
        st.session_state.selected_symbols = selected
        with st.form("add_symbol_form", clear_on_submit=True):
            input_code = st.text_input("输入代码加入标的池", placeholder="例如 518880")
            add_symbol = st.form_submit_button("加入标的池")
        if add_symbol:
            symbol = resolve_symbol_input(input_code)
            if symbol is None:
                st.error("该代码不在当前9个标的数据快照中，请先补充对应数据文件。")
            elif symbol not in st.session_state.selected_symbols:
                st.session_state.selected_symbols.append(symbol)
                st.rerun()
            else:
                st.info("该标的已经在当前标的池中。")
        selected = st.session_state.selected_symbols
        if not selected:
            st.warning("请至少选择一个标的")
            st.stop()
        data = {symbol: all_data[symbol] for symbol in selected}
        data_start = min(frame.index.min().date() for frame in data.values())
        common_end = min(frame.index.max().date() for frame in data.values())
        data_signature = tuple((symbol, frame.index.min(), frame.index.max(), len(frame)) for symbol, frame in sorted(data.items()))
        st.info(f"当前标的数据范围：{data_start} 至 {common_end}\n\n回测最早允许日期：{EARLIEST_BACKTEST_START}。早期尚未上市的标的暂不参与排名。")
        st.divider()
        st.markdown("**研究参数**")
        st.markdown("回归窗口：25日  ·  过热阈值：500  ·  换仓缓冲：5  ·  单边成本：0.05%（万分之5）")
        st.markdown("仓位默认：Top-1、目标波动率20%、10%仓位调整带")

    scores = cached_scores(data)
    if page == "回测":
        with st.sidebar:
            with st.form("backtest_form"):
                mode, settings, mode_label = position_mode_controls(st, "仓位模式", "backtest_position")
                quick_range = st.radio("快捷日期", ["自定义", "最近1年", "最近2年", "最近3年", "最近5年"], index=2)
                years = {"最近1年": 1, "最近2年": 2, "最近3年": 3, "最近5年": 5}
                quick_start = max(EARLIEST_BACKTEST_START, (pd.Timestamp(common_end) - pd.DateOffset(years=years[quick_range])).date()) if quick_range != "自定义" else EARLIEST_BACKTEST_START
                if quick_range == "自定义":
                    st.caption("可直接键盘输入日期，支持 YYYY-MM-DD、YYYY/MM/DD 或 YYYYMMDD")
                    date_input_cols = st.columns(2)
                    with date_input_cols[0]:
                        custom_start_text = st.text_input(
                            "开始日期",
                            value=str(EARLIEST_BACKTEST_START),
                            key="custom_backtest_start_text",
                        )
                    with date_input_cols[1]:
                        custom_end_text = st.text_input(
                            "结束日期",
                            value=str(common_end),
                            key="custom_backtest_end_text",
                        )
                else:
                    st.caption(f"快捷回测区间：{quick_start} 至 {common_end}")
                    custom_start_text = ""
                    custom_end_text = ""
                start_backtest = st.form_submit_button("开始回测", type="primary", use_container_width=True)

        if start_backtest:
            if quick_range == "自定义":
                start = parse_date_text(custom_start_text)
                end = parse_date_text(custom_end_text)
                if start is None or end is None:
                    st.error("日期格式无法识别，请输入 YYYY-MM-DD，例如 2024-08-25")
                    start = end = None
            else:
                start, end = quick_start, common_end
            if start is not None and end is not None:
                if start < EARLIEST_BACKTEST_START:
                    st.error(f"起始日期不得早于 {EARLIEST_BACKTEST_START}")
                elif start >= end:
                    st.error("起始日期必须早于结束日期")
                else:
                    st.session_state.backtest_request = {
                        "start": start,
                        "end": end,
                        "mode": mode,
                        "settings": settings,
                        "mode_label": mode_label,
                        "date_mode": quick_range,
                        "data_signature": data_signature,
                    }

        request = st.session_state.get("backtest_request")
        if request is not None and request.get("data_signature") != data_signature:
            st.session_state.pop("backtest_request", None)
            request = None
        if request is None:
            st.info("请设置回测参数后，点击左侧“开始回测”按钮。")
            st.stop()
        start = request["start"]
        end = request["end"]
        settings = request["settings"]
        metrics, nav, holdings, rebalances, trades = run_strategy(data, start, end, request["mode"], settings)
        st.subheader(f"回测结果：{request['mode_label']}")
        cards = [("累计收益", pct(metrics["累计收益"])), ("年化收益", pct(metrics["年化收益"])), ("最大回撤", pct(metrics["最大回撤"])), ("夏普系数", f"{metrics['夏普']:.2f}" if metrics["夏普"] == metrics["夏普"] else "—"), ("交易次数", f"{metrics['交易次数']:,}")]
        cols = st.columns(len(cards))
        for col, (label, value) in zip(cols, cards):
            col.metric(label, value)
        st.caption(f"回测区间：{start} 至 {end}。数据来自调整后日线快照；策略在首个信号日后于下一交易日开盘执行。")
        st.plotly_chart(nav_chart(nav), use_container_width=True)
        st.plotly_chart(drawdown_chart(nav), use_container_width=True)
        st.subheader("调仓记录")
        st.caption("调仓前组合收益率：上次实际调仓后收盘至本次调仓前开盘的组合收益率；调仓净值影响：本次成交前后按成交价计算的净值变化，主要反映手续费。成交价格为执行日开盘价。")
        rebalance_view = build_rebalance_summary(rebalances, trades)
        rebalance_view = rebalance_view.rename(columns={"上次调仓至本次调仓前收益率": "本次调仓前组合收益率", "原因": "调仓原因"})
        display_columns = [
            "信号日期",
            "执行日期",
            "调仓类型",
            "本次调仓前组合收益率",
            "当前持仓名称",
            "目标持仓名称",
            "目标总仓位",
            "调仓前总仓位",
            "调仓后总仓位",
            "仓位变化",
            "卖出明细",
            "买入明细",
            "调仓净值影响",
            "手续费估算",
            "成交笔数",
            "调仓原因",
        ]
        rebalance_view = rebalance_view.reindex(columns=display_columns)
        if not rebalance_view.empty:
            rebalance_view = rebalance_view.sort_values("执行日期", ascending=False)
        if rebalance_view.empty:
            st.info("该回测区间内没有发生调仓。")
        else:
            percentage_columns = [
                "本次调仓前组合收益率",
                "目标总仓位",
                "调仓前总仓位",
                "调仓后总仓位",
                "仓位变化",
                "调仓净值影响",
            ]
            signal_view = rebalance_view[rebalance_view["调仓类型"] == "信号调仓"]
            risk_view = rebalance_view[rebalance_view["调仓类型"] == "风险仓位再平衡"]
            st.markdown("#### 信号调仓")
            if signal_view.empty:
                st.info("该区间内没有发生信号调仓。")
            else:
                st.dataframe(format_percentage_columns(signal_view, percentage_columns), hide_index=True, use_container_width=True)
                st.download_button("下载信号调仓CSV", signal_view.to_csv(index=False).encode("utf-8-sig"), "kingmomentum_signal_rebalances.csv", "text/csv")
            st.markdown("#### 风险仓位再平衡")
            if mode == "原始满仓/现金":
                st.info("当前模式未启用目标波动率仓位管理，因此不会产生风险仓位再平衡记录。")
            elif risk_view.empty:
                st.info("该区间内没有触发风险仓位再平衡。")
            else:
                st.dataframe(format_percentage_columns(risk_view, percentage_columns), hide_index=True, use_container_width=True)
                st.download_button("下载风险仓位调仓CSV", risk_view.to_csv(index=False).encode("utf-8-sig"), "kingmomentum_risk_rebalances.csv", "text/csv")
            st.download_button("下载全部调仓CSV", rebalance_view.to_csv(index=False).encode("utf-8-sig"), "kingmomentum_rebalances.csv", "text/csv")
            st.download_button("下载逐笔成交明细CSV", trades.to_csv(index=False).encode("utf-8-sig"), "kingmomentum_trade_details.csv", "text/csv")
            st.caption("每一行代表一次调仓事件；卖出明细和买入明细中包含该次调仓的中文名称、代码、成交价格和成交金额，不再按买卖笔数重复展示仓位与收益率字段。")
        st.subheader("每日净值")
        st.dataframe(nav.sort_values("日期", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("下载净值CSV", nav.to_csv(index=False).encode("utf-8-sig"), "kingmomentum_nav.csv", "text/csv")
        st.subheader("每日持仓记录")
        st.dataframe(holdings.sort_values(["日期", "组合权重"], ascending=[False, False]), hide_index=True, use_container_width=True)
        st.download_button("下载每日持仓CSV", holdings.to_csv(index=False).encode("utf-8-sig"), "kingmomentum_holdings.csv", "text/csv")
        cycle_return_section(nav)

    elif page == "最新持仓":
        st.subheader("最新持仓与动量分数")
        latest_data_date = min(frame.index.max().date() for frame in data.values())
        saved_request = st.session_state.get("backtest_request")
        if saved_request is not None and saved_request.get("data_signature") != data_signature:
            st.session_state.pop("backtest_request", None)
            saved_request = None
        use_custom_backtest_end = saved_request is not None and saved_request.get("date_mode") == "自定义"
        analysis_start = saved_request["start"] if use_custom_backtest_end else EARLIEST_BACKTEST_START
        analysis_end = saved_request["end"] if use_custom_backtest_end else common_end
        if use_custom_backtest_end:
            st.write(
                f"当前数据最新日期：**{latest_data_date}**。已检测到最近一次自定义回测，"
                f"本页将展示该回测区间 **{analysis_start} 至 {analysis_end}** 内最后一个交易日的持仓。"
            )
        else:
            st.write(f"当前数据最新日期：**{latest_data_date}**。最新持仓使用各标的最近可用收盘数据计算。")
        latest_mode, latest_settings, latest_mode_label = position_mode_controls(st, "最新持仓仓位模式", "latest_position")
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("更新数据", type="primary"):
                username = st.secrets.get("PANDA_DATA_USERNAME", os.getenv("PANDA_DATA_USERNAME", ""))
                password = st.secrets.get("PANDA_DATA_PASSWORD", os.getenv("PANDA_DATA_PASSWORD", ""))
                try:
                    count, updated_to = refresh_with_pandadata(DATA_DIR, username, password, EARLIEST_BACKTEST_START, date.today())
                    st.success(f"已更新 {count} 个标的，目标日期：{updated_to}")
                    cached_data.clear()
                    cached_scores.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"更新失败：{exc}")
        with c2:
            st.caption("更新功能需要配置 PandaData 账号和可用SDK；未配置时，应用继续使用项目内置快照。建议先在本地验证更新后的数据覆盖与完整性。")
        score_day, score_table, selected_holdings, reason = latest_signal(data, scores, as_of=analysis_end)
        full_metrics, latest_nav, _, _, _ = backtest(data, scores, start=analysis_start, end=analysis_end, **latest_settings)
        holding_day = pd.Timestamp(latest_nav["日期"].iloc[-1]).date()
        latest_state = latest_nav.iloc[-1]
        current_holdings = [] if latest_state["持仓"] == "现金" else str(latest_state["持仓"]).split("|")
        current_exposure = float(latest_state["总仓位"])
        current_nav = float(latest_nav["净值"].iloc[-1])
        historical_peak = float(latest_nav["净值"].max())
        current_drawdown = current_nav / historical_peak - 1.0 if historical_peak > 0 else float("nan")
        peak_index = latest_nav["净值"].idxmax()
        peak_date = latest_nav.loc[peak_index, "日期"]
        recent_start = max(
            EARLIEST_BACKTEST_START,
            (pd.Timestamp(holding_day) - pd.DateOffset(years=2)).date(),
        )
        recent_metrics, _, _, _, _ = backtest(
            data,
            scores,
            start=recent_start,
            end=holding_day,
            **latest_settings,
        )
        next_execution_day = next_trading_day(data, pd.Timestamp(holding_day))
        volatility_day = (
            pd.Timestamp(next_execution_day)
            if next_execution_day is not None
            else pd.Timestamp(holding_day) + pd.Timedelta(days=1)
        )
        position_table = position_management_snapshot(
            data,
            score_day,
            selected_holdings,
            current_holdings,
            current_exposure,
            **latest_settings,
            volatility_day=volatility_day,
        )
        recommended_names = "、".join(ASSETS[x] for x in selected_holdings) if selected_holdings else "现金"
        recommended_codes = "、".join(selected_holdings) if selected_holdings else "—"
        st.subheader("最新建议持仓")
        st.metric("最新建议持仓", recommended_names)
        st.info(f"信号日期：{score_day.date()} · {latest_mode_label} · {reason} · 标的代码：{recommended_codes}")
        recommendation_status = str(position_table.iloc[0]["仓位判断"])
        if recommendation_status in {"需要按信号换仓", "需要增仓", "需要减仓", "转入现金"}:
            st.warning(f"⚠️ 仓位调整判断：需要调整（{recommendation_status}）")
        else:
            st.success(f"✅ 仓位调整判断：暂不需要调整（{recommendation_status}）")
        st.subheader("当前回撤")
        st.metric("当前回撤（相对历史峰值）", pct(current_drawdown))
        st.info(
            f"截至 {holding_day}，当前组合资产净值为 {current_nav:,.2f}；"
            f"历史最高资产净值为 {historical_peak:,.2f}，出现在 {peak_date}。"
            f"当前净值较该历史峰值回落 {abs(current_drawdown):.2%}。"
        )
        st.caption(f"当前回撤 =（当前组合资产净值 ÷ 历史最高资产净值）− 1；统计区间为 {analysis_start} 至 {holding_day}。")

        st.subheader("最近两年策略绩效")
        recent_cards = [
            ("累计收益", pct(recent_metrics["累计收益"])),
            ("年化收益", pct(recent_metrics["年化收益"])),
            ("最大回撤", pct(recent_metrics["最大回撤"])),
            ("年化波动", pct(recent_metrics["年化波动"])),
            ("夏普系数", f"{recent_metrics['夏普']:.2f}" if recent_metrics["夏普"] == recent_metrics["夏普"] else "—"),
            ("交易次数", f"{recent_metrics['交易次数']:,}"),
        ]
        recent_cols = st.columns(len(recent_cards))
        for col, (label, value) in zip(recent_cols, recent_cards):
            col.metric(label, value)
        st.caption(f"绩效区间：{recent_start} 至 {holding_day}。若数据不足两年，则从允许的最早日期开始计算。")

        st.subheader("全历史周期策略绩效")
        full_cards = [
            ("累计收益", pct(full_metrics["累计收益"])),
            ("年化收益", pct(full_metrics["年化收益"])),
            ("最大回撤", pct(full_metrics["最大回撤"])),
            ("年化波动", pct(full_metrics["年化波动"])),
            ("夏普系数", f"{full_metrics['夏普']:.2f}" if full_metrics["夏普"] == full_metrics["夏普"] else "—"),
            ("交易次数", f"{full_metrics['交易次数']:,}"),
        ]
        full_cols = st.columns(len(full_cards))
        for col, (label, value) in zip(full_cols, full_cards):
            col.metric(label, value)
        st.caption(f"绩效区间：{analysis_start} 至 {holding_day}。全历史绩效用于观察当前分析区间内的完整风险收益表现。")

        st.subheader("仓位管理")
        position_row = position_table.iloc[0]
        metric_values = [
            ("组合估计年化波动率", pct(position_row["组合估计年化波动率"])),
            ("目标总仓位", pct(position_row["目标总仓位"])),
            ("目标现金比例", pct(position_row["目标现金比例"])),
            ("当前策略总仓位", pct(position_row["当前策略总仓位"])),
            ("仓位判断", position_row["仓位判断"]),
        ]
        metric_cols = st.columns(len(metric_values))
        for col, (label, value) in zip(metric_cols, metric_values):
            col.metric(label, value)
        st.caption(f"目标波动率：{pct(position_row['目标波动率'])} · 仓位调整带：{pct(position_row['仓位调整带'])}。当前策略仓位为按内置数据从 {EARLIEST_BACKTEST_START} 回放至最新信号日的结果，不代表券商账户实时持仓。")
        st.subheader("仓位调整提示")
        position_status = str(position_row["仓位判断"])
        target_exposure = float(position_row["目标总仓位"])
        current_exposure = float(position_row["当前策略总仓位"])
        exposure_gap = abs(current_exposure - target_exposure)
        execution_day = next_execution_day
        execution_label = str(execution_day) if execution_day is not None else "待数据更新确认"
        current_names = str(position_row["当前策略持仓名称"])
        target_names = str(position_row["推荐持仓名称"])
        if position_status == "需要按信号换仓":
            action_text = (
                f"需要进行信号调仓：当前持仓为 {current_names}，建议持仓为 {target_names}。"
                f"目标总仓位为 {pct(target_exposure)}，请在下一交易日（{execution_label}）开盘执行。"
            )
            st.warning(f"⚠️ {action_text}")
        elif position_status == "需要增仓":
            action_text = (
                f"需要增仓：当前总仓位 {pct(current_exposure)}，目标总仓位 {pct(target_exposure)}，"
                f"应增加约 {pct(target_exposure - current_exposure)}。请在下一交易日（{execution_label}）开盘执行。"
            )
            st.warning(f"⚠️ {action_text}")
        elif position_status == "需要减仓":
            action_text = (
                f"需要减仓：当前总仓位 {pct(current_exposure)}，目标总仓位 {pct(target_exposure)}，"
                f"应减少约 {pct(current_exposure - target_exposure)}。请在下一交易日（{execution_label}）开盘执行。"
            )
            st.warning(f"⚠️ {action_text}")
        elif position_status == "转入现金":
            action_text = f"需要转入现金：当前没有有效的正分标的。请在下一交易日（{execution_label}）开盘卖出当前持仓。"
            st.warning(f"⚠️ {action_text}")
        else:
            st.success(
                f"✅ 当前无需进行仓位调整：当前总仓位 {pct(current_exposure)}，目标总仓位 {pct(target_exposure)}，"
                f"仓位偏离 {pct(exposure_gap)}，未超过调整带 {pct(float(position_row['仓位调整带']))}。"
            )
        st.caption("判断顺序：先判断推荐持仓是否变化；若持仓未变，再使用当前收盘后的最近20日波动率判断下一交易日仓位偏离是否超过调整带。信号和仓位调整均在下一交易日开盘执行，页面不会自动下单。")
        position_display = position_table.copy()
        for column in ["组合估计年化波动率", "目标波动率", "目标总仓位", "目标现金比例", "当前策略总仓位", "仓位偏离", "仓位调整带"]:
            position_display[column] = position_display[column].map(pct)
        st.dataframe(position_display, hide_index=True, use_container_width=True)
        st.caption("判断规则：推荐持仓变化优先触发信号换仓；推荐持仓不变时，只有当前仓位与目标仓位的偏离超过调整带，才触发风险仓位再平衡。")
        st.dataframe(score_table.reset_index().rename(columns={"index": "代码"}), hide_index=True, use_container_width=True)
        st.subheader("标的数据覆盖")
        st.dataframe(coverage_table(data), hide_index=True, use_container_width=True)

    else:
        st.subheader("策略说明")
        st.markdown("""
本项目使用过去 **25个交易日**的对数收盘价进行二次加权线性回归。趋势斜率反映方向，拟合优度 R² 反映趋势质量，二者共同形成动量分数。

### 一、策略基础原理

策略属于相对强弱动量轮动策略。每天收盘后，对标的池中的 ETF/LOF 分别计算趋势动量分数，再从有效标的中选择排名靠前者持有。策略的核心假设是：

1. 近期价格趋势具有一定延续性；
2. 对数价格的线性斜率可以近似描述复合收益率方向和速度；
3. R² 可以过滤斜率较高但路径杂乱、趋势质量较差的标的；
4. 当趋势失效、标的过热或所有标的都没有正动量时，降低风险或持有现金。

策略使用复权后的收盘价计算信号，使用下一交易日开盘价执行交易，因此不会使用当日收盘后才知道的价格去完成当日交易。

### 二、动量分数的公式

设回归窗口为 `W=25`，最近25个交易日编号为：

```text
j = 0, 1, 2, ..., 24
```

对每个标的，先取复权收盘价 `P_j`，并转换为对数价格：

```text
y_j = ln(P_j)
```

时间权重为：

```text
u_j = 1 + (j / 24)^2
```

越接近当前的交易日，权重越高。实际用于拟合的权重为：

```text
v_j = u_j^2 = [1 + (j / 24)^2]^2
```

对 `y_j` 进行加权线性回归：

```text
ŷ_j = β₀ + β₁j
```

其中斜率 `β₁` 的计算公式为：

```text
β₁ = Σ[v_j (j - j̄_v)(y_j - ȳ_v)] / Σ[v_j (j - j̄_v)^2]
```

加权均值为：

```text
j̄_v = Σ(v_j j) / Σv_j
ȳ_v = Σ(v_j y_j) / Σv_j
```

随后使用原始时间权重 `u_j` 计算拟合优度：

```text
R² = 1 - Σ[u_j (y_j - ŷ_j)^2] / Σ[u_j (y_j - ȳ_u)^2]
```

其中：

```text
ȳ_u = Σ(u_j y_j) / Σu_j
```

当分母为0时，`R²`按0处理；最终将 `R²`限制在 `[0, 1]` 区间。

斜率被年化为趋势收益率：

```text
Trend = exp(β₁ × 252) - 1
```

最终动量分数为：

```text
MomentumScore = [exp(β₁ × 252) - 1] × R² × 100
```

因此，一个标的只有在趋势方向较强、并且趋势路径较平滑时，才能获得较高分数。`100` 是展示缩放系数，不是百分比单位；例如分数 `240.73` 表示模型分数为240.73，并不等于240.73%的预期收益。

### 三、交易规则与熔断标准

交易规则：

- 每日收盘后计算所有标的分数；
- 分数小于等于0的标的不参与持仓；
- 分数超过500视为过热；
- 当前第一名未过热时继续持有；
- 新第一名只有超过换仓缓冲才替换；
- 如果所有分数都不为正，则持有现金；
- 信号在下一交易日开盘执行。

这里的“熔断”是策略内部的风险过滤标准，不是交易所的涨跌停熔断机制。

#### 1. 非正分过滤

```text
Score ≤ 0  →  不参与有效候选排名
```

如果所有标的分数都小于等于0：

```text
目标持仓 = 现金
```

现金切换在下一交易日开盘完成。

#### 2. 过热阈值

当前策略设置过热阈值为：

```text
Score > 500  →  过热，不进入有效候选
```

有效候选的范围是：

```text
0 < Score ≤ 500
```

如果所有正分标的都超过500，当前实现会保留已有持仓，不强制清仓；这是为了避免“所有标的暂时过热”时产生不必要的现金切换。若当前没有持仓，则没有有效候选可建立新仓位。

#### 3. 换仓缓冲

对有效候选分数进行0到100的横截面归一化：

```text
ScoreNormᵢ = (Scoreᵢ - ScoreMin) / (ScoreMax - ScoreMin) × 100
```

当 `ScoreMax = ScoreMin` 时，所有候选的归一化分数取50。当前换仓缓冲为5：

```text
新第一名的 ScoreNorm - 当前持仓的 ScoreNorm > 5
```

只有超过5，才执行新第一名替换当前持仓；否则继续持有原标的。标的过热、失效或所有分数转负时，不受普通换仓缓冲阻止。

默认风险仓位采用20%目标波动率和10%仓位调整带。调整带用于避免每日风险估计的小幅变化造成频繁交易，不代表固定持有周期。

### 四、波动率与目标仓位的计算

波动率可以理解为资产收益上下波动的幅度。目标波动率是策略希望组合长期维持的风险水平，而不是收益率目标。本项目在每个执行日开盘前，用“下一阶段准备持有的目标组合”过去20个交易日的收盘收益计算风险，而不是机械地只使用当前账户中的标的。

具体来说，先对目标组合中每个标的计算收盘到收盘的收益率。对标的 `i`：

```text
rᵢ,t = Pᵢ,t / Pᵢ,t-1 - 1
```

取执行日前最近20个收益观测，计算样本协方差矩阵，并乘以252进行年化：

```text
Σₐ = 252 × Cov(rₜ₋₁₉, ..., rₜ)
```

如果目标组合有 `N` 个标的，当前策略使用等权基础权重：

```text
wᵢ = 1 / N
```

组合估计年化波动率为：

```text
σₚ = sqrt(wᵀ Σₐ w)
```

如果目标组合只有一个标的，上式退化为该标的过去20个交易日收益波动率的年化值。如果是多个标的，协方差和相关性会影响组合波动率，不能简单地把各标的波动率做算术平均。

然后按以下方式计算总仓位：

```text
目标总仓位 = min(100%, 目标波动率 ÷ 当前估计波动率)
```

例如目标波动率为20%，当前持仓估计年化波动率为40%，则目标总仓位约为50%，剩余资金持有现金。如果当前波动率只有10%，公式会得到200%，但本策略不加杠杆，因此总仓位最高为100%。

目标波动率较低，通常意味着更低的回撤和收益；目标波动率较高，通常意味着更高的收益潜力和更大的回撤风险。它不是风险消除工具，快速下跌仍可能造成实际损失。

如果当前没有持有新目标标的，仍然可以计算仓位：直接使用新目标标的过去20个交易日的波动率决定首次买入金额，不需要先持有一段时间。例如，新标的估计波动率为60%，目标波动率为20%，首次建仓目标总仓位约为33.33%，剩余资金保留为现金。

如果当前已经持有标的，且没有发生换标的信号，则目标标的保持不变，但每天重新计算该标的最新20日波动率。若当前仓位与新目标仓位的偏离不超过10个百分点，则不交易；超过10个百分点时，下一交易日开盘增仓或减仓。也就是说，仓位管理可以在持有同一标的期间单独触发，而不需要等待换标的。

### 五、仓位调整带与两种调整情形

如果每天都根据估计波动率微调仓位，短期噪声会产生大量交易。仓位调整带就是允许实际仓位在目标仓位附近小幅偏离，只有偏离超过阈值才执行再平衡。

本项目的10%调整带表示：当实际总仓位或标的权重与目标权重的最大偏离不超过10个百分点时，暂不调整；超过10个百分点时，才在下一交易日开盘调整。形式化表示为：

```text
max( |wᵢ,current - wᵢ,target|,
     |CashCurrent - CashTarget| ) > 10%
```

才触发风险仓位再平衡。标的发生过热退出、现金切换或新第一名超过换仓缓冲时，仍按交易信号执行，不会被调整带阻止。

#### 情形A：没有持有新目标标的

如果新信号要求从标的A切换至此前没有持有的标的B，仓位计算使用标的B（或新的目标组合）过去20个交易日的波动率：

```text
TargetExposure = min(100%, 20% / σ_B)
TargetValue_B = NAV_open × TargetExposure
```

先卖出旧标的，再按目标金额买入新标的，剩余金额保留为现金。新标的不需要先持有一段时间才能计算仓位。

#### 情形B：继续持有原标的

如果排名和持仓标的没有变化，策略每天仍会重新计算该标的（或目标组合）的最新20日波动率：

```text
CurrentExposure = 持仓开盘市值 / 调仓前开盘净值
Gap = CurrentExposure - TargetExposure
```

当 `|Gap| ≤ 10%` 时不交易；当 `Gap > 10%` 时减仓；当 `Gap < -10%` 时增仓。该调整不需要换标的信号。

因此，仓位调整带不是固定持仓周期，也不是止损线。它主要用于降低仓位管理的换手和交易成本。

### 六、调仓数据如何解读

调仓记录中的“调仓类型”分为两类：

- **信号调仓**：动量排名、过热状态或现金条件发生变化，导致推荐标的发生变化；
- **风险仓位再平衡**：推荐标的没有变化，但目标波动率计算出的目标总仓位与当前仓位偏离超过调整带。

记录中的“目标总仓位”是风险模型希望持有的资产比例；“调仓前总仓位”和“调仓后总仓位”用于核对仓位调整是否达到目标；“仓位变化”表示本次实际增加或减少的资产比例。

调仓金额和净值字段的计算方式为：

```text
NAV_open = Cash + Σ(qᵢ × Pᵢ,open)
调仓前总仓位 = Σ(qᵢ × Pᵢ,open) / NAV_open
```

成交后，扣除买卖手续费并按同一开盘价重估：

```text
NAV_after = Cash_after + Σ(qᵢ,after × Pᵢ,open)
调仓后总仓位 = Σ(qᵢ,after × Pᵢ,open) / NAV_after
调仓净值影响 = NAV_after / NAV_open - 1
```

当前单边手续费为 `0.05%`：

```text
手续费估算 = Σ成交金额 × 0.0005
```

“信号调仓”表示目标标的发生变化；“风险仓位再平衡”表示目标标的没有变化，但波动率模型导致仓位偏离超过调整带。每行调仓汇总记录代表一次实际执行事件，买卖名称、价格和金额会分别列在卖出明细和买入明细中。

“本次调仓前组合收益率”反映从上次实际调仓后到本次调仓前，组合已经取得的收益或亏损；“调仓净值影响”反映按成交价格完成本次交易后净值的即时变化，主要是手续费影响。因此，调仓记录既能看到调仓发生前组合表现，也能看到仓位管理是否导致了增仓、减仓以及交易成本。
""")
        st.subheader("结合当前数据的实际计算示例")
        explanation_settings = mode_settings("均衡仓位（目标波动率20%）")
        explanation_day, _, explanation_selected, _ = latest_signal(data, scores)
        _, explanation_nav, _, explanation_rebalances, explanation_trades = backtest(
            data,
            scores,
            start=EARLIEST_BACKTEST_START,
            end=explanation_day.date(),
            **explanation_settings,
        )
        explanation_state = explanation_nav.iloc[-1]
        explanation_current = [] if explanation_state["持仓"] == "现金" else str(explanation_state["持仓"]).split("|")
        explanation_snapshot = position_management_snapshot(
            data,
            explanation_day,
            explanation_selected,
            explanation_current,
            float(explanation_state["总仓位"]),
            **explanation_settings,
        ).iloc[0]
        st.markdown(
            f"""当前数据快照截至 **{explanation_day.date()}**。最新动量推荐为 **{explanation_snapshot['推荐持仓名称']}**，组合过去20个交易日估计年化波动率为 **{pct(explanation_snapshot['组合估计年化波动率'])}**；目标波动率为 **{pct(explanation_snapshot['目标波动率'])}**，所以目标总仓位为 **{pct(explanation_snapshot['目标总仓位'])}**，当前策略总仓位为 **{pct(explanation_snapshot['当前策略总仓位'])}**，判断为 **{explanation_snapshot['仓位判断']}**。"""
        )
        explanation_summary = build_rebalance_summary(explanation_rebalances, explanation_trades)
        signal_candidates = explanation_summary[
            (explanation_summary["调仓类型"] == "信号调仓")
            & (explanation_summary["当前持仓名称"] != explanation_summary["目标持仓名称"])
            & (explanation_summary["目标总仓位"] < 0.999999)
        ]
        if signal_candidates.empty:
            signal_candidates = explanation_summary[
                (explanation_summary["调仓类型"] == "信号调仓")
                & (explanation_summary["当前持仓名称"] != explanation_summary["目标持仓名称"])
            ]
        if not signal_candidates.empty:
            signal_example = signal_candidates.iloc[-1]
            signal_symbols = [] if signal_example["目标持仓"] == "现金" else str(signal_example["目标持仓"]).split("|")
            signal_volatility = portfolio_volatility(data, signal_symbols, pd.Timestamp(signal_example["执行日期"]))
            st.markdown(
                f"""**案例一：没有持有新目标标的，调仓时先确定新仓位**

在 **{signal_example['信号日期']}** 收盘后，策略从 **{signal_example['当前持仓名称']}** 切换至 **{signal_example['目标持仓名称']}**，并于 **{signal_example['执行日期']}** 开盘执行。目标标的过去20个交易日估计年化波动率约为 **{pct(signal_volatility)}**，因此在20%目标波动率下，目标总仓位为 **{pct(signal_example['目标总仓位'])}**。执行前总仓位为 **{pct(signal_example['调仓前总仓位'])}**，执行后为 **{pct(signal_example['调仓后总仓位'])}**。

本次调仓的卖出明细：{signal_example['卖出明细'] or '无'}；买入明细：{signal_example['买入明细'] or '无'}。剩余资金不再强行买入，而是作为现金保留。"""
            )
        risk_candidates = explanation_summary[explanation_summary["调仓类型"] == "风险仓位再平衡"]
        if not risk_candidates.empty:
            risk_example = risk_candidates.iloc[-1]
            risk_symbols = [] if risk_example["目标持仓"] == "现金" else str(risk_example["目标持仓"]).split("|")
            risk_volatility = portfolio_volatility(data, risk_symbols, pd.Timestamp(risk_example["执行日期"]))
            st.markdown(
                f"""**案例二：继续持有同一标的期间，单独调整仓位**

在 **{risk_example['执行日期']}**，当前持仓和目标持仓都为 **{risk_example['目标持仓名称']}**，没有换标的信号。但该标的过去20个交易日估计年化波动率约为 **{pct(risk_volatility)}**，对应目标总仓位 **{pct(risk_example['目标总仓位'])}**；当时调仓前总仓位为 **{pct(risk_example['调仓前总仓位'])}**，偏离约 **{abs(float(risk_example['调仓前总仓位']) - float(risk_example['目标总仓位'])):.2%}**，超过10个百分点调整带，因此单独触发风险仓位再平衡。

本次没有更换标的，只执行了仓位调整：{risk_example['卖出明细'] or risk_example['买入明细'] or '无成交明细'}。如果偏离不超过10个百分点，即使波动率变化，也会继续持有而不交易。"""
            )
        st.subheader("标的与起始时间")
        st.dataframe(coverage_table(data), hide_index=True, use_container_width=True)
        st.subheader("PandaData 密钥配置")
        st.code('PANDA_DATA_USERNAME = "你的账号"\nPANDA_DATA_PASSWORD = "你的密码"', language="toml")
        st.caption("本地可保存到项目的 .streamlit/secrets.toml；Streamlit Cloud 请在应用 Settings → Secrets 粘贴上述 TOML。不要将密钥提交到 GitHub。")
        st.warning("本应用使用调整后日线数据进行策略研究；数据更新后应重新核验日期连续性、价格有效性和复权跳变。结果不构成投资建议。")


if __name__ == "__main__":
    main()
