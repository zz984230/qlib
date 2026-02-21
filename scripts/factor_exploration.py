#!/usr/bin/env python
"""
因子探索策略优化脚本

使用 Qlib 表达式引擎进行因子探索，在控制回撤(目标 <= 3%)的情况下，
找到能跑赢"买入并持有"的策略。

特性:
- 定义15个核心因子池（动量、均线偏离、成交量、波动率、技术指标）
- 因子 IC 计算和筛选
- 因子组合搜索（2-4个因子组合）
- 适度放宽的风控参数（仓位30%、回撤3%、冷却1天、止损2%）
- 与买入持有对比
- Top 5 策略排名
- PDF 报告生成（3个时间周期）
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.akshare_loader import AkshareLoader
from src.strategy.base import BaseStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# PDF 生成依赖
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
        HRFlowable,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    logger.warning(f"PDF generation dependencies not available: {e}")


# =============================================================================
# 因子池定义 (Qlib 表达式格式)
# =============================================================================

FACTOR_POOL = {
    # 价格动量
    "price_momentum_5": "($close - Ref($close, 5)) / Ref($close, 5)",
    "price_momentum_10": "($close - Ref($close, 10)) / Ref($close, 10)",
    "price_momentum_20": "($close - Ref($close, 20)) / Ref($close, 20)",

    # 均线偏离
    "ma5_deviation": "($close - Mean($close, 5)) / Mean($close, 5)",
    "ma10_deviation": "($close - Mean($close, 10)) / Mean($close, 10)",
    "ma20_deviation": "($close - Mean($close, 20)) / Mean($close, 20)",

    # 成交量
    "volume_ratio_5_20": "Mean($volume, 5) / Mean($volume, 20)",
    "volume_change": "($volume - Ref($volume, 1)) / Ref($volume, 1)",

    # 波动率
    "volatility_10": "Std($close, 10) / Mean($close, 10)",
    "volatility_20": "Std($close, 20) / Mean($close, 20)",

    # 技术指标
    "rsi_14": "RSI($close, 14) / 100",
    "bb_position": "($close - Mean($close, 20)) / (2 * Std($close, 20))",

    # K线特征
    "k_mid": "($close - $low) / ($high - $low)",  # K线位置
    "k_range": "($high - $low) / $close",  # 振幅

    # 均线趋势
    "ma_trend": "Mean($close, 5) / Mean($close, 20)",
}


class FactorExplorerStrategy(BaseStrategy):
    """因子探索策略 - 基于因子组合生成信号"""

    def __init__(
        self,
        factor_names: list[str],
        factor_weights: list[float] | None = None,
        signal_threshold_buy: float = 0.25,
        signal_threshold_sell: float = -0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factor_names = factor_names
        self.factor_weights = factor_weights or [1.0 / len(factor_names)] * len(factor_names)
        self.signal_threshold_buy = signal_threshold_buy
        self.signal_threshold_sell = signal_threshold_sell

        # 获取因子表达式
        self.factor_expressions = [FACTOR_POOL.get(name, "") for name in factor_names]

    def get_factors(self) -> list[str]:
        return self.factor_expressions

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        signals = np.zeros(len(close))

        if len(close) < 20:
            return signals

        # 计算所有因子值
        factor_values = self._calculate_factors(data)

        # 标准化每个因子
        normalized_factors = []
        for fv in factor_values:
            nf = self._normalize_factor(fv)
            normalized_factors.append(nf)

        # 加权组合
        combined = np.zeros(len(close))
        for nf, w in zip(normalized_factors, self.factor_weights):
            combined += w * np.nan_to_num(nf, nan=0)

        # 生成信号
        for i in range(len(combined)):
            if combined[i] >= self.signal_threshold_buy:
                signals[i] = min(combined[i] * 2, 1.0)
            elif combined[i] <= self.signal_threshold_sell:
                signals[i] = max(combined[i] * 2, -1.0)

        return signals

    def _calculate_factors(self, data: pd.DataFrame) -> list[np.ndarray]:
        factor_values = []
        for name in self.factor_names:
            fv = self._calculate_single_factor(name, data)
            factor_values.append(fv)
        return factor_values

    def _calculate_single_factor(self, name: str, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        high = data["high"].values if "high" in data.columns else close
        low = data["low"].values if "low" in data.columns else close
        volume = data["volume"].values if "volume" in data.columns else np.ones(len(close))

        n = len(close)
        result = np.full(n, np.nan)

        try:
            if name == "price_momentum_5":
                result[5:] = (close[5:] - close[:-5]) / close[:-5]
            elif name == "price_momentum_10":
                result[10:] = (close[10:] - close[:-10]) / close[:-10]
            elif name == "price_momentum_20":
                result[20:] = (close[20:] - close[:-20]) / close[:-20]
            elif name == "ma5_deviation":
                ma5 = self._sma(close, 5)
                result = (close - ma5) / ma5
            elif name == "ma10_deviation":
                ma10 = self._sma(close, 10)
                result = (close - ma10) / ma10
            elif name == "ma20_deviation":
                ma20 = self._sma(close, 20)
                result = (close - ma20) / ma20
            elif name == "volume_ratio_5_20":
                vol5 = self._sma(volume, 5)
                vol20 = self._sma(volume, 20)
                result = vol5 / vol20
            elif name == "volume_change":
                result[1:] = (volume[1:] - volume[:-1]) / volume[:-1]
            elif name == "volatility_10":
                std10 = self._rolling_std(close, 10)
                ma10 = self._sma(close, 10)
                result = std10 / ma10
            elif name == "volatility_20":
                std20 = self._rolling_std(close, 20)
                ma20 = self._sma(close, 20)
                result = std20 / ma20
            elif name == "rsi_14":
                rsi = self._calculate_rsi(close, 14)
                result = rsi / 100
            elif name == "bb_position":
                ma20 = self._sma(close, 20)
                std20 = self._rolling_std(close, 20)
                result = (close - ma20) / (2 * std20)
            elif name == "k_mid":
                range_hl = high - low
                range_hl[range_hl == 0] = np.nan
                result = (close - low) / range_hl
            elif name == "k_range":
                result = (high - low) / close
            elif name == "ma_trend":
                ma5 = self._sma(close, 5)
                ma20 = self._sma(close, 20)
                result = ma5 / ma20
            else:
                # 默认动量
                result[5:] = (close[5:] - close[:-5]) / close[:-5]
        except Exception:
            result = np.full(n, np.nan)

        return result

    def _normalize_factor(self, factor: np.ndarray) -> np.ndarray:
        result = np.full(len(factor), np.nan)
        valid_mask = ~np.isnan(factor)
        if not np.any(valid_mask):
            return result

        valid = factor[valid_mask]
        mean, std = np.mean(valid), np.std(valid)
        if std > 0:
            z = (factor - mean) / std
            result = np.tanh(z)
        return result

    def _sma(self, data: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        return result

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])
        return result

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        rsi = np.full(len(close), 50.0)
        if len(close) < period + 1:
            return rsi

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(close) - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))
        return rsi

    def __str__(self) -> str:
        return f"FactorExplorer({self.factor_names})"


class FactorExplorationBacktestRunner:
    """因子探索回测执行器 - 放宽风控参数"""

    def __init__(
        self,
        stop_loss_pct: float = 0.02,       # 止损 2%
        trailing_stop_pct: float = 0.015,   # 移动止损 1.5%
        daily_loss_limit: float = 0.015,    # 日亏损限制 1.5%
        max_drawdown_limit: float = 0.03,   # 回撤限制 3%
        position_size_pct: float = 0.30,    # 仓位比例 30%
        cooldown_days: int = 1,             # 冷却期 1天
    ):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.position_size_pct = position_size_pct
        self.cooldown_days = cooldown_days

    def run(self, data: pd.DataFrame, strategy: BaseStrategy, initial_cash: float = 100000) -> dict:
        close = data["close"].values
        dates = data["date"].values if "date" in data.columns else data.index

        cash = initial_cash
        position = 0
        entry_price = 0.0
        highest_price = 0.0
        portfolio_values = []
        trades = []

        trading_suspended = False
        suspend_days = 0
        portfolio_high = initial_cash
        cooldown = 0

        for i in range(len(close)):
            current_price = close[i]

            if position > 0:
                current_value = cash + position * current_price
            else:
                current_value = cash

            if current_value > portfolio_high:
                portfolio_high = current_value

            current_drawdown = (portfolio_high - current_value) / portfolio_high

            # 最大回撤限制
            if current_drawdown >= self.max_drawdown_limit:
                if position > 0:
                    cash += position * current_price
                    trades.append({
                        "date": str(dates[i])[:10],
                        "action": "force_sell",
                        "price": current_price,
                        "shares": position,
                        "reason": f"max_dd_{current_drawdown:.1%}",
                        "type": "sell",
                    })
                    position = 0
                trading_suspended = True
                suspend_days = 3

            if trading_suspended:
                suspend_days -= 1
                if suspend_days <= 0:
                    trading_suspended = False
                    portfolio_high = current_value

            if cooldown > 0:
                cooldown -= 1

            if position > 0:
                if current_price > highest_price:
                    highest_price = current_price

                pnl_ratio = (current_price - entry_price) / entry_price
                drawdown_from_high = (highest_price - current_price) / highest_price

                # 移动止损
                if pnl_ratio > 0.015 and drawdown_from_high >= self.trailing_stop_pct:
                    profit = (current_price - entry_price) * position
                    cash += position * current_price
                    trades.append({
                        "date": str(dates[i])[:10],
                        "action": "trailing_stop",
                        "price": current_price,
                        "shares": position,
                        "reason": f"trail_{drawdown_from_high:.1%}",
                        "type": "sell",
                        "profit": profit,
                    })
                    position = 0

                # 止损
                elif pnl_ratio <= -self.stop_loss_pct:
                    profit = (current_price - entry_price) * position
                    cash += position * current_price
                    trades.append({
                        "date": str(dates[i])[:10],
                        "action": "stop_loss",
                        "price": current_price,
                        "shares": position,
                        "reason": f"sl_{pnl_ratio:.1%}",
                        "type": "sell",
                        "profit": profit,
                    })
                    position = 0
                    cooldown = self.cooldown_days

            # 信号交易
            if not trading_suspended and cooldown == 0 and i >= 20:
                hist_data = data.iloc[:i+1]
                signals = strategy.generate_signals(hist_data)
                current_signal = signals[-1] if len(signals) > 0 else 0

                if current_signal > 0 and position == 0:
                    position_value = cash * self.position_size_pct
                    shares = int(position_value / current_price / 100) * 100
                    if shares > 0:
                        cost = shares * current_price
                        cash -= cost
                        position = shares
                        entry_price = current_price
                        highest_price = current_price
                        trades.append({
                            "date": str(dates[i])[:10],
                            "action": "buy",
                            "price": current_price,
                            "shares": shares,
                            "reason": f"signal_{current_signal:.2f}",
                            "type": "buy",
                        })

                elif current_signal < 0 and position > 0:
                    profit = (current_price - entry_price) * position
                    cash += position * current_price
                    trades.append({
                        "date": str(dates[i])[:10],
                        "action": "sell",
                        "price": current_price,
                        "shares": position,
                        "reason": f"signal_{current_signal:.2f}",
                        "type": "sell",
                        "profit": profit,
                    })
                    position = 0

            total_value = cash + position * current_price
            portfolio_values.append({
                "date": dates[i],
                "value": total_value,
                "cash": cash,
                "position": position,
                "price": current_price,
            })

        if position > 0:
            cash += position * close[-1]

        pv_df = pd.DataFrame(portfolio_values)
        pv_df["date"] = pd.to_datetime(pv_df["date"])
        pv_df = pv_df.set_index("date")

        returns = pv_df["value"].pct_change().dropna()

        total_return = (cash - initial_cash) / initial_cash
        max_drawdown = self._calculate_max_drawdown(pv_df["value"])
        sharpe_ratio = self._calculate_sharpe(returns)
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # 买入持有基准
        first_price = close[0]
        last_price = close[-1]
        buy_hold_shares = int(initial_cash / first_price / 100) * 100
        buy_hold_final = buy_hold_shares * last_price + (initial_cash - buy_hold_shares * first_price)
        buy_hold_return = (buy_hold_final - initial_cash) / initial_cash

        buy_hold_values = []
        for i, price in enumerate(close):
            buy_hold_values.append({
                "date": dates[i],
                "value": buy_hold_shares * price + (initial_cash - buy_hold_shares * first_price),
            })
        buy_hold_df = pd.DataFrame(buy_hold_values)
        buy_hold_df["date"] = pd.to_datetime(buy_hold_df["date"])
        buy_hold_df = buy_hold_df.set_index("date")

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "final_value": cash,
            "trades": trades,
            "portfolio_values": pv_df,
            "buy_hold_return": buy_hold_return,
            "buy_hold_values": buy_hold_df,
        }

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        cummax = values.cummax()
        drawdown = (cummax - values) / cummax
        return drawdown.max()

    def _calculate_sharpe(self, returns: pd.Series, rf: float = 0.03) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_return = returns.mean() * 252 - rf
        volatility = returns.std() * np.sqrt(252)
        return excess_return / volatility if volatility > 0 else 0.0


def calculate_factor_ic(factor_name: str, data: pd.DataFrame, forward_days: int = 5) -> float:
    """
    计算因子 IC (Information Coefficient)

    IC = corr(factor_value, forward_return)
    """
    close = data["close"].values
    n = len(close)

    if n < forward_days + 20:
        return 0.0

    # 计算因子值
    strategy = FactorExplorerStrategy([factor_name])
    factor_values = strategy._calculate_single_factor(factor_name, data)

    # 计算未来收益率
    forward_returns = np.full(n, np.nan)
    forward_returns[:-forward_days] = (close[forward_days:] - close[:-forward_days]) / close[:-forward_days]

    # 计算 IC
    valid_mask = ~np.isnan(factor_values) & ~np.isnan(forward_returns)
    if np.sum(valid_mask) < 10:
        return 0.0

    fv = factor_values[valid_mask]
    fr = forward_returns[valid_mask]

    if np.std(fv) == 0 or np.std(fr) == 0:
        return 0.0

    ic = np.corrcoef(fv, fr)[0, 1]
    return ic if not np.isnan(ic) else 0.0


def search_factor_combinations(
    data: pd.DataFrame,
    top_factors: list[str],
    max_factors_per_combo: int = 4,
    initial_cash: float = 50000,
) -> list[dict]:
    """
    搜索因子组合，找到最优策略

    Returns:
        策略结果列表，按综合评分排序
    """
    runner = FactorExplorationBacktestRunner()
    results = []

    # 遍历 2-4 个因子的组合
    for num_factors in range(2, min(max_factors_per_combo + 1, len(top_factors) + 1)):
        for combo in combinations(top_factors[:10], num_factors):  # 限制 Top 10 因子
            factor_names = list(combo)

            try:
                strategy = FactorExplorerStrategy(factor_names)
                result = runner.run(data, strategy, initial_cash)

                # 计算综合评分
                excess = result["total_return"] - result["buy_hold_return"]
                drawdown_penalty = max(0, result["max_drawdown"] - 0.03) * 10
                score = (
                    result["total_return"] * 100 +
                    excess * 50 +
                    result["sharpe_ratio"] * 5 -
                    drawdown_penalty
                )

                results.append({
                    "factor_names": factor_names,
                    "total_return": result["total_return"],
                    "buy_hold_return": result["buy_hold_return"],
                    "excess_return": excess,
                    "max_drawdown": result["max_drawdown"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "win_rate": result["win_rate"],
                    "trades": result["trades"],
                    "portfolio_values": result["portfolio_values"],
                    "buy_hold_values": result["buy_hold_values"],
                    "score": score,
                })
            except Exception as e:
                logger.debug(f"Combo {factor_names} failed: {e}")
                continue

    # 按评分排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_chinese_font():
    """获取中文字体"""
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                return 'ChineseFont'
            except Exception:
                continue
    return 'Helvetica'


def generate_chart(pv_df: pd.DataFrame, output_path: Path, buy_hold_df: pd.DataFrame = None, title: str = ""):
    """生成净值曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 净值曲线
    ax1 = axes[0, 0]
    ax1.plot(pv_df.index, pv_df['value'] / pv_df['value'].iloc[0], 'b-', linewidth=1.5, label='策略净值')

    if buy_hold_df is not None and len(buy_hold_df) > 0:
        buy_hold_nav = buy_hold_df['value'] / buy_hold_df['value'].iloc[0]
        ax1.plot(buy_hold_df.index, buy_hold_nav, 'g--', linewidth=1.5, label='买入持有', alpha=0.8)

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'净值曲线对比 - {title}', fontsize=12)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('净值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 回撤曲线
    ax2 = axes[0, 1]
    cummax = pv_df['value'].cummax()
    drawdown = (cummax - pv_df['value']) / cummax * 100
    ax2.fill_between(pv_df.index, 0, -drawdown, color='red', alpha=0.3)
    ax2.plot(pv_df.index, -drawdown, 'r-', linewidth=1)
    ax2.axhline(y=-3, color='orange', linestyle='--', label='目标回撤 -3%')
    ax2.set_title('回撤曲线', fontsize=12)
    ax2.set_xlabel('日期')
    ax2.set_ylabel('回撤 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 日收益率分布
    ax3 = axes[1, 0]
    returns = pv_df['value'].pct_change().dropna() * 100
    ax3.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax3.axvline(x=returns.mean(), color='red', linestyle='--', label=f'均值: {returns.mean():.2f}%')
    ax3.set_title('日收益率分布', fontsize=12)
    ax3.set_xlabel('日收益率 (%)')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 现金 vs 持仓
    ax4 = axes[1, 1]
    ax4.stackplot(pv_df.index, pv_df['cash'], pv_df['position'] * pv_df['price'],
                  labels=['现金', '持仓'], colors=['#2ecc71', '#3498db'], alpha=0.8)
    ax4.set_title('资金分配', fontsize=12)
    ax4.set_xlabel('日期')
    ax4.set_ylabel('金额')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_factor_exploration_pdf(
    stock_code: str,
    stock_name: str,
    top_strategies: list[dict],
    factor_ic_results: list[tuple],
    results_by_period: dict,
    data_info: dict,
    chart_paths: dict,
    output_path: Path,
):
    """创建因子探索 PDF 报告"""
    if not REPORTLAB_AVAILABLE:
        logger.error("Please install reportlab: uv pip install reportlab matplotlib")
        return None

    chinese_font = get_chinese_font()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ChineseTitle',
        parent=styles['Title'],
        fontName=chinese_font,
        fontSize=20,
        leading=26,
        alignment=1,
        spaceAfter=15,
    )

    heading_style = ParagraphStyle(
        'ChineseHeading',
        parent=styles['Heading1'],
        fontName=chinese_font,
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
        textColor=colors.HexColor('#1a5276'),
    )

    subheading_style = ParagraphStyle(
        'ChineseSubHeading',
        parent=styles['Heading2'],
        fontName=chinese_font,
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=6,
        textColor=colors.HexColor('#2874a6'),
    )

    normal_style = ParagraphStyle(
        'ChineseNormal',
        parent=styles['Normal'],
        fontName=chinese_font,
        fontSize=10,
        leading=14,
        spaceBefore=3,
        spaceAfter=3,
    )

    story = []

    # 标题页
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"{stock_name} ({stock_code})", title_style))
    story.append(Paragraph("因子探索策略优化报告", title_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.5*cm))

    # 基本信息
    info_data = [
        ["报告时间", data_info.get('generate_time', '-')],
        ["初始资金", f"{data_info.get('initial_cash', 0):,.0f} 元"],
        ["风控参数", "仓位30%, 回撤3%, 冷却1天, 止损2%"],
    ]
    info_table = Table(info_data, colWidths=[4*cm, 10*cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)

    # 因子 IC 分析
    story.append(PageBreak())
    story.append(Paragraph("一、因子 IC 分析", heading_style))

    ic_header = ["因子名称", "IC值", "IC评级"]
    ic_rows = [ic_header]
    for factor_name, ic in factor_ic_results[:15]:
        if abs(ic) >= 0.05:
            rating = "强"
            color = colors.HexColor('#27ae60')
        elif abs(ic) >= 0.03:
            rating = "中"
            color = colors.HexColor('#f39c12')
        else:
            rating = "弱"
            color = colors.HexColor('#95a5a6')
        ic_rows.append([factor_name, f"{ic:.4f}", rating])

    ic_table = Table(ic_rows, colWidths=[6*cm, 4*cm, 4*cm])
    ic_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    story.append(ic_table)

    # Top 5 策略
    story.append(PageBreak())
    story.append(Paragraph("二、Top 5 策略排名", heading_style))

    for rank, strategy in enumerate(top_strategies[:5], 1):
        story.append(Paragraph(f"策略 #{rank}: {', '.join(strategy['factor_names'][:3])}{'...' if len(strategy['factor_names']) > 3 else ''}", subheading_style))

        perf_data = [
            ["指标", "数值"],
            ["策略收益率", f"{strategy['total_return']:.2%}"],
            ["买入持有收益率", f"{strategy['buy_hold_return']:.2%}"],
            ["超额收益", f"{strategy['excess_return']:+.2%}"],
            ["最大回撤", f"{strategy['max_drawdown']:.2%}"],
            ["夏普比率", f"{strategy['sharpe_ratio']:.2f}"],
            ["胜率", f"{strategy['win_rate']:.2%}"],
        ]

        perf_table = Table(perf_data, colWidths=[5*cm, 5*cm])
        perf_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874a6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(perf_table)
        story.append(Spacer(1, 0.5*cm))

    # 多周期分析
    story.append(PageBreak())
    story.append(Paragraph("三、多周期绩效对比 (最佳策略)", heading_style))

    period_names = {"1y": "近1年", "3m": "近3月", "1m": "近1月"}

    comparison_header = ["指标", "近1年", "近3月", "近1月"]
    comparison_rows = [comparison_header]

    metrics = [
        ("策略收益率", "total_return", "{:.2%}"),
        ("买入持有收益率", "buy_hold_return", "{:.2%}"),
        ("超额收益", "excess_return", "{:.2%}"),
        ("最大回撤", "max_drawdown", "{:.2%}"),
        ("夏普比率", "sharpe_ratio", "{:.2f}"),
    ]

    for metric_name, metric_key, fmt in metrics:
        row = [metric_name]
        for period in ["1y", "3m", "1m"]:
            if period in results_by_period and len(results_by_period[period]) > 0:
                best = results_by_period[period][0]
                row.append(fmt.format(best.get(metric_key, 0)))
            else:
                row.append("-")
        comparison_rows.append(row)

    comparison_table = Table(comparison_rows, colWidths=[4*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    comparison_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    story.append(comparison_table)

    # 图表
    for period in ["1y", "3m", "1m"]:
        if period in chart_paths and chart_paths[period].exists():
            story.append(PageBreak())
            story.append(Paragraph(f"四、{period_names[period]}可视化分析", heading_style))
            img = Image(str(chart_paths[period]), width=16*cm, height=11*cm)
            story.append(img)

    # 结论
    story.append(PageBreak())
    story.append(Paragraph("五、结论与建议", heading_style))

    conclusions = ["<b>因子探索结果:</b>"]

    if top_strategies:
        best = top_strategies[0]
        conclusions.append(f"- 最佳策略因子组合: {', '.join(best['factor_names'])}")
        conclusions.append(f"- 最佳策略收益率: {best['total_return']:.2%}")
        conclusions.append(f"- 买入持有收益率: {best['buy_hold_return']:.2%}")
        conclusions.append(f"- 超额收益: {best['excess_return']:+.2%}")
        conclusions.append(f"- 最大回撤: {best['max_drawdown']:.2%}")

        if best['max_drawdown'] <= 0.03:
            conclusions.append("- <b>风险控制: 达标</b> (回撤 <= 3%)")
        else:
            conclusions.append("- <b>风险控制: 需优化</b> (回撤 > 3%)")

        if best['excess_return'] > 0:
            conclusions.append("- <b>收益表现: 跑赢基准</b>")
        else:
            conclusions.append("- <b>收益表现: 跑输基准</b>")

    for conclusion in conclusions:
        story.append(Paragraph(conclusion, normal_style))
        story.append(Spacer(1, 0.2*cm))

    # 免责声明
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=9,
        textColor=colors.grey,
        alignment=1,
    )
    story.append(Paragraph("<b>免责声明</b>", disclaimer_style))
    story.append(Paragraph("本报告仅供参考，不构成投资建议。过往业绩不代表未来表现。", disclaimer_style))
    story.append(Paragraph(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", disclaimer_style))

    doc.build(story)
    logger.info(f"PDF report generated: {output_path}")
    return output_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="因子探索策略优化")
    parser.add_argument("--symbol", type=str, default="601138", help="股票代码")
    parser.add_argument("--name", type=str, default="工业富联", help="股票名称")
    parser.add_argument("--cash", type=float, default=50000, help="初始资金")
    parser.add_argument("--max-factor-combo", type=int, default=3, help="最大因子组合数")
    args = parser.parse_args()

    # 时间周期定义
    today = datetime.now()
    end_date = today.strftime("%Y-%m-%d")

    periods = {
        "1y": {
            "name": "近1年",
            "start": (today - timedelta(days=365)).strftime("%Y-%m-%d"),
        },
        "3m": {
            "name": "近3月",
            "start": (today - timedelta(days=90)).strftime("%Y-%m-%d"),
        },
        "1m": {
            "name": "近1月",
            "start": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
        },
    }

    logger.info(f"[INFO] 开始因子探索: {args.symbol} ({args.name})")

    # 获取数据
    loader = AkshareLoader()
    full_data = loader.get_stock_data(
        symbol=args.symbol,
        start_date=periods["1y"]["start"],
        end_date=end_date,
        adjust="qfq",
    )

    if full_data is None or len(full_data) == 0:
        logger.error(f"[ERROR] 无法获取 {args.symbol} 数据")
        return 1

    logger.info(f"[OK] 获取到 {len(full_data)} 条数据")

    # Step 1: 计算所有因子的 IC
    print("\n" + "=" * 70)
    print("Step 1: 因子 IC 分析")
    print("=" * 70)

    factor_ic_results = []
    for factor_name in FACTOR_POOL.keys():
        ic = calculate_factor_ic(factor_name, full_data)
        factor_ic_results.append((factor_name, ic))

    # 按 IC 绝对值排序
    factor_ic_results.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n因子 IC 排名 (Top 10):")
    print("-" * 50)
    for i, (name, ic) in enumerate(factor_ic_results[:10], 1):
        status = "[强]" if abs(ic) >= 0.05 else "[中]" if abs(ic) >= 0.03 else "[弱]"
        print(f"{i:2}. {name:25} IC={ic:+.4f} {status}")

    # Step 2: 获取高 IC 因子
    top_factors = [name for name, ic in factor_ic_results if abs(ic) >= 0.01][:10]

    print(f"\n[OK] 选取 Top {len(top_factors)} 因子用于组合搜索")

    # Step 3: 多周期回测
    print("\n" + "=" * 70)
    print("Step 2: 多周期因子组合搜索")
    print("=" * 70)

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_period = {}
    chart_paths = {}

    for period_key, period_info in periods.items():
        start_date = period_info["start"]
        period_name = period_info["name"]

        # 过滤数据
        full_data["date"] = pd.to_datetime(full_data["date"])
        period_data = full_data[full_data["date"] >= start_date].copy()

        if len(period_data) < 30:
            logger.warning(f"[WARN] {period_name} 数据不足，跳过")
            continue

        logger.info(f"[INFO] 搜索 {period_name} 最优因子组合...")

        # 搜索因子组合
        strategy_results = search_factor_combinations(
            period_data,
            top_factors,
            max_factors_per_combo=args.max_factor_combo,
            initial_cash=args.cash,
        )

        if strategy_results:
            results_by_period[period_key] = strategy_results

            best = strategy_results[0]
            status = "[OK]" if best["max_drawdown"] <= 0.03 else "[WARN]"
            print(f"\n{period_name} 最佳策略:")
            print(f"  因子: {best['factor_names']}")
            print(f"  策略收益: {best['total_return']:.2%} vs 买入持有: {best['buy_hold_return']:.2%}")
            print(f"  超额收益: {best['excess_return']:+.2%}, 最大回撤: {best['max_drawdown']:.2%} {status}")

            # 生成图表
            if REPORTLAB_AVAILABLE:
                chart_path = output_dir / f"{args.symbol}_{period_key}_factor_chart.png"
                generate_chart(
                    best["portfolio_values"],
                    chart_path,
                    best["buy_hold_values"],
                    title=f"{period_name} - {', '.join(best['factor_names'][:2])}"
                )
                chart_paths[period_key] = chart_path
        else:
            print(f"\n{period_name}: 未找到有效策略")

    # Step 4: 汇总结果
    print("\n" + "=" * 70)
    print("Step 3: Top 5 策略排名 (近1年)")
    print("=" * 70)

    if "1y" in results_by_period:
        for i, strategy in enumerate(results_by_period["1y"][:5], 1):
            factors_str = ", ".join(strategy['factor_names'][:3])
            if len(strategy['factor_names']) > 3:
                factors_str += f"... ({len(strategy['factor_names'])}个)"
            excess_sign = "+" if strategy['excess_return'] >= 0 else ""
            print(f"{i}. [{factors_str}]")
            print(f"   收益: {strategy['total_return']:.2%} vs 持有: {strategy['buy_hold_return']:.2%} "
                  f"(超额{excess_sign}{strategy['excess_return']:.2%})")
            print(f"   回撤: {strategy['max_drawdown']:.2%}, 夏普: {strategy['sharpe_ratio']:.2f}")

    # Step 5: 生成 PDF 报告
    if REPORTLAB_AVAILABLE and results_by_period:
        pdf_path = output_dir / f"{args.symbol}_factor_exploration_report.pdf"

        data_info = {
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'initial_cash': args.cash,
        }

        # 获取 Top 5 策略（优先用1年期数据）
        top_strategies = results_by_period.get("1y", [])
        if not top_strategies:
            for period in ["3m", "1m"]:
                if period in results_by_period:
                    top_strategies = results_by_period[period]
                    break

        logger.info("[INFO] 生成 PDF 报告...")
        create_factor_exploration_pdf(
            stock_code=args.symbol,
            stock_name=args.name,
            top_strategies=top_strategies[:5],
            factor_ic_results=factor_ic_results,
            results_by_period=results_by_period,
            data_info=data_info,
            chart_paths=chart_paths,
            output_path=pdf_path,
        )

        print(f"\n[OK] PDF 报告已保存: {pdf_path}")
    else:
        print("\n[WARN] PDF 生成跳过 (reportlab 未安装)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
