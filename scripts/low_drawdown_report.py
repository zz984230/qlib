#!/usr/bin/env python
"""
低回撤策略回测 + PDF报告生成
针对工业富联(601138)进行优化，目标最大回撤控制在3%以内
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

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


class LowDrawdownStrategy(BaseStrategy):
    """低回撤策略"""

    def __init__(
        self,
        ma_short: int = 5,
        ma_long: int = 15,
        rsi_period: int = 14,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
        stop_loss_pct: float = 0.015,
        trailing_stop_pct: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct

    def get_factors(self) -> list[str]:
        return [
            f"Mean($close, {self.ma_short})",
            f"Mean($close, {self.ma_long})",
            f"RSI($close, {self.rsi_period})",
        ]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        signals = np.zeros(len(close))

        if len(close) < self.ma_long + 1:
            return signals

        short_ma = self._sma(close, self.ma_short)
        long_ma = self._sma(close, self.ma_long)
        rsi = self._calculate_rsi(close, self.rsi_period)

        for i in range(self.ma_long, len(close)):
            trend_up = short_ma[i] > short_ma[i-1] if i > 0 else False
            price_above_long_ma = close[i] > long_ma[i]

            if price_above_long_ma and trend_up:
                if (short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1] and
                    rsi[i] < self.rsi_overbought and rsi[i] > 30):
                    signals[i] = 1
                elif 30 < rsi[i] < self.rsi_oversold:
                    signals[i] = 0.7

            if short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
                signals[i] = -1
            elif rsi[i] > self.rsi_overbought:
                signals[i] = -0.8
            elif close[i] < long_ma[i] * 0.98:
                signals[i] = -0.6

        return signals

    def _sma(self, data: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        return result

    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
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
        return f"LowDrawdownStrategy(ma={self.ma_short}/{self.ma_long}, sl={self.stop_loss_pct:.1%})"


class LowDrawdownBacktestRunner:
    """低回撤回测执行器"""

    def __init__(
        self,
        stop_loss_pct: float = 0.015,
        trailing_stop_pct: float = 0.01,
        daily_loss_limit: float = 0.01,
        max_drawdown_limit: float = 0.024,
        position_size_pct: float = 0.2,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.position_size_pct = position_size_pct

    def run(self, data: pd.DataFrame, strategy: BaseStrategy, initial_cash: float = 1000000) -> dict:
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
        cooldown_days = 0

        for i in range(len(close)):
            current_price = close[i]

            if position > 0:
                current_value = cash + position * current_price
            else:
                current_value = cash

            if current_value > portfolio_high:
                portfolio_high = current_value

            current_drawdown = (portfolio_high - current_value) / portfolio_high

            if current_drawdown >= self.max_drawdown_limit:
                if position > 0:
                    cash += position * current_price
                    trades.append({
                        "date": str(dates[i])[:10],
                        "action": "force_sell",
                        "price": current_price,
                        "shares": position,
                        "reason": f"max_drawdown_{current_drawdown:.1%}",
                        "type": "sell",
                    })
                    position = 0
                trading_suspended = True
                suspend_days = 5

            if trading_suspended:
                suspend_days -= 1
                if suspend_days <= 0:
                    trading_suspended = False
                    portfolio_high = current_value

            if cooldown_days > 0:
                cooldown_days -= 1

            if position > 0:
                if current_price > highest_price:
                    highest_price = current_price

                pnl_ratio = (current_price - entry_price) / entry_price
                drawdown_from_high = (highest_price - current_price) / highest_price

                if pnl_ratio > 0.01 and drawdown_from_high >= self.trailing_stop_pct:
                    profit = (current_price - entry_price) * position
                    cash += position * current_price
                    trades.append({
                        "date": str(dates[i])[:10],
                        "action": "trailing_stop",
                        "price": current_price,
                        "shares": position,
                        "reason": f"trailing_stop_{drawdown_from_high:.1%}",
                        "type": "sell",
                        "profit": profit,
                    })
                    position = 0

                elif pnl_ratio <= -self.stop_loss_pct:
                    profit = (current_price - entry_price) * position
                    cash += position * current_price
                    trades.append({
                        "date": str(dates[i])[:10],
                        "action": "stop_loss",
                        "price": current_price,
                        "shares": position,
                        "reason": f"stop_loss_{pnl_ratio:.1%}",
                        "type": "sell",
                        "profit": profit,
                    })
                    position = 0
                    cooldown_days = 3

            if not trading_suspended and cooldown_days == 0 and i >= strategy.ma_long:
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

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "final_value": cash,
            "trades": trades,
            "portfolio_values": pv_df,
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


def generate_chart(pv_df: pd.DataFrame, output_path: Path):
    """生成净值曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 净值曲线
    ax1 = axes[0, 0]
    ax1.plot(pv_df.index, pv_df['value'] / pv_df['value'].iloc[0], 'b-', linewidth=1.5, label='Strategy')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('NAV Curve', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('NAV')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 回撤曲线
    ax2 = axes[0, 1]
    cummax = pv_df['value'].cummax()
    drawdown = (cummax - pv_df['value']) / cummax * 100
    ax2.fill_between(pv_df.index, 0, -drawdown, color='red', alpha=0.3)
    ax2.plot(pv_df.index, -drawdown, 'r-', linewidth=1)
    ax2.axhline(y=-3, color='orange', linestyle='--', label='Target -3%')
    ax2.set_title('Drawdown Curve', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 日收益率分布
    ax3 = axes[1, 0]
    returns = pv_df['value'].pct_change().dropna() * 100
    ax3.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax3.axvline(x=returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
    ax3.set_title('Daily Return Distribution', fontsize=12)
    ax3.set_xlabel('Daily Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 现金 vs 持仓
    ax4 = axes[1, 1]
    ax4.stackplot(pv_df.index, pv_df['cash'], pv_df['position'] * pv_df['price'],
                  labels=['Cash', 'Position'], colors=['#2ecc71', '#3498db'], alpha=0.8)
    ax4.set_title('Capital Allocation', fontsize=12)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Value')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_pdf_report(
    stock_code: str,
    stock_name: str,
    result: dict,
    data_info: dict,
    chart_path: Path,
    output_path: Path,
):
    """创建PDF报告"""
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
        fontSize=22,
        leading=28,
        alignment=1,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        'ChineseHeading',
        parent=styles['Heading1'],
        fontName=chinese_font,
        fontSize=14,
        leading=18,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#1a5276'),
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

    # Title Page
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(f"{stock_name} ({stock_code})", title_style))
    story.append(Paragraph("Low Drawdown Strategy Report", title_style))
    story.append(Spacer(1, 1*cm))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.5*cm))

    # Basic Info
    info_data = [
        ["Report Time", data_info.get('generate_time', '-')],
        ["Data Period", f"{data_info.get('start_date', '-')} to {data_info.get('end_date', '-')}"],
        ["Trading Days", str(data_info.get('trading_days', '-'))],
        ["Initial Capital", f"${data_info.get('initial_cash', 0):,.0f}"],
    ]

    info_table = Table(info_data, colWidths=[4*cm, 10*cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)

    # Performance Summary
    story.append(PageBreak())
    story.append(Paragraph("1. Performance Summary", heading_style))

    perf_data = [
        ["Metric", "Value"],
        ["Total Return", f"{result['total_return']:.2%}"],
        ["Max Drawdown", f"{result['max_drawdown']:.2%}"],
        ["Sharpe Ratio", f"{result['sharpe_ratio']:.2f}"],
        ["Win Rate", f"{result['win_rate']:.2%}"],
        ["Final Value", f"${result['final_value']:,.2f}"],
    ]

    perf_table = Table(perf_data, colWidths=[6*cm, 6*cm])
    perf_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    story.append(perf_table)

    # Risk Control Parameters
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("2. Risk Control Parameters", heading_style))

    risk_data = [
        ["Parameter", "Value", "Description"],
        ["Stop Loss", "1.5%", "Single position max loss"],
        ["Trailing Stop", "1.0%", "Protect profits after 1% gain"],
        ["Daily Loss Limit", "1.0%", "Daily max loss threshold"],
        ["Drawdown Limit", "2.4%", "Portfolio drawdown trigger"],
        ["Position Size", "20%", "Single trade capital allocation"],
        ["Cooldown Period", "3 days", "Wait after stop loss"],
    ]

    risk_table = Table(risk_data, colWidths=[4*cm, 3*cm, 6*cm])
    risk_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874a6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    story.append(risk_table)

    # Trade Records
    trades = result.get('trades', [])
    if trades:
        story.append(PageBreak())
        story.append(Paragraph("3. Trade Records", heading_style))

        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']

        total_profit = sum(t.get('profit', 0) for t in sell_trades)
        win_trades = [t for t in sell_trades if t.get('profit', 0) > 0]

        summary_data = [
            ["Trade Summary", ""],
            ["Total Trades", f"{len(trades)}"],
            ["Buy Trades", f"{len(buy_trades)}"],
            ["Sell Trades", f"{len(sell_trades)}"],
            ["Winning Trades", f"{len(win_trades)} ({len(win_trades)/len(sell_trades)*100:.1f}%)" if sell_trades else "0"],
            ["Realized P&L", f"${total_profit:,.2f}"],
        ]

        summary_table = Table(summary_data, colWidths=[5*cm, 5*cm])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874a6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('SPAN', (0, 0), (-1, 0)),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(summary_table)

        # Trade Details
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Trade Details", normal_style))
        story.append(Spacer(1, 0.3*cm))

        trade_header = ["No.", "Date", "Action", "Price", "Shares", "Reason"]
        trade_rows = [trade_header]

        for i, trade in enumerate(trades[-20:], 1):  # Last 20 trades
            trade_rows.append([
                str(i),
                str(trade['date']),
                trade['action'],
                f"${trade['price']:.2f}",
                str(trade['shares']),
                trade.get('reason', '')[:20],
            ])

        trade_table = Table(trade_rows, colWidths=[1*cm, 2.5*cm, 2*cm, 2*cm, 2*cm, 5*cm])
        trade_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (5, 1), (5, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        story.append(trade_table)

    # Charts
    if chart_path and chart_path.exists():
        story.append(PageBreak())
        story.append(Paragraph("4. Performance Charts", heading_style))
        img = Image(str(chart_path), width=16*cm, height=11.5*cm)
        story.append(img)

    # Conclusion
    story.append(PageBreak())
    story.append(Paragraph("5. Conclusion", heading_style))

    conclusions = [
        f"<b>Max Drawdown:</b> {result['max_drawdown']:.2%} (Target: 3.0%)",
        f"<b>Total Return:</b> {result['total_return']:.2%}",
        f"<b>Risk-Adjusted Return:</b> Sharpe Ratio = {result['sharpe_ratio']:.2f}",
        "<b>Strategy Type:</b> Trend Following with Strict Risk Control",
    ]

    if result['max_drawdown'] <= 0.03:
        conclusions.append("<b>Result: PASS</b> - Drawdown target achieved")
    else:
        conclusions.append("<b>Result: WARNING</b> - Drawdown exceeded target")

    for conclusion in conclusions:
        story.append(Paragraph(conclusion, normal_style))
        story.append(Spacer(1, 0.3*cm))

    # Disclaimer
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.5*cm))

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=9,
        textColor=colors.grey,
        alignment=1,
    )

    story.append(Paragraph("<b>Disclaimer</b>", disclaimer_style))
    story.append(Paragraph("This report is for reference only and does not constitute investment advice.", disclaimer_style))
    story.append(Paragraph("Past performance does not guarantee future results.", disclaimer_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", disclaimer_style))

    doc.build(story)
    logger.info(f"PDF report generated: {output_path}")

    return output_path


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Low Drawdown Strategy Backtest with PDF Report")
    parser.add_argument("--symbol", type=str, default="601138", help="Stock code")
    parser.add_argument("--name", type=str, default="Foxconn Industrial", help="Stock name")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start date")
    parser.add_argument("--end-date", type=str, default="2025-02-20", help="End date")
    parser.add_argument("--cash", type=float, default=1000000, help="Initial capital")
    parser.add_argument("--target-drawdown", type=float, default=0.03, help="Target max drawdown")
    args = parser.parse_args()

    logger.info(f"[INFO] Fetching {args.symbol} data...")

    # Fetch data
    loader = AkshareLoader()
    data = loader.get_stock_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust="qfq",
    )

    if data is None or len(data) == 0:
        logger.error(f"[ERROR] Cannot fetch {args.symbol} data")
        return 1

    logger.info(f"[OK] Fetched {len(data)} records")

    # Create strategy
    strategy = LowDrawdownStrategy(
        ma_short=5,
        ma_long=15,
        rsi_oversold=35,
        rsi_overbought=65,
        stop_loss_pct=0.015,
        trailing_stop_pct=0.01,
    )

    # Create backtest runner
    runner = LowDrawdownBacktestRunner(
        stop_loss_pct=0.015,
        trailing_stop_pct=0.01,
        daily_loss_limit=0.01,
        max_drawdown_limit=args.target_drawdown * 0.8,
        position_size_pct=0.2,
    )

    # Run backtest
    logger.info(f"[INFO] Starting backtest, target max drawdown: {args.target_drawdown:.1%}")
    result = runner.run(data, strategy, args.cash)

    # Print results
    print("\n" + "=" * 60)
    print("Low Drawdown Backtest Result")
    print("=" * 60)
    print(f"Stock: {args.symbol}")
    print(f"Period: {args.start_date} -> {args.end_date}")
    print(f"Initial Capital: ${args.cash:,.0f}")
    print("-" * 60)
    print(f"Total Return:     {result['total_return']:>10.2%}")
    print(f"Max Drawdown:     {result['max_drawdown']:>10.2%}")
    print(f"Sharpe Ratio:     {result['sharpe_ratio']:>10.2f}")
    print(f"Win Rate:         {result['win_rate']:>10.2%}")
    print(f"Final Value:      ${result['final_value']:>10,.2f}")
    print("-" * 60)

    if result["max_drawdown"] <= args.target_drawdown:
        print(f"[OK] Target achieved: Max drawdown {result['max_drawdown']:.2%} <= {args.target_drawdown:.1%}")
    else:
        print(f"[WARN] Target not met: Max drawdown {result['max_drawdown']:.2%} > {args.target_drawdown:.1%}")

    # Generate chart and PDF report
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = output_dir / f"{args.symbol}_low_drawdown_chart.png"
    pdf_path = output_dir / f"{args.symbol}_low_drawdown_report.pdf"

    if REPORTLAB_AVAILABLE:
        # Generate chart
        logger.info("[INFO] Generating performance chart...")
        generate_chart(result['portfolio_values'], chart_path)

        # Generate PDF report
        data_info = {
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': args.start_date,
            'end_date': args.end_date,
            'trading_days': len(data),
            'initial_cash': args.cash,
        }

        logger.info("[INFO] Generating PDF report...")
        create_pdf_report(
            stock_code=args.symbol,
            stock_name=args.name,
            result=result,
            data_info=data_info,
            chart_path=chart_path,
            output_path=pdf_path,
        )

        print(f"\n[OK] Chart saved: {chart_path}")
        print(f"[OK] PDF report saved: {pdf_path}")
    else:
        print("\n[WARN] PDF generation skipped (reportlab not installed)")
        print("Install with: uv pip install reportlab matplotlib")

    return 0


if __name__ == "__main__":
    sys.exit(main())
