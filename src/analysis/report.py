"""
PDF 报告生成器

生成专业的回测分析报告，包含：
- 执行摘要
- 回测详情
- 风险分析
- 交易分析
- 可视化图表
- AI 优化建议
"""

import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd

from src.analysis.metrics import PerformanceMetrics, calculate_all_metrics
from src.analysis.visualizer import BacktestVisualizer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """PDF 报告生成器"""

    def __init__(
        self,
        output_dir: str = "reports",
        template_name: str = "default",
    ):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
            template_name: 模板名称
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.template_name = template_name
        self.visualizer = BacktestVisualizer()

    def generate_report(
        self,
        portfolio_value: pd.Series,
        strategy_name: str,
        benchmark: Optional[pd.Series] = None,
        positions: Optional[pd.DataFrame] = None,
        trades: Optional[pd.DataFrame] = None,
        output_filename: Optional[str] = None,
        include_charts: bool = True,
    ) -> Path:
        """
        生成 PDF 报告

        Args:
            portfolio_value: 组合净值序列
            strategy_name: 策略名称
            benchmark: 基准净值序列
            positions: 持仓记录
            trades: 交易记录
            output_filename: 输出文件名
            include_charts: 是否包含图表

        Returns:
            生成的报告路径
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        # 计算指标
        metrics = calculate_all_metrics(portfolio_value, benchmark)

        # 生成图表
        charts = {}
        if include_charts:
            charts = self._generate_charts(portfolio_value, benchmark)

        # 创建 PDF
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"backtest_report_{timestamp}.pdf"

        output_path = self.output_dir / output_filename

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        # 样式
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # 居中
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
        )
        body_style = styles["Normal"]

        # 构建文档内容
        story = []

        # 标题
        story.append(Paragraph("回测分析报告", title_style))
        story.append(Paragraph(f"策略: {strategy_name}", styles["Heading2"]))
        story.append(
            Paragraph(
                f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                body_style,
            )
        )
        story.append(Spacer(1, 20))

        # 执行摘要
        story.append(Paragraph("1. 执行摘要", heading_style))
        summary_data = [
            ["指标", "数值"],
            ["回测区间", f"{metrics.start_date} 至 {metrics.end_date}"],
            ["总收益率", f"{metrics.total_return:.2%}"],
            ["年化收益率", f"{metrics.annual_return:.2%}"],
            ["最大回撤", f"{metrics.max_drawdown:.2%}"],
            ["夏普比率", f"{metrics.sharpe_ratio:.2f}"],
            ["胜率", f"{metrics.win_rate:.2%}"],
        ]

        if metrics.excess_return != 0:
            summary_data.append(["超额收益", f"{metrics.excess_return:.2%}"])

        summary_table = self._create_table(summary_data, col_widths=[4 * cm, 6 * cm])
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # 净值曲线图
        if "nav_curve" in charts:
            story.append(Paragraph("2. 净值曲线", heading_style))
            img = Image(charts["nav_curve"], width=16 * cm, height=8 * cm)
            story.append(img)
            story.append(Spacer(1, 10))

        # 风险分析
        story.append(PageBreak())
        story.append(Paragraph("3. 风险分析", heading_style))
        risk_data = [
            ["风险指标", "数值", "说明"],
            ["最大回撤", f"{metrics.max_drawdown:.2%}", "历史最高点到最低点的最大跌幅"],
            ["年化波动率", f"{metrics.volatility:.2%}", "收益率的标准差（年化）"],
            ["下行波动率", f"{metrics.downside_volatility:.2%}", "负收益的标准差（年化）"],
            ["VaR (95%)", f"{metrics.var_95:.2%}", "95% 置信度下的最大日损失"],
            ["CVaR (95%)", f"{metrics.cvar_95:.2%}", "超过 VaR 的平均损失"],
        ]
        risk_table = self._create_table(risk_data, col_widths=[3.5 * cm, 3 * cm, 7 * cm])
        story.append(risk_table)
        story.append(Spacer(1, 20))

        # 收益分布图
        if "returns_dist" in charts:
            story.append(Paragraph("收益分布", heading_style))
            img = Image(charts["returns_dist"], width=14 * cm, height=7 * cm)
            story.append(img)

        # 风险调整收益
        story.append(Spacer(1, 20))
        story.append(Paragraph("4. 风险调整收益", heading_style))
        risk_adj_data = [
            ["指标", "数值", "说明"],
            ["夏普比率", f"{metrics.sharpe_ratio:.2f}", "超额收益 / 总波动率"],
            ["索提诺比率", f"{metrics.sortino_ratio:.2f}", "超额收益 / 下行波动率"],
            ["卡玛比率", f"{metrics.calmar_ratio:.2f}", "年化收益 / 最大回撤"],
        ]

        if metrics.information_ratio != 0:
            risk_adj_data.append(["信息比率", f"{metrics.information_ratio:.2f}", "超额收益 / 跟踪误差"])

        risk_adj_table = self._create_table(risk_adj_data, col_widths=[3.5 * cm, 3 * cm, 7 * cm])
        story.append(risk_adj_table)

        # 月度收益热力图
        if "monthly_returns" in charts:
            story.append(PageBreak())
            story.append(Paragraph("5. 月度收益分析", heading_style))
            img = Image(charts["monthly_returns"], width=16 * cm, height=9 * cm)
            story.append(img)

        # 滚动指标
        if "rolling_metrics" in charts:
            story.append(PageBreak())
            story.append(Paragraph("6. 滚动绩效指标", heading_style))
            img = Image(charts["rolling_metrics"], width=14 * cm, height=10 * cm)
            story.append(img)

        # 基准比较
        if benchmark is not None and metrics.beta != 0:
            story.append(PageBreak())
            story.append(Paragraph("7. 基准比较", heading_style))
            benchmark_data = [
                ["指标", "策略", "基准"],
                ["总收益率", f"{metrics.total_return:.2%}", f"{metrics.benchmark_return:.2%}"],
                ["超额收益", f"{metrics.excess_return:.2%}", "-"],
                ["Beta", f"{metrics.beta:.2f}", "1.00"],
                ["Alpha", f"{metrics.alpha:.2%}", "-"],
                ["跟踪误差", f"{metrics.tracking_error:.2%}", "-"],
            ]
            benchmark_table = self._create_table(benchmark_data, col_widths=[4 * cm, 4 * cm, 4 * cm])
            story.append(benchmark_table)

        # 交易统计
        story.append(Spacer(1, 20))
        story.append(Paragraph("8. 交易统计", heading_style))
        trading_data = [
            ["指标", "数值"],
            ["胜率", f"{metrics.win_rate:.2%}"],
            ["盈亏比", f"{metrics.profit_loss_ratio:.2f}"],
            ["平均盈利", f"{metrics.avg_win:.4f}"],
            ["平均亏损", f"{metrics.avg_loss:.4f}"],
            ["最大连续盈利次数", f"{metrics.max_consecutive_wins}"],
            ["最大连续亏损次数", f"{metrics.max_consecutive_losses}"],
        ]
        trading_table = self._create_table(trading_data, col_widths=[5 * cm, 5 * cm])
        story.append(trading_table)

        # 页脚
        story.append(Spacer(1, 30))
        story.append(Paragraph("---", body_style))
        story.append(
            Paragraph(
                f"报告由 Qlib AI Strategy 系统自动生成 | {datetime.now().strftime('%Y-%m-%d')}",
                ParagraphStyle("Footer", fontSize=9, textColor=colors.grey),
            )
        )

        # 生成 PDF
        doc.build(story)

        logger.info(f"报告已生成: {output_path}")
        return output_path

    def _generate_charts(
        self,
        portfolio_value: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> dict[str, Path]:
        """生成所有图表"""
        charts = {}

        # 创建临时图表目录
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # 净值曲线
        fig = self.visualizer.plot_nav_curve(portfolio_value, benchmark, show_drawdown=True)
        charts["nav_curve"] = self.visualizer.save_figure(fig, "nav_curve.png", figures_dir)

        # 收益分布
        fig = self.visualizer.plot_returns_distribution(portfolio_value)
        charts["returns_dist"] = self.visualizer.save_figure(fig, "returns_dist.png", figures_dir)

        # 月度收益
        try:
            fig = self.visualizer.plot_monthly_returns(portfolio_value)
            charts["monthly_returns"] = self.visualizer.save_figure(fig, "monthly_returns.png", figures_dir)
        except Exception as e:
            logger.warning(f"月度收益图生成失败: {e}")

        # 滚动指标
        try:
            fig = self.visualizer.plot_rolling_metrics(portfolio_value, benchmark=benchmark)
            charts["rolling_metrics"] = self.visualizer.save_figure(fig, "rolling_metrics.png", figures_dir)
        except Exception as e:
            logger.warning(f"滚动指标图生成失败: {e}")

        return charts

    def _create_table(
        self,
        data: list[list],
        col_widths: Optional[list] = None,
    ):
        """创建样式化的表格"""
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        table = Table(data, colWidths=col_widths)

        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.2, 0.4, 0.6)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ])

        # 交替行颜色
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.add("BACKGROUND", (0, i), (-1, i), colors.white)

        table.setStyle(style)
        return table


if __name__ == "__main__":
    import numpy as np

    # 测试
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = np.random.normal(0.0005, 0.015, 252)
    portfolio_value = pd.Series(1000000 * (1 + returns).cumprod(), index=dates)

    generator = ReportGenerator()
    report_path = generator.generate_report(
        portfolio_value=portfolio_value,
        strategy_name="测试策略",
    )

    print(f"报告已生成: {report_path}")
