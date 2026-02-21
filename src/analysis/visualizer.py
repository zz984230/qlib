"""
可视化模块

生成回测相关图表：
- 净值曲线图
- 回撤图
- 月度收益热力图
- 收益分布图
- 滚动指标图
- 因子分析图
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """回测可视化器"""

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: tuple[int, int] = (12, 6),
        dpi: int = 150,
    ):
        """初始化可视化器"""
        self.figsize = figsize
        self.dpi = dpi

        # 设置样式
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid")

        # 颜色配置
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "positive": "#2ca02c",
            "negative": "#d62728",
            "benchmark": "#9467bd",
        }

    def plot_nav_curve(
        self,
        portfolio_value: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "净值曲线",
        show_drawdown: bool = True,
    ) -> plt.Figure:
        """
        绘制净值曲线

        Args:
            portfolio_value: 组合净值序列
            benchmark: 基准净值序列
            title: 图表标题
            show_drawdown: 是否显示回撤区域

        Returns:
            matplotlib Figure 对象
        """
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=self.figsize,
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )
        else:
            fig, ax1 = plt.subplots(figsize=self.figsize)
            ax2 = None

        # 标准化净值
        nav = portfolio_value / portfolio_value.iloc[0]
        ax1.plot(nav.index, nav.values, label="策略", color=self.colors["primary"], linewidth=1.5)

        # 基准
        if benchmark is not None and len(benchmark) > 0:
            benchmark_nav = benchmark / benchmark.iloc[0]
            ax1.plot(
                benchmark_nav.index, benchmark_nav.values,
                label="基准", color=self.colors["benchmark"],
                linewidth=1.5, alpha=0.7, linestyle="--"
            )

        ax1.set_ylabel("净值")
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 回撤区域
        if show_drawdown and ax2 is not None:
            cummax = nav.cummax()
            drawdown = (cummax - nav) / cummax

            ax2.fill_between(
                drawdown.index, 0, drawdown.values,
                color=self.colors["negative"], alpha=0.3
            )
            ax2.set_ylabel("回撤")
            ax2.set_xlabel("日期")
            ax2.grid(True, alpha=0.3)

            # 标注最大回撤
            max_dd_idx = drawdown.idxmax()
            max_dd = drawdown.max()
            ax2.annotate(
                f"最大回撤: {max_dd:.2%}",
                xy=(max_dd_idx, max_dd),
                xytext=(max_dd_idx, max_dd + 0.05),
                fontsize=9,
                ha="center",
            )

        plt.tight_layout()
        return fig

    def plot_returns_distribution(
        self,
        portfolio_value: pd.Series,
        title: str = "日收益率分布",
    ) -> plt.Figure:
        """
        绘制收益率分布直方图

        Args:
            portfolio_value: 组合净值序列
            title: 图表标题

        Returns:
            matplotlib Figure 对象
        """
        returns = portfolio_value.pct_change().dropna()

        fig, ax = plt.subplots(figsize=self.figsize)

        # 直方图
        n, bins, patches = ax.hist(
            returns, bins=50, density=True,
            alpha=0.7, color=self.colors["primary"],
            edgecolor="white", linewidth=0.5
        )

        # 正态分布拟合
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(
            x, 1 / (sigma * np.sqrt(2 * np.pi)) *
            np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)),
            color=self.colors["negative"], linewidth=2,
            label=f"正态分布 (μ={mu:.4f}, σ={sigma:.4f})"
        )

        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("日收益率")
        ax.set_ylabel("频率密度")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_monthly_returns(
        self,
        portfolio_value: pd.Series,
        title: str = "月度收益热力图",
    ) -> plt.Figure:
        """
        绘制月度收益热力图

        Args:
            portfolio_value: 组合净值序列
            title: 图表标题

        Returns:
            matplotlib Figure 对象
        """
        returns = portfolio_value.pct_change().dropna()

        # 计算月度收益
        monthly_returns = returns.resample("M").apply(
            lambda x: (1 + x).prod() - 1
        )

        # 创建年月矩阵
        monthly_df = pd.DataFrame({
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        })

        pivot = monthly_df.pivot(index="year", columns="month", values="return")

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.8)))

        sns.heatmap(
            pivot * 100,  # 转换为百分比
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "收益率 (%)"},
        )

        # 设置标签
        month_labels = ["1月", "2月", "3月", "4月", "5月", "6月",
                        "7月", "8月", "9月", "10月", "11月", "12月"]
        ax.set_xticklabels(month_labels[:len(pivot.columns)], rotation=45, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("年份")
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    def plot_rolling_metrics(
        self,
        portfolio_value: pd.Series,
        window: int = 60,
        benchmark: Optional[pd.Series] = None,
        title: str = "滚动绩效指标",
    ) -> plt.Figure:
        """
        绘制滚动绩效指标

        Args:
            portfolio_value: 组合净值序列
            window: 滚动窗口
            benchmark: 基准净值序列
            title: 图表标题

        Returns:
            matplotlib Figure 对象
        """
        returns = portfolio_value.pct_change().dropna()

        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)

        # 滚动收益
        rolling_return = returns.rolling(window).mean() * 252
        axes[0].plot(rolling_return.index, rolling_return.values,
                     color=self.colors["primary"], linewidth=1.5)
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)
        axes[0].set_ylabel("年化收益")
        axes[0].set_title("滚动收益 (60日)", fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 滚动波动率
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        axes[1].plot(rolling_vol.index, rolling_vol.values,
                     color=self.colors["secondary"], linewidth=1.5)
        axes[1].set_ylabel("年化波动率")
        axes[1].set_title("滚动波动率 (60日)", fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # 滚动夏普
        rolling_sharpe = (rolling_return - 0.03) / rolling_vol
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values,
                     color=self.colors["positive"], linewidth=1.5)
        axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.5)
        axes[2].axhline(1, color="gray", linestyle=":", linewidth=0.5)
        axes[2].set_ylabel("夏普比率")
        axes[2].set_xlabel("日期")
        axes[2].set_title("滚动夏普比率 (60日)", fontsize=10)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig

    def plot_underwater(
        self,
        portfolio_value: pd.Series,
        title: str = "水下图 (Underwater Plot)",
    ) -> plt.Figure:
        """
        绘制水下图

        Args:
            portfolio_value: 组合净值序列
            title: 图表标题

        Returns:
            matplotlib Figure 对象
        """
        nav = portfolio_value / portfolio_value.iloc[0]
        cummax = nav.cummax()
        drawdown = (cummax - nav) / cummax

        fig, ax = plt.subplots(figsize=self.figsize)

        # 填充区域
        ax.fill_between(
            drawdown.index, 0, -drawdown.values,
            color=self.colors["negative"], alpha=0.5
        )

        # 标注主要回撤
        significant_dd = drawdown[drawdown > 0.05]  # 超过 5% 的回撤

        for idx, dd in significant_dd.groupby(
            (significant_dd.diff().isna() | (significant_dd.diff() == 0)).cumsum()
        ):
            if len(dd) > 5:  # 只标注持续时间较长的回撤
                min_idx = dd.idxmax()
                ax.annotate(
                    f"{dd.max():.1%}",
                    xy=(min_idx, -dd.max()),
                    xytext=(min_idx, -dd.max() - 0.03),
                    fontsize=8,
                    ha="center",
                )

        ax.set_ylabel("回撤")
        ax.set_xlabel("日期")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_compare_strategies(
        self,
        strategies: dict[str, pd.Series],
        title: str = "策略比较",
    ) -> plt.Figure:
        """
        比较多个策略

        Args:
            strategies: 策略名称到净值序列的映射
            title: 图表标题

        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.tab10.colors

        for i, (name, values) in enumerate(strategies.items()):
            nav = values / values.iloc[0]
            ax.plot(
                nav.index, nav.values,
                label=name, color=colors[i % len(colors)],
                linewidth=1.5
            )

        ax.set_ylabel("净值")
        ax.set_xlabel("日期")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        output_dir: str = "reports/figures",
    ) -> Path:
        """
        保存图表到文件

        Args:
            fig: matplotlib Figure 对象
            filename: 文件名
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"图表已保存: {filepath}")
        return filepath

    def figure_to_bytes(self, fig: plt.Figure) -> bytes:
        """将图表转换为字节流"""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = np.random.normal(0.0005, 0.015, 252)
    portfolio_value = pd.Series(1000000 * (1 + returns).cumprod(), index=dates)

    viz = BacktestVisualizer()

    # 测试各种图表
    fig1 = viz.plot_nav_curve(portfolio_value, show_drawdown=True)
    viz.save_figure(fig1, "test_nav_curve.png")

    fig2 = viz.plot_returns_distribution(portfolio_value)
    viz.save_figure(fig2, "test_returns_dist.png")

    print("图表生成测试完成!")
