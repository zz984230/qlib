"""报告生成模块

提供 HTML 报告生成功能。
"""

from src.report.html_generator import (
    HtmlReportGenerator,
    generate_summary_html,
)

__all__ = [
    "HtmlReportGenerator",
    "generate_summary_html",
]
