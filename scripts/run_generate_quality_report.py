"""Generate the integrated M7 manufacturing quality summary report."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.generate_quality_report import (
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    build_summary_payload,
    render_markdown_report,
    write_report_outputs,
)


def main() -> int:
    output_dir = PROJECT_ROOT / "reports"
    try:
        payload = build_summary_payload(PROJECT_ROOT)
        markdown = render_markdown_report(payload)
        write_report_outputs(payload, markdown, output_dir)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    markdown_path = output_dir / REPORT_MARKDOWN_NAME
    json_path = output_dir / REPORT_JSON_NAME
    print(f"Wrote Markdown report: {markdown_path}")
    print(f"Wrote JSON summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
