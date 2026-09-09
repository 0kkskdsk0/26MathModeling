#!/usr/bin/env python3
"""Compile a CUMCM LaTeX project with XeLaTeX and summarize likely fixes."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ERROR_PATTERNS = [
    re.compile(r"^!(?P<msg>.+)$"),
    re.compile(r"^LaTeX Error: (?P<msg>.+)$"),
    re.compile(r"^(?P<file>[^:\s][^:]*\.tex):(?P<line>\d+): (?P<msg>.+)$"),
]

WARNING_PATTERNS = [
    re.compile(r"LaTeX Warning: (?P<msg>.+)"),
    re.compile(r"Package (?P<pkg>[\w-]+) Warning: (?P<msg>.+)"),
    re.compile(r"(?P<msg>Overfull \\hbox .+)"),
    re.compile(r"(?P<msg>Underfull \\hbox .+)"),
]


def run_pass(project: Path, main: str, halt_on_error: bool) -> subprocess.CompletedProcess[str]:
    cmd = [
        "xelatex",
        "-interaction=nonstopmode",
        "-file-line-error",
    ]
    if halt_on_error:
        cmd.append("-halt-on-error")
    cmd.append(main)
    return subprocess.run(
        cmd,
        cwd=project,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def parse_log(text: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        for pattern in ERROR_PATTERNS:
            match = pattern.search(stripped)
            if match:
                item = {k: v for k, v in match.groupdict().items() if v}
                item.setdefault("line_in_log", str(idx + 1))
                context = "\n".join(lines[idx : min(idx + 5, len(lines))])
                item["context"] = context
                errors.append(item)
                break
        for pattern in WARNING_PATTERNS:
            match = pattern.search(stripped)
            if match:
                item = {k: v for k, v in match.groupdict().items() if v}
                item.setdefault("line_in_log", str(idx + 1))
                warnings.append(item)
                break
    return errors, warnings


def read_log(project: Path, main_path: Path) -> str:
    log_path = project / (main_path.stem + ".log")
    if log_path.exists():
        return log_path.read_text(encoding="utf-8", errors="replace")
    return ""


def pdf_info(project: Path, main_path: Path) -> dict[str, object]:
    pdf = project / (main_path.stem + ".pdf")
    info: dict[str, object] = {
        "pdf": str(pdf),
        "exists": pdf.exists(),
    }
    if not pdf.exists():
        return info
    info["size_bytes"] = pdf.stat().st_size
    if shutil.which("pdfinfo"):
        proc = subprocess.run(
            ["pdfinfo", str(pdf)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        page_match = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.MULTILINE)
        if page_match:
            info["pages"] = int(page_match.group(1))
    return info


def likely_fix(error: dict[str, str]) -> str:
    msg = error.get("msg", "")
    context = error.get("context", "")
    joined = f"{msg}\n{context}"
    if "Undefined control sequence" in joined:
        return "检查命令拼写或缺失宏包；若是复制来的符号，优先改成模板已支持的标准 LaTeX 命令。"
    if "Missing $" in joined or "extra }" in joined or "math mode" in joined:
        return "检查该行附近的数学分隔符、上下标、花括号是否成对。"
    if "File `" in joined and "not found" in joined or "Cannot determine size of graphic" in joined:
        return "检查图片路径、扩展名和大小写；图片建议放在 figures/ 并用相对路径引用。"
    if "Misplaced alignment tab character &" in joined:
        return "普通文本里的 & 要写成 \\&；只有表格或 align 环境中才能直接用 &。"
    if "Runaway argument" in joined:
        return "通常是花括号未闭合，或 % 把行尾注释掉导致参数跨行失控。"
    if "Unicode" in joined:
        return "替换不可见字符、全角反斜杠或模板字体不支持的特殊符号。"
    return "先定位日志中的文件和行号，检查该行及前后 3 行的命令、括号、数学模式和特殊字符。"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".", help="LaTeX project root")
    parser.add_argument("--main", default="main.tex", help="main TeX file relative to project")
    parser.add_argument("--passes", type=int, default=2, help="XeLaTeX passes")
    parser.add_argument("--no-halt-on-error", action="store_true", help="continue after hard errors")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    main_path = Path(args.main)
    if not (project / main_path).exists():
        print(f"ERROR: main file not found: {project / main_path}", file=sys.stderr)
        return 2
    if not shutil.which("xelatex"):
        print("ERROR: xelatex not found on PATH", file=sys.stderr)
        return 2

    runs = []
    combined_output = ""
    success = True
    for pass_no in range(1, args.passes + 1):
        proc = run_pass(project, args.main, halt_on_error=not args.no_halt_on_error)
        combined_output += f"\n===== XeLaTeX pass {pass_no} =====\n{proc.stdout}"
        runs.append({"pass": pass_no, "returncode": proc.returncode})
        if proc.returncode != 0:
            success = False
            break

    log_text = read_log(project, main_path)
    parse_target = log_text or combined_output
    errors, warnings = parse_log(parse_target)
    for error in errors:
        error["likely_fix"] = likely_fix(error)

    result = {
        "project": str(project),
        "main": args.main,
        "success": success and not errors,
        "runs": runs,
        "errors": errors[:20],
        "warnings": warnings[:50],
        "pdf": pdf_info(project, main_path),
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Project: {result['project']}")
        print(f"Main: {args.main}")
        print(f"Success: {result['success']}")
        print(f"PDF: {result['pdf']}")
        if errors:
            print("\nHard errors:")
            for item in errors[:8]:
                loc = ""
                if "file" in item or "line" in item:
                    loc = f"{item.get('file', '')}:{item.get('line', '')} "
                print(f"- {loc}{item.get('msg', '').strip()}")
                print(f"  Fix: {item['likely_fix']}")
        if warnings:
            print("\nWarnings:")
            for item in warnings[:12]:
                print(f"- {item.get('msg', '').strip()}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
