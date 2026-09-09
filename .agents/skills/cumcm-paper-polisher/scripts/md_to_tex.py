#!/usr/bin/env python3
"""Convert simple Markdown paper fragments to CUMCM-friendly LaTeX snippets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SECTION_COMMANDS = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]


def protect_math(text: str) -> tuple[str, list[str]]:
    parts: list[str] = []

    def repl(match: re.Match[str]) -> str:
        parts.append(match.group(0))
        return f"@@MATH{len(parts)-1}@@"

    pattern = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\$[^$\n]+\$", re.S)
    return pattern.sub(repl, text), parts


def restore_math(text: str, parts: list[str]) -> str:
    for idx, part in enumerate(parts):
        text = text.replace(f"@@MATH{idx}@@", part)
    return text


def escape_text(text: str) -> str:
    protected, math_parts = protect_math(text)
    protected = re.sub(r"(?<!\\)&", r"\\&", protected)
    protected = re.sub(r"(?<!\\)%", r"\\%", protected)
    protected = re.sub(r"(?<!\\)#", r"\\#", protected)
    protected = re.sub(r"(?<!\\)_(?![A-Za-z0-9]*\})", r"\\_", protected)
    return restore_math(protected, math_parts)


def convert_inline(text: str) -> str:
    text = escape_text(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    return text


def is_table_start(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    return "|" in lines[i] and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i + 1])


def split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def convert_table(rows: list[str], label_prefix: str) -> str:
    header = split_table_row(rows[0])
    body = [split_table_row(row) for row in rows[2:]]
    cols = "l" + "Y" * max(0, len(header) - 1)
    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{待填写表题}",
        rf"  \label{{tab:{label_prefix}}}",
        rf"  \begin{{tabularx}}{{\textwidth}}{{{cols}}}",
        r"    \toprule",
        "    " + " & ".join(convert_inline(cell) for cell in header) + r" \\",
        r"    \midrule",
    ]
    for row in body:
        row = row + [""] * (len(header) - len(row))
        lines.append("    " + " & ".join(convert_inline(cell) for cell in row[: len(header)]) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabularx}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def convert_image(line: str) -> str | None:
    match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
    if not match:
        return None
    caption = match.group(1).strip() or "待填写图题"
    path = match.group(2).strip()
    stem = Path(path).stem
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-") or "figure"
    return "\n".join(
        [
            r"\begin{figure}[!htbp]",
            r"  \centering",
            rf"  \includegraphics[width=0.78\textwidth]{{{path}}}",
            rf"  \caption{{{convert_inline(caption)}}}",
            rf"  \label{{fig:{label}}}",
            r"\end{figure}",
        ]
    )


def convert_markdown(text: str, base_level: int) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    code_lang = ""
    list_stack: list[str] = []
    table_count = 1

    def close_lists() -> None:
        while list_stack:
            out.append(r"\end{" + list_stack.pop() + "}")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                close_lists()
                in_code = True
                code_lang = stripped[3:].strip() or "text"
                lang_opt = "" if code_lang == "text" else f"[language={code_lang}]"
                out.append(r"\begin{lstlisting}" + lang_opt)
            else:
                out.append(r"\end{lstlisting}")
                in_code = False
            i += 1
            continue

        if in_code:
            out.append(line)
            i += 1
            continue

        if is_table_start(lines, i):
            close_lists()
            rows = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i].strip():
                rows.append(lines[i])
                i += 1
            out.append(convert_table(rows, f"md-table-{table_count}"))
            table_count += 1
            continue

        image = convert_image(stripped)
        if image:
            close_lists()
            out.append(image)
            i += 1
            continue

        heading = re.match(r"^(#{1,5})\s+(.+)$", stripped)
        if heading:
            close_lists()
            level = min(len(heading.group(1)) + base_level - 1, len(SECTION_COMMANDS)) - 1
            out.append(rf"\{SECTION_COMMANDS[level]}{{{convert_inline(heading.group(2).strip())}}}")
            i += 1
            continue

        unordered = re.match(r"^[-*+]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if unordered or ordered:
            env = "itemize" if unordered else "enumerate"
            content = (unordered or ordered).group(1)
            if not list_stack or list_stack[-1] != env:
                close_lists()
                list_stack.append(env)
                out.append(r"\begin{" + env + "}")
            out.append(r"  \item " + convert_inline(content))
            i += 1
            continue

        if stripped == "":
            close_lists()
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        close_lists()
        out.append(convert_inline(stripped))
        i += 1

    close_lists()
    if in_code:
        out.append(r"\end{lstlisting}")
    return "\n".join(out).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Markdown file")
    parser.add_argument("-o", "--output", help="TeX output file; stdout if omitted")
    parser.add_argument("--base-level", type=int, default=1, help="# maps to this LaTeX section level")
    args = parser.parse_args()

    source = Path(args.input)
    text = source.read_text(encoding="utf-8", errors="replace")
    tex = convert_markdown(text, args.base_level)
    if args.output:
        Path(args.output).write_text(tex, encoding="utf-8")
    else:
        print(tex, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
