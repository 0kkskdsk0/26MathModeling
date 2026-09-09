#!/usr/bin/env python3
"""Static checks for a section-based CUMCM LaTeX project."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Finding:
    severity: str
    code: str
    file: str
    line: int | None
    message: str
    suggestion: str


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def add(findings: list[Finding], severity: str, code: str, file: Path, line: int | None, message: str, suggestion: str) -> None:
    findings.append(Finding(severity, code, str(file), line, message, suggestion))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_verbatim_like(text: str) -> str:
    """Blank verbatim-like environments while preserving line numbers."""

    def repl(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    for env in ["lstlisting", "verbatim", "tcode"]:
        text = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", repl, text, flags=re.S)
    return text


def active_inputs(main_text: str) -> list[str]:
    inputs = []
    for match in re.finditer(r"(?m)^(?!\s*%)\s*\\(?:input|include)\{([^}]+)\}", main_text):
        target = match.group(1)
        if not target.endswith(".tex"):
            target += ".tex"
        inputs.append(target)
    return inputs


def collect_tex_files(project: Path, main_file: Path, main_text: str) -> list[Path]:
    files = [main_file]
    for target in active_inputs(main_text):
        path = project / target
        if path.exists():
            files.append(path)
    sections = project / "sections"
    if sections.exists():
        for path in sorted(sections.glob("*.tex")):
            if path not in files:
                files.append(path)
    return files


def split_keys(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def graphic_exists(project: Path, image_path: str) -> bool:
    raw = image_path.strip()
    direct = project / raw
    if direct.exists():
        return True
    if direct.suffix:
        return False
    for suffix in [".pdf", ".png", ".jpg", ".jpeg", ".eps"]:
        if (project / f"{raw}{suffix}").exists():
            return True
    return False


def environment_blocks(text: str, env: str) -> list[tuple[int, str]]:
    pattern = re.compile(rf"\\begin\{{{re.escape(env)}\}}(.*?)\\end\{{{re.escape(env)}\}}", re.S)
    return [(m.start(), m.group(0)) for m in pattern.finditer(text)]


def pdf_pages_and_size(pdf: Path) -> tuple[int | None, int | None]:
    if not pdf.exists():
        return None, None
    pages = None
    if shutil.which("pdfinfo"):
        proc = subprocess.run(["pdfinfo", str(pdf)], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        match = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.MULTILINE)
        if match:
            pages = int(match.group(1))
    return pages, pdf.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".", help="LaTeX project root")
    parser.add_argument("--main", default="main.tex", help="main TeX file relative to project")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    main_file = project / args.main
    findings: list[Finding] = []

    if not main_file.exists():
        add(findings, "ERROR", "missing-main", main_file, None, "主 TeX 文件不存在。", "确认 --main 参数或把 main.tex 放在项目根目录。")
        print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2) if args.json else findings[0].message)
        return 1

    main_text = read(main_file)
    if "cumcmthesis" not in main_text:
        add(findings, "WARN", "template-class", main_file, None, "main.tex 未检测到 cumcmthesis 文档类。", "如果使用官方/自定义模板可忽略；否则建议使用 2026 CUMCM 模板。")
    if r"\tableofcontents" in main_text:
        add(findings, "ERROR", "toc-active", main_file, line_number(main_text, main_text.find(r"\tableofcontents")), "正文包含目录命令。", "国赛提交正文通常不放目录，删除或注释 \\tableofcontents。")

    for match in re.finditer(r"(?m)^(?!\s*%)\s*\\(?:input|include)\{([^}]+)\}", main_text):
        target = match.group(1)
        tex_target = target if target.endswith(".tex") else f"{target}.tex"
        if not (project / tex_target).exists():
            add(findings, "ERROR", "missing-input", main_file, line_number(main_text, match.start()), f"引用的章节文件不存在：{tex_target}", "检查文件名、大小写和 sections/ 路径。")
        if "99_latex_examples" in target:
            add(findings, "WARN", "examples-active", main_file, line_number(main_text, match.start()), "示例章节仍处于启用状态。", "正式提交前删除或注释 \\input{sections/99_latex_examples}。")

    tex_files = collect_tex_files(project, main_file, main_text)
    all_text = ""
    label_defs: dict[str, tuple[Path, int]] = {}
    refs: list[tuple[str, Path, int]] = []
    cites: list[tuple[str, Path, int]] = []
    bibitems: dict[str, tuple[Path, int]] = {}

    for path in tex_files:
        text = read(path)
        check_text = strip_verbatim_like(text)
        all_text += "\n" + text
        rel = path.relative_to(project) if path.is_relative_to(project) else path

        for forbidden in [r"\documentclass", r"\begin{document}", r"\end{document}"]:
            idx = check_text.find(forbidden)
            if idx >= 0 and path != main_file:
                add(findings, "ERROR", "section-preamble", rel, line_number(check_text, idx), f"章节文件中出现 {forbidden}。", "章节文件只能放正文片段，不要放完整文档结构。")

        for match in re.finditer(r"TODO|待填写|这里写|【[^】]*】|\?\?\?", check_text):
            add(findings, "WARN", "placeholder", rel, line_number(check_text, match.start()), f"疑似占位内容：{match.group(0)[:30]}", "替换为真实内容，未知数值要标注待确认。")

        for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", check_text):
            image_path = match.group(1)
            if not graphic_exists(project, image_path):
                add(findings, "ERROR", "missing-figure", rel, line_number(check_text, match.start()), f"图片文件不存在：{image_path}", "把图片放入 figures/，或修正 \\includegraphics 路径。")
            if re.search(r"\s|[，。；：！]", image_path):
                add(findings, "WARN", "figure-name", rel, line_number(check_text, match.start()), f"图片路径含空格或中文标点：{image_path}", "建议使用 ASCII 文件名，例如 figures/q1_cost_compare.png。")

        for env in ["figure", "table"]:
            for idx, block in environment_blocks(check_text, env):
                line = line_number(check_text, idx)
                if r"\caption" not in block:
                    add(findings, "WARN", f"{env}-caption", rel, line, f"{env} 环境缺少 caption。", "为每张图/表添加具体标题。")
                if r"\label" not in block:
                    add(findings, "WARN", f"{env}-label", rel, line, f"{env} 环境缺少 label。", "添加稳定 label，便于正文引用。")
                caption_pos = block.find(r"\caption")
                label_pos = block.find(r"\label")
                if caption_pos >= 0 and label_pos >= 0 and label_pos < caption_pos:
                    add(findings, "WARN", f"{env}-label-order", rel, line, f"{env} 中 label 在 caption 前。", "建议把 \\label 放在 \\caption 后面。")

        for match in re.finditer(r"\\label\{([^}]+)\}", check_text):
            key = match.group(1).strip()
            loc = (rel, line_number(check_text, match.start()))
            if key in label_defs:
                prev_file, prev_line = label_defs[key]
                add(findings, "ERROR", "duplicate-label", rel, loc[1], f"重复 label：{key}", f"第一次出现在 {prev_file}:{prev_line}，请改成唯一名称。")
            else:
                label_defs[key] = loc

        for match in re.finditer(r"\\(?:cref|Cref|ref|eqref)\{([^}]+)\}", check_text):
            for key in split_keys(match.group(1)):
                refs.append((key, rel, line_number(check_text, match.start())))

        for match in re.finditer(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", check_text):
            for key in split_keys(match.group(1)):
                cites.append((key, rel, line_number(check_text, match.start())))

        for match in re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", check_text):
            bibitems[match.group(1).strip()] = (rel, line_number(check_text, match.start()))

        for match in re.finditer(r"(?<!\\)%", check_text):
            before = check_text[max(0, match.start() - 12) : match.start()]
            if re.search(r"\d\s*$", before):
                add(findings, "WARN", "percent", rel, line_number(check_text, match.start()), "数字后的 % 可能未转义。", "正文百分号写成 \\%，例如 20\\%。")

    for key, path, line in refs:
        if key not in label_defs:
            add(findings, "WARN", "undefined-ref", path, line, f"引用未定义 label：{key}", "确认目标图表/公式是否有对应 \\label。")

    for key, path, line in cites:
        if key not in bibitems:
            add(findings, "WARN", "undefined-cite", path, line, f"引用未定义文献：{key}", "在参考文献中补充对应 \\bibitem，或修正 cite key。")

    cited_keys = {key for key, _, _ in cites}
    for key, (path, line) in bibitems.items():
        if key not in cited_keys:
            add(findings, "INFO", "uncited-bibitem", path, line, f"参考文献未在正文引用：{key}", "只保留正文真正引用的文献，或补充正文引用。")

    for command in [r"\schoolname", r"\membera", r"\memberb", r"\memberc", r"\supervisor"]:
        pattern = re.compile(re.escape(command) + r"\{([^}]*)\}")
        for match in pattern.finditer(main_text):
            if match.group(1).strip():
                add(findings, "WARN", "identity-metadata", main_file.relative_to(project), line_number(main_text, match.start()), f"{command} 中填有身份信息。", "电子提交 withoutpreface 通常不显示，但最终提交前仍建议确认是否应清空。")

    pdf = project / (Path(args.main).stem + ".pdf")
    pages, size = pdf_pages_and_size(pdf)
    if size is not None and size > 20 * 1024 * 1024:
        add(findings, "WARN", "pdf-size", pdf.relative_to(project), None, "PDF 超过 20 MB。", "压缩图片或减少不必要的位图。")
    if pages is not None and pages > 31:
        add(findings, "INFO", "pdf-pages", pdf.relative_to(project), None, f"PDF 共 {pages} 页。", "正文页数限制需结合摘要和附录位置人工判断。")

    severity_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    findings.sort(key=lambda f: (severity_order.get(f.severity, 9), f.file, f.line or 0, f.code))

    if args.json:
        print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2))
    else:
        if not findings:
            print("No findings.")
        else:
            for finding in findings:
                loc = f"{finding.file}" + (f":{finding.line}" if finding.line else "")
                print(f"[{finding.severity}] {finding.code} {loc}")
                print(f"  {finding.message}")
                print(f"  建议：{finding.suggestion}")

    return 1 if any(f.severity == "ERROR" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
