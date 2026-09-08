"""
使用 nbformat 安全编辑 Jupyter Notebook 的可复用模板。

用法：
    uv run python edit_notebook.py <notebook_path>

修改下方的 edit_notebook() 函数以执行你想要的改动。
"""

import argparse
import sys

import nbformat


def edit_notebook(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """
    在原地修改 notebook 并返回。

    示例：追加一个 markdown cell 和一个 code cell。
    """
    md_cell = nbformat.v4.new_markdown_cell("## 新章节\n\n由 edit_notebook.py 添加")
    code_cell = nbformat.v4.new_code_cell('print("hello from edit_notebook.py")')

    nb.cells.append(md_cell)
    nb.cells.append(code_cell)

    # 示例：按内容匹配修改已有 cell
    # for cell in nb.cells:
    #     if cell.cell_type == "code" and "def match_alarm" in cell.source:
    #         cell.source = cell.source.replace("return None", 'return f"未知告警({row[\'idx\']})"')

    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 nbformat 安全编辑 Jupyter Notebook")
    parser.add_argument("notebook", help=".ipynb 文件路径")
    args = parser.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    print(f"读取 {args.notebook}: {len(nb.cells)} 个 cells")

    nb = edit_notebook(nb)
    print(f"编辑后: {len(nb.cells)} 个 cells")

    nbformat.write(nb, args.notebook)
    print(f"已写入 {args.notebook}")

    # 验证
    nbformat.read(args.notebook, as_version=4)
    print("验证通过")

    return 0


if __name__ == "__main__":
    sys.exit(main())
