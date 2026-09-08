---
name: edit-notebook-with-nbformat
description: Safely read, modify, and save Jupyter Notebook (.ipynb) files using the nbformat library. Use when creating, editing, appending, or restructuring notebook cells. Avoid SearchReplace or full-file rewrite to preserve execution counts, outputs, and kernel metadata.
---

# 使用 nbformat 编辑 Jupyter Notebook

## 适用场景

当需要创建、编辑、追加、插入、删除或重新排列 `.ipynb` 文件中的 cell 时使用本 skill。

## 核心规则

1. **必须使用 `nbformat`** 进行结构化读写。
2. **禁止对 `.ipynb` 使用 `SearchReplace`**。
3. **尽量避免 `DeleteFile + Write` 重建**，除非必要。
4. **编辑后必须验证**：使用 `nbformat.read()` 重新读取确认无损坏。

## 快速开始

```python
import nbformat

nb = nbformat.read('analysis.ipynb', as_version=4)

# 在这里修改 cells，例如：
# nb.cells.append(nbformat.v4.new_code_cell('print("hello")'))

nbformat.write(nb, 'analysis.ipynb')

# 验证
nbformat.read('analysis.ipynb', as_version=4)
```

## 参考资源

- 详细指南和示例代码：[references/guide.md](references/guide.md)
- 可复用脚本模板：[scripts/edit_notebook.py](scripts/edit_notebook.py)
