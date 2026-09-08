# Jupyter Notebook 结构化编辑指南

本指南说明如何使用 `nbformat` 库安全地读写 `.ipynb` 文件。

## 为什么用 nbformat

`.ipynb` 文件底层是 JSON，包含 cells、metadata、execution counts、outputs 等结构。直接用文本替换（SearchReplace）或全文重写很容易破坏格式，导致 Notebook 无法打开或丢失执行结果。`nbformat` 提供结构化 API，可以安全地操作 Notebook 对象。

## 运行环境

项目已使用 `uv` 管理依赖，`nbformat` 随 Jupyter 一起安装。执行脚本时使用：

```bash
uv run python your_script.py
```

## 基础操作

### 读取 Notebook

```python
import nbformat

nb = nbformat.read('analysis.ipynb', as_version=4)
print(f'cell 总数: {len(nb.cells)}')
```

### 追加 Cell

```python
import nbformat

nb = nbformat.read('analysis.ipynb', as_version=4)

md_cell = nbformat.v4.new_markdown_cell('## 新章节\n\n说明文字。')
code_cell = nbformat.v4.new_code_cell('print("hello")')

nb.cells.append(md_cell)
nb.cells.append(code_cell)

nbformat.write(nb, 'analysis.ipynb')
```

### 指定位置插入 Cell

```python
import nbformat

nb = nbformat.read('analysis.ipynb', as_version=4)
new_cell = nbformat.v4.new_code_cell('# 在索引 3 之前插入\nimport numpy as np')
nb.cells.insert(3, new_cell)

nbformat.write(nb, 'analysis.ipynb')
```

### 修改已有 Cell

通过内容定位，然后修改 `cell.source`：

```python
import nbformat

nb = nbformat.read('analysis.ipynb', as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code' and 'def match_alarm' in cell.source:
        cell.source = cell.source.replace(
            'return None',
            'return f"未知告警({row[\'idx\']})"'
        )

nbformat.write(nb, 'analysis.ipynb')
```

### 删除 Cell

```python
import nbformat

nb = nbformat.read('analysis.ipynb', as_version=4)

# 删除最后一个 cell
nb.cells.pop()

# 或删除最后两个 cell
# nb.cells = nb.cells[:-2]

nbformat.write(nb, 'analysis.ipynb')
```

### 验证 Notebook 合法性

```python
import nbformat

nb = nbformat.read('analysis.ipynb', as_version=4)
print(f'合法的 notebook，共 {len(nb.cells)} 个 cells')
```

## 完整示例：未识别告警兜底

在 `analysis.ipynb` 末尾追加说明和代码：

```python
import nbformat

nb = nbformat.read('analysis.ipynb', as_version=4)

md = nbformat.v4.new_markdown_cell('## 未识别告警兜底\n\n对于 Excel 字典中不存在的索引，显示为 `未知告警(索引)`。')
nb.cells.append(md)

code = nbformat.v4.new_code_cell('''
# 将 match_alarm 中的 return None 改为兜底显示
# 请在 match_alarm 函数中手动替换，或运行上方的修改脚本
''')
nb.cells.append(code)

nbformat.write(nb, 'analysis.ipynb')

# 验证
nb2 = nbformat.read('analysis.ipynb', as_version=4)
print(f'完成。总 cell 数: {len(nb2.cells)}')
```

## 常见错误

| 错误做法 | 后果 |
|---------|------|
| 对 `.ipynb` 使用 SearchReplace | JSON 结构损坏，Notebook 无法解析 |
| DeleteFile + Write 重建 | 丢失 execution counts、outputs、kernel metadata |
| 手动编辑原始 JSON | 容易引入语法错误 |

## 推荐流程

1. `nbformat.read()` 读取
2. 修改 `nb.cells`（append / insert / pop / 编辑 source）
3. `nbformat.write()` 保存
4. 再次 `nbformat.read()` 验证
