# -*- coding: utf-8 -*-
"""问题二远视模型（taskCompositeFar）。

模块清单：

- `assumption_audit.py`：跨日近视假设的否证实验与不可检验性论证（只读既有产物）；
- `value_function.py`：终端价值函数 V_{d+1}(s) 的因果采样、次梯度提取、切线生成、
  网格自适应加密与收敛判定；
- `model_far.py`：含终端价值项的稀疏 LP 装配（SciPy + HiGHS），保留 theta ≡ 0 的退化开关；
- `solve_year_far.py`：全年滚动求解主程序，含配对对照、导出、图表、退化核验与扰动检验；
- `validate_far.py`：复用 taskComposite 的 12 条断言并新增 C13—C21；
- `common.py`：共享上下文（路径、数据加载、情景池与核权重的口径复用）。

口径说明见 `outputs/taskCompositeFar/frozen_rule_far.json` 与
`outputs/taskCompositeFar/assumption_audit.md`。
"""
