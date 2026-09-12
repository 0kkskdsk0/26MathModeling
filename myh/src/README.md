# 微网电力调控 · 问题三 / 问题四求解代码

本目录实现 C 题「微网与外部电网电力调控策略」的问题三、问题四。建模文档见
`../doc/problem3_solving_final.md` 与 `../doc/problem4_final.md`；数据见仓库根目录
`ProblemC/附件/`。

## 一、环境依赖

- Python ≥ 3.10
- 依赖库：`numpy`、`scipy`（内置 HiGHS 求解器）、`pandas`、`openpyxl`

```bash
pip install numpy scipy pandas openpyxl
```

无需额外安装 LP 求解器，代码通过 `scipy.optimize.linprog / milp` 调用 HiGHS。

## 二、目录结构与模块说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 全局常数与参数（路径、时间、储能物理、随机规划） |
| `data_loader.py` | 读取附件 1–4，统一成 numpy 数组；含季节划分 `season` |
| `forecast.py` | 负载/光伏因果预测 + 电价三尺度预测 |
| `scenarios.py` | 联合残差情景生成（问题三二元 / 问题四三元），同季节条件化 |
| `lp.py` | 轻量 LP/MILP 建模封装（scipy HiGHS） |
| `dispatch.py` | 规划层两阶段随机 LP + 执行层情景 MPC（`solve_planning`/`solve_mpc`） |
| `terminal_value.py` | 跨日末端价值（确定性 24h 影子，仅 `USE_TERMINAL_VALUE=True` 时用） |
| `rolling.py` | 滚动控制器（四阶段滚动 + 情景 MPC + M0/M1/M2/M3） |
| `settlement.py` | 官方结算与验算残差 |
| `export.py` | 按附件 5 模板写 result*.xlsx |
| `run_problem3.py` | 问题三入口：M0/M1/M2/M3 四策略费用对比（并行） |
| `run_problem4.py` | 问题四入口：C2/C3/PI2/PI3 四配置（并行） |

## 三、运行方式（务必在仓库根目录执行）

代码是包结构（`myh.src.*`），**必须在仓库根目录**（`26MathModeling/`）下用 `-m` 运行：

```bash
cd d:\myh\大学\2026数模国赛\26MathModeling

# 问题三：M0/M1/M2/M3 四策略全年费用对比，写 result3.xlsx
python -m myh.src.run_problem3 --jobs 4

# 问题四：波动电价重算问题二/三，写 result4-2.xlsx、result4-3.xlsx
python -m myh.src.run_problem4 --jobs 4
```

> 若不想用 `-m`，也可先设置 `PYTHONPATH=.` 再直接跑脚本：
> ```bash
> PYTHONPATH=. python myh/src/run_problem3.py --jobs 4
> ```

### 常用参数

| 参数 | 含义 | 示例 |
| --- | --- | --- |
| `--jobs N` | 并行进程数（默认 4） | `--jobs 2` |
| `--seed N` | 随机种子（默认 0） | `--seed 1` |
| `--days N` | 只跑前 N 天（调试，很快） | `--days 33` |
| `--q 0.8` | 报童下界分位（问题三；None 关闭） | `--q None` |
| `--skip-pi` | 问题四跳过 PI-DA 信息基准 | `--skip-pi` |

### 快速验证（先小样本跑通再全年）

```bash
# 问题三小样本（33 天，几十秒）
python -m myh.src.run_problem3 --days 33 --jobs 2

# 问题四小样本
python -m myh.src.run_problem4 --days 33 --jobs 2 --skip-pi
```

## 四、问题三详细说明

跑 `M0/M1/M2/M3` 四种更新策略，比全年官方总费用：

- `M0`：只 0:00 定一次
- `M1`：0:00 + 6:00
- `M2`：0:00 + 6:00 + 12:00
- `M3`：0:00 + 6:00 + 12:00 + 18:00（全开）

公共初始储能量由 M3 从 1 月 1 日 6000 kWh 暖启动得到；四策略从该共同状态跑 2/1–12/31。
输出 `summary_problem3.json`（各策略费用分解 + ΔC6/ΔC12/ΔC18 + 验算）与 `result3.xlsx`（M3 逐日明细）。

## 五、问题四详细说明

跑四组配置（波动电价 = 附件 4）：

- `C2`（4-2 因果）：只 0:00 → `result4-2.xlsx`
- `C3`（4-3 因果）：0/6/12/18 → `result4-3.xlsx`
- `PI2`、`PI3`（完美价格信息基准，只进 summary，不写文件）

输出 `summary_problem4.json`（四配置费用 + 信息价值 `info_value_4-2/4-3`）。

## 六、关键配置参数（`config.py`）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `N_SCENARIOS` | 50 | 规划层联合情景数 |
| `MPC_N_SCENARIOS` | 5 | 执行层情景 MPC 情景数（越大越稳、越慢） |
| `G_FLOOR_Q` | 0.8 | 报童下界分位（`None` 关闭） |
| `USE_TERMINAL_VALUE` | False | 是否含终端价值 θ（False=跨日近视，与问题二一致） |
| `USE_CVAR` / `LAMBDA_RISK` | False / 0.0 | 是否加 CVaR 风险项 |
| `ETA_C` / `ETA_D` | 0.9 / 0.9 | 充放电效率（往返 90% 口径取 `sqrt(0.9)`） |

**重要**：`USE_TERMINAL_VALUE=False` 是「跨日近视」口径（与第二问一致），此时储能每晚放空到下限、应急电会偏高。若要含 θ，改 `USE_TERMINAL_VALUE=True`。

## 七、输出文件（写在 `myh/output/`）

| 文件 | 内容 |
| --- | --- |
| `result3.xlsx` | 问题三 M3：计划/调整购电量、充放电、紧急购电（4 工作表，334 天） |
| `result4-2.xlsx` | 问题四-2：计划购电量、充放电、紧急购电（3 工作表） |
| `result4-3.xlsx` | 问题四-3：计划/调整购电量、充放电、紧急购电（4 工作表） |
| `summary_problem3.json` | 问题三四策略费用 + 增量价值 + 验算 |
| `summary_problem4.json` | 问题四四配置费用 + 信息价值 |

## 八、预计耗时

- 问题三（4 策略，4 进程并行，`MPC_N_SCENARIOS=5`）：约 **50–60 分钟**。
- 问题四（4 配置，4 进程并行）：约 **1–2 小时**。
- 小样本 `--days 33`：几十秒。
