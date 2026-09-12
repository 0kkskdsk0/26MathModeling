"""全局配置与物理常数（对齐 problem3_solving_final.md）。

所有能量量纲统一为 kWh，功率统一为 kW；时间离散为一天 144 个十分钟区间。
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent      # myh/
REPO_DIR = BASE_DIR.parent                              # 仓库根目录
DATA_DIR = REPO_DIR / "ProblemC" / "附件"
TEMPLATE_DIR = DATA_DIR / "附件5"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACH1 = DATA_DIR / "附件1.xlsx"
ATTACH2 = DATA_DIR / "附件2.xlsx"
ATTACH3 = DATA_DIR / "附件3.xlsx"
ATTACH4 = DATA_DIR / "附件4.xlsx"

# 时间离散
T = 144
DT = 1.0 / 6.0
N_DAYS = 365

STAGE_HOURS = [0, 6, 12, 18]
STAGE_START = [0, 36, 72, 108]     # 0 基：0:00/6:00/12:00/18:00

OUTPUT_START_IDX = 31              # 2025-02-01（0 基）
OUTPUT_END_IDX = 364               # 2025-12-31（0 基，含）

# 储能物理（附录 1）
CAPACITY = 12000.0
S_MIN = 1200.0
S_MAX = 10800.0
S0 = 6000.0
P_MAX = 5000.0
E_BAR = P_MAX * DT                # ≈ 833.333
ETA_C = 0.9
ETA_D = 0.9

# 费用系数
EMERGENCY_MULT = 5.0
ADJUST_FRAC = 0.5

# 随机规划
N_SCENARIOS = 50
MPC_N_SCENARIOS = 5         # 执行层情景 MPC 的情景数（可小于规划层；5 约 1h 内跑完全套）
ALPHA_CVAR = 0.95
LAMBDA_RISK = 0.0
USE_CVAR = False

# 报童下界（方案 A）：计划购电量不得低于情景净负荷的该分位，对冲 wait-and-see 乐观偏差。
# 理论最优 0.8（应急电 5 倍价），None 关闭。
G_FLOOR_Q = 0.8

# 跨日近视（与第二问一致）：默认不含终端价值 θ，接受"储能的跨日影响有限"。
# 设 True 可做 θ 消融实验（终端价值用式 24-26 的确定性 24h 影子）。
USE_TERMINAL_VALUE = False

# 终端价值（式 24-26：17 网格 + 确定性 24h 影子 + 按周缓存）
TERM_GRID_STEP = 600.0
TERM_GRID = [S_MIN + i * TERM_GRID_STEP for i in range(17)]
TERM_END_SOC = 6000.0
