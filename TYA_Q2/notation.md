# Notation

## 1. 时间离散化

题目每间隔时长为 10 分钟，一天划分为 144 个时段。定义：

- 时段指标集：$\mathcal{T} = \{1, 2, \ldots, 144\}$，时段 $t$ 对应时间区间 $[(t-1)\Delta,\; t\Delta]$；
- 时段长度：$\Delta = \dfrac{1}{6}$ h（10 分钟）；
- $t = 1$ 对应 0:00–0:10，$t = 144$ 对应 23:50–24:00。


## 2. 参数
对于每个$t \in \mathcal{T}$,
- $L_t$：负载电量，kWh。$\ell_t$是负载功率，$ L_t=  \ell_t\cdot \Delta$；
- $R_t$：可用光伏电量，kWh；$R_t = P_t^{\mathrm{PV}} \cdot \Delta$,$P_t^{\mathrm{PV}}$是光伏发电功率。
- $G_t$：常规购电量，kWh；
- $E_t$：紧急购电量，kWh；
- $C_t,D_t$：交流**母线侧**充电量和放电量，kWh；
- $S_t$：时段t开始时储能电量，kWh；
- $W_t$：弃光或无法利用的剩余电量，kWh；
- $P_t$：时间t区间内的电价，元/kWh。

| 符号 | 含义| 取值 |
| -------------------- | ------------------------------------------------------------------ | -------------------- |
| $S^{\min}, S^{\max}$ | 储电量下/上限（防过充过放）                                                     | $1200,\ 10800$ kWh   |
| $\bar{e}$            | 单时段最大充/放电能量，$\bar{e} = 5000 \text{ kW} \times \Delta = 5000/6$ kWh | $\approx 833.33$ kWh |
| $\eta$               | 充放电效率      | $0.9$                |

## 3. 约束条件（由第一问继承而来，求解其它问时应按需修改）

- 能量平衡：(∀$t \in \mathcal{T}$)

$$
G_t+(E_t)+R_t+D_t=L_t+C_t+W_t.
$$

- SOC递推：

$$
S_{t+1}=S_t+\eta C_t-\frac{D_t}{\eta}.
$$

- 基础约束：

$$
1200\le S_t\le10800,
\qquad 0\le C_t,D_t\le833.333,
$$

- 非负性：
$$
G_t,E_t,W_t\ge0.
$$

主方案暂取 $\eta_c=\eta_d=0.9$。题目"充放电效率为90%"也可能被理解为往返效率90%，此时应取 $\eta_c=\eta_d=\sqrt{0.9}$；后者先按下不表，统一理解为 $\eta_c=\eta_d=0.9$。

## 4. 问题二扩展：日期、情景与两阶段记号

问题二需要对全年逐日建模并引入随机情景，在第 1—3 节基础上增加以下记号。

**统一记号规则**：带上标 $\omega$ 的量为情景设想值；不带 $\omega$ 的量为实际值（附件2实测数据或模型执行后的真实路径）；预测量若引入加 $\widehat{\cdot}$。计划购电量 $G_{d,t}$ 不带 $\omega$，以此体现它是对所有情景统一取值的第一阶段变量（非预期性）。

- 日期：$d\in\mathcal{D}$，$\mathcal{D}$ 为 2025 全年 365 天（数据域）；$\mathcal{D}_{\mathrm{out}}=\{2025\text{-}02\text{-}01,\ldots,12\text{-}31\}$ 为输出域，共 334 天；
- 历史池：$\mathcal{H}_d=\{j\in\mathcal{D}:j<d,\ j\text{ 与 }d\text{ 同季节}\}$，情景抽样的原料；
- 情景：$\omega\in\Omega_d\subseteq\mathcal{H}_d$，$\omega$ 本身就是历史日日期，以整天为单位抽样；$\pi_{d,\omega}\ge0$，$\sum_{\omega\in\Omega_d}\pi_{d,\omega}=1$；
- 情景数据：$L_{d,t}^{\omega},R_{d,t}^{\omega}$ 为情景 $\omega$ 对应历史日在时段 $t$ 的负载、光伏电量（kWh），数据出处即 $L_{d,t}^{\omega}=L_{\omega,t}$；净负载 $N_{d,t}^{\omega}=L_{d,t}^{\omega}-R_{d,t}^{\omega}$（kWh）；
- 第一阶段决策变量：$G_{d,t}\ge0$，计划购电量；
- 第二阶段变量（均带 $\omega$）：$C_{d,t}^{\omega},D_{d,t}^{\omega}\in[0,\bar e]$，$S_{d,t}^{\omega}\in[S^{\min},S^{\max}]$（$t\in\mathcal{T}^+$），$E_{d,t}^{\omega}\ge0$（紧急购电，单价 $5P_t$），$W_{d,t}^{\omega}\ge0$（弃置电量，弃光与已购未利用合并）；
- 实际执行路径：$C_{d,t},D_{d,t},S_{d,t},E_{d,t},W_{d,t}$（不带 $\omega$）；
- 跨日衔接：$S_{d+1,1}=S_{d,145}$，$S_{2025\text{-}01\text{-}01,1}=6000$；问题二不设 $S_{d,145}=S_{d,1}$ 循环约束；
- 终端价值：$V_{d+1}(s)$ 为次日以储能 $s$ 开局的最小期望费用；$\theta_d^{\omega}\ge a_{d,q}S_{d,145}^{\omega}+b_{d,q}$（$q=0,\ldots,Q-1$，$a_{d,q}\le0$ 且随 $q$ 单调不减）为其分段线性下界支撑。**近视模型取 $\theta\equiv0$；远视模型把 $\sum_{\omega\in\Omega_d}\pi_{d,\omega}\theta_d^{\omega}$ 计入目标函数**，两个模型在其余部分逐元素相同，详见 `outputs/taskCompositeFar/frozen_rule_far.json` 与 `Q2Project.md` 第 4.4 节；
- 全年结算：$C_{\mathrm{official}}=\sum_{d\in\mathcal{D}_{\mathrm{out}}}\sum_{t\in\mathcal{T}}(P_tG_{d,t}+5P_tE_{d,t})$，只含真实发生的计划费与紧急费，$\theta$ 不计入。

