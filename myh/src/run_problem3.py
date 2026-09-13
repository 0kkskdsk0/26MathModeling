"""问题三主运行（并行）：M0/M1/M2/M3 四策略费用对比 + result3.xlsx。

用法：
  python -m myh.src.run_problem3 --jobs 4                          # 默认
  python -m myh.src.run_problem3 --jobs 4 --no-floor --mpc-nscen 50 --tag _nofloor
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import myh.src.config as config
from myh.src.data_loader import get_data
from myh.src.rolling import RollingController
from myh.src.settlement import price_matrix, official_cost, verify
from myh.src.export import write_result


def _run_strategy(spec):
    """单策略完整运行（进程池工作函数，须模块级可 pickle）。"""
    name, update_set, seed, warm_soc, n_days, g_floor_q, mpc_nscen, tag = spec
    config.G_FLOOR_Q = g_floor_q
    config.MPC_N_SCENARIOS = mpc_nscen
    data = get_data()
    ctrl = RollingController(data, update_set, seed=seed)
    ctrl.run_year(n_days=n_days,
                  start_day=config.OUTPUT_START_IDX, start_soc=warm_soc)
    P = price_matrix(data)
    out = slice(config.OUTPUT_START_IDX, min(config.OUTPUT_END_IDX + 1, n_days))
    cost = official_cost(ctrl.GF, ctrl.G0, ctrl.E, P, out_slice=out)
    summ = {
        "update_set": update_set,
        "cost_normal": round(cost["normal"], 2),
        "cost_adjust": round(cost["adjust"], 2),
        "cost_emergency": round(cost["emergency"], 2),
        "cost_total": round(cost["total"], 2),
        "emergency_kwh": round(float(ctrl.E[out].sum()), 2),
    }
    if name == "M3":
        summ["verify"] = verify(ctrl, data, out_slice=out)
        if n_days == config.N_DAYS:
            write_result(config.TEMPLATE_DIR / "result3.xlsx",
                         config.OUTPUT_DIR / f"result3{tag}.xlsx", ctrl, data)
    return name, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--q", type=float, default=None, help="报童下界分位（默认用 config）")
    ap.add_argument("--no-floor", action="store_true", help="关闭报童下界")
    ap.add_argument("--mpc-nscen", type=int, default=None, help="执行层情景数")
    ap.add_argument("--tag", type=str, default=None, help="输出文件名后缀（默认按配置自动生成）")
    args = ap.parse_args()

    g_floor_q = None if args.no_floor else (args.q if args.q is not None else config.G_FLOOR_Q)
    mpc_nscen = args.mpc_nscen if args.mpc_nscen is not None else config.MPC_N_SCENARIOS
    config.G_FLOOR_Q = g_floor_q
    config.MPC_N_SCENARIOS = mpc_nscen
    tag = args.tag if args.tag is not None else config.output_tag()

    data = get_data()
    n_days = config.N_DAYS if args.days is None else args.days

    # 1. 公共暖启动（串行）：M3 从 1/1 6000 kWh 起，得到 2/1 公共初始 SOC
    print("== 公共暖启动：M3 从 1 月 1 日 6000 kWh 起 ==", flush=True)
    t0 = time.time()
    warm = RollingController(data, [0, 1, 2, 3], seed=args.seed)
    warm.run_year(n_days=config.OUTPUT_START_IDX + 1, start_day=0, start_soc=config.S0)
    warm_soc = float(warm.S[config.OUTPUT_START_IDX, 0])
    print(f"暖启动 SOC(2月1日0:00)={warm_soc:.1f} kWh，耗时 {time.time()-t0:.1f}s", flush=True)

    # 2. 四策略并行
    strategies = [("M0", [0]), ("M1", [0, 1]), ("M2", [0, 1, 2]), ("M3", [0, 1, 2, 3])]
    specs = [(name, us, args.seed, warm_soc, n_days, g_floor_q, mpc_nscen, tag)
             for name, us in strategies]
    jobs = args.jobs or min(4, len(specs))

    results = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for name, summ in ex.map(_run_strategy, specs):
            results[name] = summ
            print(f"[{name}] total={summ['cost_total']} 紧急={summ['cost_emergency']} "
                  f"紧急电量={summ['emergency_kwh']}kWh", flush=True)

    # 3. 增量价值
    C = {n: results[n]["cost_total"] for n in ["M0", "M1", "M2", "M3"]}
    results["deltas"] = {
        "dC6": round(C["M0"] - C["M1"], 2),
        "dC12": round(C["M1"] - C["M2"], 2),
        "dC18": round(C["M2"] - C["M3"], 2),
    }
    print(f"\n增量价值：ΔC6={results['deltas']['dC6']}  ΔC12={results['deltas']['dC12']}  "
          f"ΔC18={results['deltas']['dC18']}")
    if "verify" in results.get("M3", {}):
        v = results["M3"]["verify"]
        print(f"验算（M3）：max平衡残差={v['max_bal']:.2e}  max SOC残差={v['max_soc']:.2e}  "
              f"费用复算残差={v['cost_res']:.2e}  SOC违规={v['soc_viol']}  "
              f"同时充放电={v['simultaneous_cd']} 紧急购电充电={v['emergency_charge']}")

    with open(config.OUTPUT_DIR / f"summary_problem3{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n已写出 summary：{config.OUTPUT_DIR / f'summary_problem3{tag}.json'}")


if __name__ == "__main__":
    main()
