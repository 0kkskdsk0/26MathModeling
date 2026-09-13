"""问题四主运行：波动电价下重算问题二（result4-2）与问题三（result4-3），四组并行。

C2/C3 为因果口径主结果（写 result4-2/4-3），PI2/PI3 为完美价格信息基准（只进 summary）。

用法：
  python -m myh.src.run_problem4 --jobs 4                          # 默认
  python -m myh.src.run_problem4 --jobs 4 --no-floor --mpc-nscen 50 --tag _nofloor
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import myh.src.config as config
from myh.src.data_loader import get_data
from myh.src.rolling import RollingController
from myh.src.settlement import price_matrix, official_cost, verify
from myh.src.export import write_result


def _run_config(spec):
    """单配置完整运行（进程池工作函数）。"""
    name, update_set, price_info, has_adjust, outfile, seed, days, g_floor_q, mpc_nscen, tag = spec
    config.G_FLOOR_Q = g_floor_q
    config.MPC_N_SCENARIOS = mpc_nscen
    data = get_data()
    ctrl = RollingController(data, update_set, seed=seed,
                             price_type="varying", price_info=price_info)
    ctrl.run_year(n_days=days, start_day=0, start_soc=config.S0)
    P = price_matrix(data, "varying")
    out = slice(config.OUTPUT_START_IDX, min(config.OUTPUT_END_IDX + 1, days))
    cost = official_cost(ctrl.GF, ctrl.G0, ctrl.E, P, out_slice=out)
    summ = {
        "update_set": update_set,
        "price_info": price_info,
        "cost_normal": round(cost["normal"], 2),
        "cost_adjust": round(cost["adjust"], 2),
        "cost_emergency": round(cost["emergency"], 2),
        "cost_total": round(cost["total"], 2),
        "emergency_kwh": round(float(ctrl.E[out].sum()), 2),
    }
    if name == "4-3_causal_C3":
        summ["verify"] = verify(ctrl, data, out_slice=out, price_type="varying")
    if outfile is not None and days == config.N_DAYS:
        stem = outfile[:-5] if outfile.endswith(".xlsx") else outfile
        write_result(config.TEMPLATE_DIR / outfile, config.OUTPUT_DIR / f"{stem}{tag}.xlsx",
                     ctrl, data, price_type="varying", has_adjust=has_adjust)
    return name, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-pi", action="store_true", help="跳过 PI-DA 信息基准")
    ap.add_argument("--days", type=int, default=None, help="只跑前 N 天（调试）")
    ap.add_argument("--jobs", type=int, default=None, help="并行进程数")
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

    days = config.N_DAYS if args.days is None else args.days
    configs = [
        ("4-2_causal_C2", [0], "causal", False, "result4-2.xlsx"),
        ("4-3_causal_C3", [0, 1, 2, 3], "causal", True, "result4-3.xlsx"),
    ]
    if not args.skip_pi:
        configs += [
            ("4-2_perfect_PI2", [0], "perfect", False, None),
            ("4-3_perfect_PI3", [0, 1, 2, 3], "perfect", True, None),
        ]
    specs = [(n, us, info, has_adj, out, args.seed, days, g_floor_q, mpc_nscen, tag)
             for n, us, info, has_adj, out in configs]

    jobs = args.jobs or min(4, len(specs))
    results = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for name, summ in ex.map(_run_config, specs):
            results[name] = summ
            print(f"[{name}] total={summ['cost_total']} 紧急={summ['cost_emergency']} "
                  f"紧急电量={summ['emergency_kwh']}kWh", flush=True)

    if "4-2_causal_C2" in results and "4-2_perfect_PI2" in results:
        results["info_value_4-2"] = round(
            results["4-2_causal_C2"]["cost_total"] - results["4-2_perfect_PI2"]["cost_total"], 2)
    if "4-3_causal_C3" in results and "4-3_perfect_PI3" in results:
        results["info_value_4-3"] = round(
            results["4-3_causal_C3"]["cost_total"] - results["4-3_perfect_PI3"]["cost_total"], 2)

    with open(config.OUTPUT_DIR / f"summary_problem4{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n已写出 summary：{config.OUTPUT_DIR / f'summary_problem4{tag}.json'}")


if __name__ == "__main__":
    main()
