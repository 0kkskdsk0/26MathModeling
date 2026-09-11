"""用 nbformat 增量维护并执行第二问的理解对齐 notebook。"""
from pathlib import Path
import os
import sys
import tempfile
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / 'Q2_understanding.ipynb'


def run(nb):
    """顺序执行，保存真实输出并打印文本输出供本轮分析使用。"""
    nbf.write(nb, NOTEBOOK)
    with tempfile.TemporaryDirectory(prefix='q2_ipython_') as profile:
        env = dict(os.environ, IPYTHONDIR=profile)
        NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute(env=env)
    nbf.write(nb, NOTEBOOK)
    nbf.validate(nbf.read(NOTEBOOK, as_version=4))
    for cell in nb.cells[-3:]:
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text)
            elif output.output_type in ('display_data', 'execute_result'):
                print(output.get('data', {}).get('text/plain', ''))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1] == 'start':
        nb = nbf.v4.new_notebook(metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
        nb.cells = [
            nbf.v4.new_markdown_cell('# 第二问：形式化讨论前的理解对齐\n\n从 `Q2Project.md` 出发，检验三个与问题定义直接相关的候选认识：附件能否被当成0点已知量，储能能否吸收所有净负荷波动，以及历史是否具有可用结构。本轮只核对证据，不求解购电策略。\n\n原始来源为项目根目录的 `ProblemC/C题.md`、附件1、附件2及 result2 模板；项目约定见 `notation.md`。可复用逻辑位于 `utils/DataExplore/problem2/context_data.py`。全年统计是事后描述，不是日前可用的预测输入。'),
            nbf.v4.new_markdown_cell('## 分析1：输入和输出的时间与单位是否一致\n\n先检查矩阵、日期、缺失值及模板标签。因为功率必须乘以1/6小时才是时段电量，而时段错移会影响计费与指定时刻输出，不能只检查数据是否能读入。'),
            nbf.v4.new_code_cell('from pathlib import Path\nimport json\nimport numpy as np\nfrom utils.DataExplore.problem2.context_data import inspect_sources, load_inputs, audit_matrix, energy_summary, causal_baselines\nroot = Path.cwd().parent\nprint(json.dumps(inspect_sources(root), ensure_ascii=False, default=str, indent=2))\ntypical, actual = load_inputs(root)\nprint("附件1列名：", typical.columns.tolist())\nfor name, frame in actual.items():\n    print(name, json.dumps(audit_matrix(frame), ensure_ascii=False))'),
            nbf.v4.new_markdown_cell('结果解读将在查看实际执行输出后补充。')]
        run(nb)
    elif sys.argv[1] == 'energy':
        nb = nbf.read(NOTEBOOK, as_version=4)
        nb.cells[-1].source = '两个实际功率矩阵均为365×144，日期连续、无重复日期，缺失、非有限值及负值均为0。暂时没有插补或异常删除的依据。附件1有144条记录；负载峰值7978.8849 kW，光伏峰值10216.2 kW。\n\n输入标签从00:10到0:00+1；结果模板从0:10–0:20到0:00–0:10+1，与 `notation.md` 的0:00–0:10至23:50–24:00不一致。应在正式建模前确定输入点值代表哪个区间，并明确结果映射，不能只按列顺序匹配就认为物理时刻已对齐。下面仅做对整天列移位不敏感的电量统计。\n\n计划购电模板335行含1行表头，对应2月1日至12月31日334天。1月可用于形成初始历史，这是合理解释而非题面明说的训练规则；还需处理1月1日6000 kWh如何传递到2月1日。'
        nb.cells.extend([
            nbf.v4.new_markdown_cell('## 分析2：净负荷与储能的规模关系\n\n电池既受能量容量限制，也受充放电功率限制。将负载减去光伏得到净负荷，可以观察系统何时缺电、何时存在可转移的剩余能源。按每个记录代表10分钟区间功率的离散化假设统计电量，并核对典型日与全年均值关系，避免把全年汇总特征误当作历史预测。'),
            nbf.v4.new_code_cell('load = actual["小区负载"]\npv = actual["光伏发电实际功率"]\nassert load.index.equals(pv.index) and load.columns.equals(pv.columns)\nassert [str(x) for x in typical.iloc[:, 0]] == [str(x) for x in load.columns]\nstats, selected = energy_summary(load, pv)\nprint(json.dumps(stats, ensure_ascii=False, indent=2))\nprint(selected.round(2).to_string())\nprint("固定电价范围：", typical["电价"].min(), typical["电价"].max())\nprint("典型日负载与全年同刻均值最大差_kW：", np.max(np.abs(typical["小区负载"].to_numpy() - load.mean().to_numpy())))\nprint("典型日光伏与全年同刻均值最大差_kW：", np.max(np.abs(typical["光伏发电预测功率"].to_numpy() - pv.mean().to_numpy())))'),
            nbf.v4.new_markdown_cell('结果解读将在查看实际执行输出后补充。')])
        run(nb)
    elif sys.argv[1] == 'baselines':
        nb = nbf.read(NOTEBOOK, as_version=4)
        nb.cells[-1].source = '平均日负载111024.81 kWh，平均日光伏55482.86 kWh，平均日净用电55541.95 kWh。全年10130个时段光伏超过负载，占19.27%；241个时段的盈余功率超过5000 kW，最大6601.99 kW。在无售电或其他消纳通道、5000 kW限制按母线侧理解的条件下，这些时段仅靠电池无法吸收全部光伏盈余。\n\n电池可用储能跨度9600 kWh，按既定单向效率0.9，单次从上限放到下限可向母线释放8640 kWh，约为平均日净用电量的15.56%。这是单次放电容量比较，并不是电池全天累计吞吐上限。因此必须同时考虑功率、容量和跨时段安排。\n\n四个指定日的日净用电分别约62209、19335、62117、89038 kWh；冬至日无光伏盈余，夏至日有51个盈余时段。固定的是每天重复的价格曲线（0.3713–1.3952元/kWh），并不是全天单一常数电价。\n\n附件1负载与全年同刻均值最大差约0.0000504 kW，光伏约0.0431 kW。它们高度接近，但光伏不能严格声称只有四位小数舍入差。第二问只指定使用附件1的电价，不能未经说明把其负载、光伏作为无信息泄漏的历史预测基线。下面只使用过去日期构造预测。'
        nb.cells.extend([
            nbf.v4.new_markdown_cell('## 分析3：历史信息是否足以提供可检验的预测基线\n\n在2月1日至12月31日，用前一日、前一周、过去7日均值预测同一记录位置。所有预测只依赖当日前的数据。这一步只判断历史结构和剩余不确定性，不选择最终预测器。WAPE定义为绝对误差之和除以实际值绝对值之和；净负荷允许为负，因此分母不是净电量的代数和。再用“负载上周同刻＋光伏过去7日均值”检查相邻时段残差相关性，判断独立误差假设是否值得怀疑。'),
            nbf.v4.new_code_cell('baselines, error_info = causal_baselines(load, pv)\nprint(baselines.round(4).to_string(index=False))\nprint(json.dumps(error_info, ensure_ascii=False, indent=2))'),
            nbf.v4.new_markdown_cell('结果解读将在查看实际执行输出后补充。')])
        run(nb)
    elif sys.argv[1] == 'run':
        run(nbf.read(NOTEBOOK, as_version=4))
    elif sys.argv[1] == 'normalize':
        nb = nbf.read(NOTEBOOK, as_version=4)
        nb.cells[-2].source = nb.cells[-2].source.replace('assert [str(x) for x in typical.iloc[:, 0]] == [str(x) for x in load.columns]', 'from utils.DataExplore.problem2.context_data import label_minutes\nassert label_minutes(typical.iloc[:, 0]) == label_minutes(load.columns)\nassert label_minutes(load.columns) == list(range(10, 1441, 10))')
        run(nb)
    elif sys.argv[1] == 'finish':
        nb = nbf.read(NOTEBOOK, as_version=4)
        nb.cells[-1].source = '334天严格使用日前历史的核对中，负载上周同刻WAPE为3.8294%，明显低于前一日同刻的14.0795%；光伏过去7日均值为6.4485%，低于前一日的7.7914%。分别预测再相减得到净负荷WAPE为9.1089%，MAE为273.35 kW。历史具有可利用结构，预测误差仍须进入决策理解；误差量本身不等于紧急购电量，后者还取决于计划与储能控制。\n\n组合净负荷残差日内相邻相关系数为0.8497，提示偏差会连续出现。这个汇总相关性是探索证据，不是完成了独立同分布检验；它可能同时含有日级偏差与时段结构。后续若构建随机模型，应检验或保留时序相关性，不能未经验证逐时独立抽样。\n\n这些数字是全年回测后的模型比较，不能声称在2月1日已知道全年最优基线。若据此确定最终模型，应明确验证期与测试期，或滚动选择；本轮没有调参或生成正式策略。'
        nb.cells.append(nbf.v4.new_markdown_cell('## 小结：进入形式化讨论前的共同起点\n\n**当前工作方式的理解。** `Q2Project.md` 强调先识别问题、明确符号和假设，再形式化并探索算法；数据探索应服务于理解。本轮沿用 `notation.md` 的母线侧电量与单向效率0.9约定，不把队友 `myh/solution.md` 中的算法建议视为已决定方案。\n\n**题面确定的部分。** 每天0点制定当天计划；每日重复同一条电价曲线；缺电时以当时电价5倍紧急购电；常规部分按计划量计费；设备受容量、功率、效率约束；正式结果覆盖2—12月。物理缺口最终必须补齐，不能将失供电仅作为可付罚金的软约束。\n\n**有依据但仍属解释的部分。** 附件2是实际轨迹，应作为历史学习和逐时回放数据；它在文件中可见，不代表当天0点预知全天。第二问不引入附件3的预报。计划购电量固定而储能日内反馈，是合理候选信息结构，但需要在讨论中明确。当前时段测量是在控制前、过程中还是之后可用，也需说明。若允许在线响应，动作只能依赖已揭示历史与当前允许观测的信息。\n\n**形式化前需要决定的事项。**\n\n1. 信息与决策时点：哪些量0点承诺，哪些量逐时响应。两阶段场景模型若允许第二阶段看到完整全天轨迹再调度电池，会产生预知未来的放松；不能自动当成可执行反馈策略。\n2. 跨日状态：第二问没有重申日循环条件。合理候选是次日日初承接前日日末；题目给出的是1月1日6000 kWh。需要说明1月运行与2月1日状态来源，以及日末剩余电量如何计入规划，不能默认每日重置。\n3. 能量与结算：现有G、E、C、D是不同阶段的决策或实现量，S是状态，L、R是外生实现量，不宜全部列作已知参数。若计划购电视作足量交付，W需要容纳计划多买造成的未利用电量；若可少取但照付，应区分计划量与实际取电量。无售电、允许弃光或剩余弃置及充放电互斥都应明确。\n4. 目标口径：题面要求节省费用；期望费用、风险惩罚或最坏情况分别是不同建模选择，不能自动加入。紧急购电单价是5倍，若已单独按5倍计费，不能再为同一紧急电量重复加1倍基础费。\n5. 时间口径：输入点标签、数学区间和输出模板须统一；当前只完成格式规范化，没有裁定10分钟偏移。\n\n**下一步讨论顺序。** 先确定信息结构，再划分外生量、决策量与状态并写出可行性和结算关系，最后再考察与经典问题的对应。储能的时间耦合和反馈可实施性是对齐经典问题时必须保留的结构。\n\n**执行说明。** 使用本机已有的 Miniconda Python 与 nbformat/nbclient 执行（应用捆绑运行时缺少 notebook 依赖），原始Excel只读，未改动核心工作文档或统一符号。三个分析代码单元均保存真实执行结果；本轮未求解策略、未填写result2。'))
        nbf.write(nb, NOTEBOOK)
        checked = nbf.read(NOTEBOOK, as_version=4)
        nbf.validate(checked)
        cells = checked.cells
        for i, cell in enumerate(cells):
            if cell.cell_type == 'code':
                assert cell.execution_count is not None
                assert cells[i-1].cell_type == cells[i+1].cell_type == 'markdown'
                assert not any(o.output_type == 'error' for o in cell.outputs)
        assert not any('结果解读将在' in c.source for c in cells)
        print('Verified: 3 executed analysis cells, saved outputs, complete interpretations, valid notebook.')
