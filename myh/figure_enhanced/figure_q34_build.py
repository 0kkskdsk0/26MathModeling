"""Reproducible Q3/Q4 publication figures from saved results only."""
from pathlib import Path
import argparse, sys, json, hashlib, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--myh', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.myh.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/'figure'))
    import plot_results as base
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    from plot_style import export_figure, audit_design
    from PIL import Image, ImageOps, ImageDraw
    base.CONFIGS['C'] = 'nofloor_mpc5'
    font = base.setup_style()
    plt.rcParams.update({'font.size':11, 'axes.labelsize':11, 'xtick.labelsize':10,
                         'ytick.labelsize':10, 'legend.fontsize':10, 'axes.titlesize':12})
    s, w, report = base.load_data(root, out)
    blue,gold,green,red,ink,gray = base.BLUE,base.GOLD,base.GREEN,base.RED,base.INK,base.GRAY
    colors = [blue,green,gold]
    confs = ['A','B','C']
    descriptions = {'A':'报童下界 0.8；MPC 5 情景','B':'无报童下界；MPC 50 情景','C':'无报童下界；MPC 5 情景'}
    period = '2025.02.01—12.31 · 334 天实际回测输出'
    catalog = []
    report['layout_checks'] = {}
    report['user_style_override'] = '用户明确要求主标题、小标题和标注，覆盖技能的无标题默认规则；保留其他设计检查。'

    def canvas(title, subtitle, n=2, height=7.8):
        fig, axs = plt.subplots(n,1,figsize=(9,height),squeeze=False)
        fig.subplots_adjust(left=.13,right=.97,bottom=.14,top=.80,hspace=.65)
        fig.text(.07,.957,title,fontsize=19,fontweight='bold',va='top',color=ink)
        fig.text(.07,.902,subtitle,fontsize=10.5,color=gray,va='top')
        return fig,axs[:,0]

    def panel(ax,text):
        ax.set_title(text,loc='left',pad=12,fontweight='bold',fontsize=12)
        base.axes_grid(ax)

    def save(fig,name,title,note,source):
        fig.text(.07,.057,note,fontsize=9,color=gray,va='top',linespacing=1.6)
        from publication_layout import export_separate
        report['layout_checks'][name]=export_separate(fig,out,name,export_figure)
        catalog.append((name,title,note,source))
        plt.close(fig)

    # 1: cost composition, all 12 Q3 records.
    fig,axs=canvas('问题三｜不同更新策略的结算费用构成',period,3,10.8)
    fig.subplots_adjust(top=.82,hspace=.78)
    for ax,c in zip(axs,confs):
        bottom=np.zeros(4)
        for key,col,label in zip(base.COST_KEYS,[blue,gold,red],['常规购电费','计划调整费','紧急购电费']):
            vals=np.array([s[3,c][f'M{i}'][key] for i in range(4)])/1e4
            ax.bar(np.arange(4),vals,.55,bottom=bottom,color=col,edgecolor='white',label=label)
            bottom+=vals
        for i,v in enumerate(bottom): ax.text(i,v+28,f'{v:,.1f}',ha='center',fontsize=10)
        ax.set(ylim=(0,2180),ylabel='累计费用 / 万元',xticks=range(4),
               xticklabels=['M0\n仅 0:00','M1\n新增 6:00','M2\n再增 12:00','M3\n再增 18:00'])
        panel(ax,f'({chr(97+confs.index(c))}) 配置 {c}：{descriptions[c]}')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.56,.87),ncol=3)
    save(fig,'01_q3_cost','问题三：策略费用构成','柱顶为三项费用之和；全部柱从零起算，三个面板共用尺度。\n结算费用不包含优化目标中的终端影子价值。','6 份 JSON 中的 3 份 summary_problem3_*.json')

    # 2: sequential marginal values, recomputed instead of trusting rounded prose.
    fig,axs=canvas('问题三｜新增预报更新带来的费用节省',period+'；正值表示节省，负值表示增支',3,9.5)
    fig.subplots_adjust(left=.24,top=.81,hspace=.75)
    marginal=[]
    for ax,c in zip(axs,confs):
        vals=np.array([s[3,c][f'M{i}']['cost_total']-s[3,c][f'M{i+1}']['cost_total'] for i in range(3)])/1e4
        assert np.max(np.abs(vals*1e4-np.array([s[3,c]['deltas'][k] for k in ['dC6','dC12','dC18']])))<.03
        for i,v in enumerate(vals):
            ax.barh(i,v,.49,color=green if v>=0 else red)
            ax.annotate(f'{v:+.2f}',(v,i),xytext=(6 if v>=0 else -6,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center')
            marginal.append([c,[6,12,18][i],v])
        ax.axvline(0,color=ink,lw=.8)
        ax.set(xlim=(-140,100),ylim=(2.6,-.6),yticks=range(3),yticklabels=['6:00  M0→M1','12:00  M1→M2','18:00  M2→M3'],xlabel='更新前费用 − 更新后费用 / 万元')
        panel(ax,f'({chr(97+confs.index(c))}) 配置 {c}：{descriptions[c]}')
        base.axes_grid(ax,'x')
    base.write_csv(out/'data'/'marginal_values.csv',['configuration','new_update_hour','savings_10k_yuan'],marginal)
    save(fig,'02_q3_update_value','问题三：更新时刻的顺序边际收益','收益按 6:00→12:00→18:00 的指定顺序计算，不是独立消融或 Shapley 值。\n图示为本次回测差额，现有数据不足以给出统计显著性或置信区间。','summary_problem3_*.json；由相邻策略总费用作差')

    # 3: paired cost points plus information differences; small differences are explicit.
    fig,axs=canvas('问题四｜价格信息对结算费用的影响',period+'；C 为因果预测，PI 为提前已知当日电价',2,8.2)
    ax=axs[0]
    for j,c in enumerate(confs):
        for k in range(2):
            yy=j*3+k
            a=s[4,c][base.Q4_KEYS[k]]['cost_total']/1e4
            b=s[4,c][base.Q4_KEYS[k+2]]['cost_total']/1e4
            ax.plot([a,b],[yy,yy],color=colors[j],lw=2)
            ax.scatter(a,yy,marker='o',color=colors[j],s=55,zorder=3)
            ax.scatter(b,yy,marker='D',facecolors='white',edgecolors=colors[j],s=40,zorder=4)
            ax.annotate(f'{a:.1f} / {b:.1f}',(max(a,b),yy),xytext=(10,0),textcoords='offset points',va='center',fontsize=9.5)
    ax.set(xlim=(1460,2180),ylim=(8,-1),yticks=[0,1,3,4,6,7],yticklabels=[f'{c} · {k}' for c in confs for k in ['4-2','4-3']],xlabel='累计结算费用 / 万元')
    panel(ax,'(a) 因果价格与完美价格方案的配对费用')
    base.axes_grid(ax,'x')
    fig.legend(handles=[Line2D([],[],marker='o',color=gray,ls='',label='因果 C'),Line2D([],[],marker='D',mfc='white',color=gray,ls='',label='完美价格 PI')],loc='upper right',bbox_to_anchor=(.96,.88),ncol=2)
    ax=axs[1]
    for j,c in enumerate(confs):
        vals=[(s[4,c][base.Q4_KEYS[k]]['cost_total']-s[4,c][base.Q4_KEYS[k+2]]['cost_total'])/1e4 for k in range(2)]
        assert max(abs(vals[k]*1e4-s[4,c][f'info_value_4-{k+2}']) for k in range(2))<.03
        bars=ax.bar(np.arange(2)+(j-1)*.24,vals,.21,color=colors[j],label=f'配置 {c}')
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+(.5 if v>=0 else -.5),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=10)
    ax.set(ylim=(-3.3,21.5),xticks=[0,1],xticklabels=['4-2：C2 − PI2','4-3：C3 − PI3'],ylabel='费用节省 / 万元')
    ax.axhline(0,color=ink,lw=.8)
    panel(ax,'(b) 完美价格信息相对因果预测的费用节省')
    ax.legend(ncol=3,loc='upper left')
    save(fig,'03_q4_price_information','问题四：价格信息价值','上图标注顺序为 C / PI；下图保留真实负值，不将差额截断为零。\nA：下界 0.8、MPC5；B：无下界、MPC50；C：无下界、MPC5。','summary_problem4_*.json；有限回测费用差，不等同于理论非负 EVPI')

    # 4: controlled configuration comparisons with same C baseline.
    fig,axs=canvas('问题三｜报童下界与 MPC 情景数的配置对照','M3 全时刻更新策略；两项比较均以配置 C（无下界、MPC5）为基准',2,7.8)
    comparisons=[('A','启用报童下界\nC → A'),('B','增加 MPC 情景\nC → B')]
    rows=[]
    for ax,field,unit,letter in zip(axs,['cost_total','emergency_kwh'],['累计费用变化 / %','紧急购电量变化 / %'],['a','b']):
        baseline=s[3,'C']['M3'][field]
        for i,(c,label) in enumerate(comparisons):
            delta=s[3,c]['M3'][field]-baseline
            val=delta/baseline*100
            ax.plot([0,val],[i,i],color=colors[confs.index(c)],lw=3)
            ax.scatter(val,i,s=100,color=colors[confs.index(c)],zorder=3)
            ax.annotate(f'{val:+.1f}%  ({delta/1e4:+.1f} '+('万元)' if field=='cost_total' else '万 kWh)'),(val,i),xytext=(10 if val>=0 else -10,0),textcoords='offset points',va='center',ha='left' if val>=0 else 'right',fontsize=10)
            rows.append([c,field,baseline,s[3,c]['M3'][field],val])
        ax.axvline(0,color=gray,lw=.9,ls='--')
        ax.set(yticks=[0,1],yticklabels=[x[1] for x in comparisons],ylim=(1.6,-.6),xlabel=unit,xlim=(-135,65) if field=='emergency_kwh' else (-25,45))
        panel(ax,f'({letter}) '+('经济性：结算费用的变化' if field=='cost_total' else '运行表现：紧急购电依赖的变化'))
        base.axes_grid(ax,'x')
    fig.subplots_adjust(left=.24)
    base.write_csv(out/'data'/'configuration_comparison.csv',['configuration','metric','baseline_C','value','change_pct'],rows)
    save(fig,'04_q3_configuration_comparison','问题三：配置单因素对照','C→A：MPC 均为 5，仅切换报童下界；C→B：均无下界，MPC 从 5 增至 50。\n百分比均相对 C 计算；仅描述已保存回测，不外推最优参数或普遍因果结论。','summary_problem3_*.json 的 M3 记录')

    # 5–7: reuse established parsing and operation plots, adding full captions and panel titles.
    meta={
      'result_q3_q4_monthly_operation':('05_monthly_operation','月度运行｜购电费用与紧急购电需求',['(a) 月均日常规购电及调整费用','(b) 月均日紧急购电量'],'每月按实际输出天数取日均，包含无紧急购电的日期。\n上图不含紧急购电费，不能解释为每日总费用。'),
      'result_q3_q4_storage_operation':('06_storage_operation','储能运行｜日内充放电与隔夜储能',['(a) 六个四小时时段的平均充放电量','(b) 日末储能量的经验累计分布'],'正柱为充电、负柱为放电；同一四小时区间含两者不表示同时充放电。\n下图每条曲线包含 334 个日末观测；两条竖虚线为 1.2 / 10.8 MWh 安全边界。'),
      'result_q3_q4_purchase_adjustment':('07_purchase_adjustment','计划调整｜月均日内增购与减购分布',['(a) 问题三 M3：固定电价','(b) 问题四 C3：波动电价'],'调整量 = 最终生效计划 − 0:00 原计划；红色增购、蓝色减购。\n共享对称色标，完整覆盖数据；竖线标记 6:00、12:00、18:00 更新时刻。')}
    def decorated_save(fig,unused,name):
        new,title,titles,note=meta[name]
        fig.set_size_inches(9,8.5)
        for t in fig.findobj(plt.Text):
            if t.get_fontsize()<9: t.set_fontsize(9)
        data_axes=[ax for ax in fig.axes if ax.get_label()!='<colorbar>']
        for ax,ttl in zip(data_axes,titles): ax.set_title(ttl,loc='left',fontsize=12,fontweight='bold',pad=12)
        fig.subplots_adjust(top=.79,bottom=.16,hspace=.65)
        fig.text(.07,.963,title,fontsize=19,fontweight='bold',va='top')
        fig.text(.07,.909,period+'；配置 B：无报童下界、MPC50',fontsize=10.5,color=gray,va='top')
        for leg in fig.legends: leg.set_bbox_to_anchor((.55,.867))
        if name.endswith('purchase_adjustment'): fig.axes[-1].set_position([.89,.23,.025,.53])
        save(fig,new,title,note,'result3 / result4-2 / result4-3_nofloor_mpc50.xlsx')
    base.save=decorated_save
    base.monthly(w,out)
    base.storage(w,out)
    report.update(base.adjustment(w,out))

    # 8: real day-level distributions across all configurations, no invented uncertainty.
    fig,axs=canvas('紧急购电｜逐日分布与高需求尾部',period+'；每组包含全部 334 天，零购电日也计入',2,8.3)
    distrows=[]
    for ax,q,title in zip(axs,['3','4-3'],['(a) 问题三 M3：固定电价','(b) 问题四 C3：波动电价']):
        arrays=[w[q,c]['emergency']/1000 for c in confs]
        boxes=ax.boxplot(arrays,positions=[1,2,3],widths=.43,patch_artist=True,showfliers=True,whis=1.5,medianprops={'color':ink,'linewidth':1.6},flierprops={'marker':'.','markersize':4,'alpha':.5,'markeredgecolor':gray})
        for patch,col in zip(boxes['boxes'],colors): patch.set(facecolor=col,alpha=.65)
        for i,(c,vals) in enumerate(zip(confs,arrays),1):
            p95=float(np.quantile(vals,.95)); zeros=int(np.sum(vals<=1e-9))
            ax.scatter(i,p95,marker='D',s=32,color=red,zorder=4)
            ax.text(i,.95,f'零购电 {zeros} 天\nP95 = {p95:.2f}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9.5)
            distrows.append([q,c,334,zeros,float(np.mean(vals)),float(np.median(vals)),p95,float(np.max(vals))])
        ax.set(xticks=[1,2,3],xticklabels=['A：下界 0.8 / MPC5','B：无下界 / MPC50','C：无下界 / MPC5'],ylabel='每日紧急购电量 / MWh',ylim=(-.3,max(map(np.max,arrays))*1.4))
        panel(ax,title)
    fig.legend(handles=[Line2D([],[],color=red,marker='D',ls='',label='日购电量的 95% 分位数')],loc='upper right',bbox_to_anchor=(.96,.86))
    base.write_csv(out/'data'/'emergency_distribution.csv',['question','configuration','days','zero_days','mean_mwh','median_mwh','p95_mwh','max_mwh'],distrows)
    save(fig,'08_emergency_distribution','紧急购电：每日分布','箱体为第 25%—75% 分位数，中线为中位数，须线按 1.5 倍四分位距绘制。\n圆点保留须线外观测；红菱形为 P95 分位数，不是置信区间或供电中断概率。','6 份 result3 / result4-3 工作簿的紧急购电量工作表')

    for source in report['sources']:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    report.update({'font':font,'python':sys.version,'figures':len(catalog),'workbooks':len(w),
                   'scope':'6 JSON summaries + 9 workbooks; saved outputs only; no reoptimization',
                   'source_hashes_unchanged':True})
    (out/'data'/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Standalone contact sheet and gallery for convenient review.
    thumbs=[]
    for name,_,_,_ in catalog:
        im=Image.open(out/(name+'.png')).convert('RGB')
        assert im.width==3600
        thumbs.append(ImageOps.pad(im,(650,760),color='white'))
    sheet=Image.new('RGB',(1300,760*4),'#e3e7ea')
    for i,im in enumerate(thumbs): sheet.paste(im,((i%2)*650,(i//2)*760))
    sheet.save(out/'00_overview.jpg',quality=92)
    text=['# 问题三、问题四结果图（含标题增强版）','',
          '8 张 400 DPI PNG（3600 像素宽）及 SVG 矢量文件。延续原 figure 配色、宋体与 Times 数字、白底细网格。',
          '按用户要求加入主标题、子图编号、小标题、单位、图例和图下注释。未覆盖原图、原始输出、模型文档或求解代码。','',
          '统计期：2025-02-01 至 2025-12-31，共 334 天。A=下界0.8/MPC5；B=无下界/MPC50；C=无下界/MPC5。',
          '问题三 M0/M1/M2/M3 依次允许 0 点、再加 6 点、再加 12 点、再加 18 点更新。问题四 C2/C3 为因果价格方案，PI2/PI3 为相应完美价格对照。','',
          '## 数据核对','',
          '6 份 JSON、9 份 Excel 均通过费用合计、日期和维度、紧急电量、日首末储能递推、跨日连续性及边界核对。详见 data/validation.json。',
          '输入文件 SHA-256 在运行前后相同。所有图都通过缺字、字体最小值、文字出界和刻度重叠检查。',
          '技能默认无标题的规则由本次用户明确要求覆盖，其他数据与版面检查保留。','',
          '## 使用边界','',
          'analysis.md 的消融实验只有文字汇总、缺少独立原始输出，未用来出图。其“45.4/6.0 约4.6倍”的说法也与算术不符，故未沿用。',
          '不从日首末两点伪造十分钟 SOC；不从事件总电量推断逐时应急费用；不编造弃光率、统计显著性和置信区间。',
          '购电 Excel 模板的时段标题偏移十分钟；按 export.py 列位置及 notation.md，将首列解释为 00:00—00:10，原文件不修改。',
          'C→A / C→B 是已有配置的对照结果，不能据此证明所有参数情景下的普遍因果结论或 q 的最优值。','']
    for name,title,note,source in catalog:
        text.extend([f'## {title}','',f'![{title}]({name}.png)','',note.replace('\n',' '),'',f'数据来源：{source}。',''])
    text.extend(['## 复现','', '`python figure_q34_build.py --myh .. --out .`','',
                 '依赖现有 ../figure/plot_results.py 及其 utils/plot_style.py；使用 Python 3.12，matplotlib、numpy、openpyxl、Pillow。',
                 'data/ 保存逐日、逐月、费用、更新收益和分布指标 CSV，可直接核对。'])
    (out/'README.md').write_text('\n'.join(text),encoding='utf-8')
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>问题三与问题四 · 结果图</title><style>body{max-width:1100px;margin:40px auto;background:#f4f6f8;color:#334a5e;font-family:serif}article{background:white;padding:24px;margin:30px 0;border-radius:12px}img{width:100%;height:auto}p{line-height:1.8}a{color:#5b7fa3}</style><h1>问题三与问题四 · 结果图</h1><p>真实回测输出 · 334天 · 400 DPI · PNG / SVG</p>'
    for name,title,note,source in catalog:
        html+=f'<article><h2>{title}</h2><a href="{name}.png"><img src="{name}.png"></a><p>{note}</p><a href="vector/{name}.svg">SVG 矢量版</a></article>'
    (out/'gallery.html').write_text(html,encoding='utf-8')
    target=out/'figure_q34_build.py'
    if Path(__file__).resolve()!=target: shutil.copy2(__file__,target)
    for filename in ['publication_layout.py','compose_annotations.py']:
        origin=Path(__file__).parent/filename
        if origin.resolve()!=(out/filename).resolve(): shutil.copy2(origin,out/filename)
    with (out/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 本次版式修改\n\n坐标轴、刻度、轴名与色标均为黑色。上下子图间留出 1.55 英寸；子标题距主体 0.66 英寸。图例为绘图区外白底细框。\n\nbody/ 为不含后期标注的纯主体 PNG/SVG；annotations/ 为独立 JSON 和透明标注层；vector/ 的 SVG 包含 editable_annotations 可编辑文本层。\n\n后续只改 annotations/*.json 的文字或 x/y 坐标，再运行 `python compose_annotations.py`，即可重新排版，不必运行数据绘图。重新完整绘图会重建标注初始布局，修改后请保留 JSON 备份。\n')
    print('PASS: 8 figures, 9 workbooks, all data/layout checks; input hashes unchanged.')

if __name__=='__main__':
    main()
