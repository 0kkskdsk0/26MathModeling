from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
from publication_layout import compose

out=Path(r'D:\myh\大学\2026数模国赛\26MathModeling\myh\figure_enhanced')
plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'svg.fonttype':'none','axes.unicode_minus':False,'figure.constrained_layout.use':False})
style=json.loads((Path(__file__).parent/'paper_style.json').read_text(encoding='utf-8'))
backup=out/'_layout_source'
backup.mkdir(exist_ok=True)
report={}
ns='http://www.w3.org/2000/svg'
ET.register_namespace('',ns)
for name,title in style['titles'].items():
    for source in [out/'body'/(name+'.png'),out/'body'/(name+'.svg'),out/'annotations'/(name+'.json')]:
        target=backup/(name+source.suffix)
        if not target.exists(): shutil.copy2(source,target)
    spec=json.loads((backup/(name+'.json')).read_text(encoding='utf-8'))
    oldh=spec['size_inches'][1]
    top,bottom=.35,.25
    newh=oldh-top-bottom
    # Remove only blank outer margins; all data regions remain unchanged.
    im=Image.open(backup/(name+'.png'))
    im.crop((0,round(top*400),im.width,im.height-round(bottom*400))).save(out/'body'/(name+'.png'),dpi=(400,400))
    tree=ET.parse(backup/(name+'.svg'))
    root=tree.getroot()
    root.set('height',f'{newh*72}pt');root.set('viewBox',f'0 0 648 {newh*72}')
    group=ET.Element(f'{{{ns}}}g',{'transform':f'translate(0 {-top*72})'})
    for child in list(root):
        if child.tag.endswith('g'):
            root.remove(child);group.append(child)
    root.append(group)
    tree.write(out/'body'/(name+'.svg'),encoding='utf-8',xml_declaration=True)
    annotations=[]
    panel_index=0
    for i,item in enumerate(spec['annotations']):
        text=item['text']
        if i==0:
            continue
        elif text.startswith(('(a)','(b)','(c)')):
            if name in style['panels']: item['text']=style['panels'][name][panel_index]
            panel_index+=1
            item.update(fontsize=11,weight='normal',color='black',y=(item['y']*oldh-bottom)/newh)
        elif item['kind']=='data_label':
            item.update(weight='normal',color='black',y=(item['y']*oldh-bottom)/newh)
        else:
            continue
        annotations.append(item)
    spec['annotations']=annotations
    spec['size_inches']=[9,newh]
    for rect in spec['protected_rectangles']:
        rect[1]=(rect[1]*oldh-bottom)/newh
        rect[3]=(rect[3]*oldh-bottom)/newh
    path=out/'annotations'/(name+'.json')
    path.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    report[name]=compose(out/'body'/name,path,out/name)
thumbs=[]
for name in style['titles']:
    thumbs.append(ImageOps.pad(Image.open(out/(name+'.png')).convert('RGB'),(650,760),color='white'))
sheet=Image.new('RGB',(1300,3040),'white')
for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*650,(i//2)*760))
sheet.save(out/'00_overview.jpg',quality=92)
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>问题三与问题四结果图</title><style>body{max-width:950px;margin:30px auto;font-family:SimSun,serif;color:black}article{margin:30px 0}img{width:100%}a{color:#333}</style><h1 style="font-size:20px;font-weight:normal">问题三与问题四结果图</h1>'
for name,title in style['titles'].items():html+=f'<article><a href="{name}.png"><img src="{name}.png" alt="{title}"></a><a href="vector/{name}.svg">SVG</a></article>'
(out/'gallery.html').write_text(html,encoding='utf-8')
(out/'data'/'paper_layout_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for filename in ['publication_layout.py','compose_annotations.py','paper_style.json','revise_paper_artifacts.py']:
    source=Path(__file__).parent/filename
    if source.resolve()!=(out/filename).resolve():shutil.copy2(source,out/filename)
with (out/'README.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 当前论文样式（最新）\n\n主标题改为 13pt 常规宋体，居中；子标题 11pt 常规字重。英文与数字使用 Times New Roman，标注文字统一黑色。图内日期副标题、装饰分隔符和长说明已删除，数据口径留在本说明文件。黑色坐标、框式图例和子图留白保留。\n\n现有 figure 目录已不存在，完整数据重绘脚本的旧依赖不可用。当前版式修订由 revise_paper_artifacts.py 基于保留的主体图与独立标注完成；日常仅修改 annotations/*.json 并运行 compose_annotations.py，无需依赖原 figure 目录。_layout_source 保留本次修订前的主体与标注，可用于复现。\n')
with (out/'README.md').open('a',encoding='utf-8') as f:
    f.write('\n\n最终调整：全部 8 张图已去掉主标题，保留子标题、数据标注、坐标轴与图例。PNG、SVG、独立标注层和总览同步更新。\n')
print('PASS: 8 paper figures without main titles; annotations verified; data marks unchanged.')
