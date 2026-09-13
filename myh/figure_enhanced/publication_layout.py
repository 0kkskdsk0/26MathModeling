"""Separate body rendering and externally editable annotation composition."""
from pathlib import Path
import json, copy, xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox
from PIL import Image

def compose(body_stem, spec_path, final_stem):
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8'))
    width,height=spec['size_inches']
    overlay=plt.figure(figsize=(width,height),dpi=120,facecolor='none')
    texts=[]
    for item in spec['annotations']:
        texts.append(overlay.text(item['x'],item['y'],item['text'],
            fontsize=item['fontsize'],color=item['color'],fontweight=item['weight'],
            ha=item['ha'],va=item['va'],rotation=item['rotation'],linespacing=1.5))
    overlay.canvas.draw()
    renderer=overlay.canvas.get_renderer()
    boxes=[t.get_window_extent(renderer) for t in texts]
    for i,b in enumerate(boxes):
        assert b.x0>=0 and b.y0>=0 and b.x1<=overlay.bbox.width and b.y1<=overlay.bbox.height, ('annotation outside canvas',spec['annotations'][i])
        for j in range(i):
            assert not b.overlaps(boxes[j]), ('annotation overlap',spec['annotations'][i]['text'],spec['annotations'][j]['text'])
    for i,b in enumerate(boxes):
        for rect in spec['protected_rectangles']:
            r=Bbox.from_extents(rect[0]*overlay.bbox.width,rect[1]*overlay.bbox.height,rect[2]*overlay.bbox.width,rect[3]*overlay.bbox.height)
            assert not b.overlaps(r), ('annotation covers body or legend',spec['annotations'][i]['text'])
    layer=Path(spec_path).with_suffix('.png')
    overlay.savefig(layer,dpi=400,transparent=True)
    overlay_svg=Path(spec_path).with_suffix('.svg')
    overlay.savefig(overlay_svg,transparent=True)
    body=Image.open(str(body_stem)+'.png').convert('RGBA')
    body=Image.alpha_composite(body,Image.open(layer).convert('RGBA'))
    body.convert('RGB').save(str(final_stem)+'.png',dpi=(400,400))
    qa=Path(final_stem).parent/'_qa'
    qa.mkdir(exist_ok=True)
    body.convert('L').save(qa/(Path(final_stem).name+'_grayscale.png'),dpi=(400,400))
    ns='http://www.w3.org/2000/svg'
    ET.register_namespace('',ns)
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    tree=ET.parse(str(body_stem)+'.svg')
    root=tree.getroot()
    # Prefix IDs and their references in the annotation document.
    overlay_text=overlay_svg.read_text(encoding='utf-8')
    layer_root=ET.fromstring(overlay_text)
    for el in layer_root.iter():
        if 'id' in el.attrib: el.set('id','annotation_'+el.attrib['id'])
        for key,value in list(el.attrib.items()):
            if value.startswith('#'): el.set(key,'#annotation_'+value[1:]) if key.endswith('href') else None
            if 'url(#' in value: el.set(key,value.replace('url(#','url(#annotation_'))
    group=ET.SubElement(root,f'{{{ns}}}g',{'id':'editable_annotations'})
    for child in layer_root:
        if child.tag.endswith('defs'): root.append(copy.deepcopy(child))
        elif child.tag.endswith('g'): group.append(copy.deepcopy(child))
    vector=Path(final_stem).parent/'vector'
    vector.mkdir(exist_ok=True)
    tree.write(vector/(Path(final_stem).name+'.svg'),encoding='utf-8',xml_declaration=True)
    plt.close(overlay)
    return {'annotation_count':len(texts),'overlap_check':'PASS','body_protection_check':'PASS'}

def export_separate(fig,out,name,export_figure):
    """Reserve physical whitespace, export clean body, then compose external layer."""
    axes=[a for a in fig.axes if a.get_label()!='<colorbar>' and not hasattr(a,'_colorbar')]
    n=len(axes)
    style=json.loads((Path(__file__).parent/'paper_style.json').read_text(encoding='utf-8'))
    h=11.4 if n==3 else 8.7
    fig.set_size_inches(9,h)
    fig.set_layout_engine(None)
    # Each body retains adequate physical height; 1.55 inch gaps hold labels and titles.
    top=1.4
    bottom=.65
    gap=1.35
    body_h=(h-top-bottom-(n-1)*gap)/n
    for i,ax in enumerate(axes):
        old=ax.get_position()
        ytop=h-top-i*(body_h+gap)
        ax.set_position([old.x0,(ytop-body_h)/h,old.width,body_h/h])
        ax.tick_params(axis='both',which='both',colors='black',labelcolor='black')
        ax.xaxis.label.set_color('black'); ax.yaxis.label.set_color('black')
        ax.xaxis.get_offset_text().set_color('black');ax.yaxis.get_offset_text().set_color('black')
        for spine in ax.spines.values(): spine.set_color('black');spine.set_linewidth(.9)
    for ax in fig.axes:
        if ax in axes: continue
        ax.set_position([.89,.19,.024,.57])
        ax.tick_params(colors='black',labelcolor='black')
        ax.yaxis.label.set_color('black')
        for spine in ax.spines.values(): spine.set_edgecolor('black')

    # Consistent boxed legends sit entirely outside every data region.
    legends=[]
    for leg in list(fig.legends):
        handles=leg.legend_handles
        labels=[t.get_text() for t in leg.get_texts()]
        leg.remove()
        legends.append(fig.legend(handles,labels,loc='lower right',bbox_to_anchor=(.96,axes[0].get_position().y1+.08/h),ncol=min(3,len(labels)),frameon=True,facecolor='white',edgecolor='#555555',framealpha=1,borderpad=.55))
    for ax in axes:
        old=ax.get_legend()
        if old is None: continue
        handles=old.legend_handles; labels=[t.get_text() for t in old.get_texts()]
        old.remove()
        # Store axes legend as a figure legend in the dedicated panel header band.
        pos=ax.get_position()
        legends.append(fig.legend(handles,labels,loc='lower right',bbox_to_anchor=(pos.x1,pos.y1+.08/h),ncol=min(3,len(labels)),frameon=True,facecolor='white',edgecolor='#555555',framealpha=1,borderpad=.55))
    for leg in legends:
        leg.get_frame().set_linewidth(.65)
        for t in leg.get_texts(): t.set_color('black')

    # Extract annotation content and absolute layout from plotting objects.
    # The body export below contains none of these objects.
    candidates=[]
    for i,t in enumerate(list(fig.texts)):
        content=t.get_text()
        if not content: continue
        if i!=0:
            t.remove()
            continue
        t.set_text(style['titles'][name])
        t.set_position((.5,1-.25/h))
        t.set_horizontalalignment('center')
        t.set_fontsize(13)
        t.set_fontweight('normal')
        t.set_color('black')
        candidates.append((t,'outside'))
    titles=[]
    for panel_index,ax in enumerate(axes):
        pos=ax.get_position()
        title=ax.get_title(loc='left')
        if name in style['panels']: title=style['panels'][name][panel_index]
        if title:
            # 0.65 inch title-to-body spacing leaves room for a framed legend.
            t=fig.text(pos.x0,pos.y1+.57/h,title,ha='left',va='bottom',fontsize=11,fontweight='normal',color='black')
            candidates.append((t,'outside'));titles.append(ax)
        for t in list(ax.texts):
            if '零购电' in t.get_text():
                t.remove()
            else: candidates.append((t,'data_label'))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    specs=[]
    for t,kind in candidates:
        t.set_color('black')
        t.set_fontweight('normal')
        point=t.get_transform().transform(t.get_position())
        if kind=='data_label' and t.get_verticalalignment()=='baseline':
            t.set_verticalalignment('bottom')
            point[1]+=5
        specs.append({'text':t.get_text(),'x':float(point[0]/fig.bbox.width),'y':float(point[1]/fig.bbox.height),
           'fontsize':float(t.get_fontsize()),'color':t.get_color(),'weight':t.get_fontweight(),
           'ha':t.get_horizontalalignment(),'va':t.get_verticalalignment(),'rotation':float(t.get_rotation()),'kind':kind})
    # Numerical data labels must avoid data marks; protected mark boxes are exported.
    protected=[]
    for ax in axes:
        for patch in ax.patches:
            b=patch.get_window_extent(renderer)
            if b.width>0 and b.height>0: protected.append([b.x0/fig.bbox.width,b.y0/fig.bbox.height,b.x1/fig.bbox.width,b.y1/fig.bbox.height])
        for coll in ax.collections:
            offsets=coll.get_offsets()
            if not len(offsets): continue
            points=coll.get_offset_transform().transform(offsets)
            for x,y in points:
                protected.append([(x-5)/fig.bbox.width,(y-5)/fig.bbox.height,(x+5)/fig.bbox.width,(y+5)/fig.bbox.height])
        for im in ax.images:
            b=ax.get_window_extent(renderer)
            protected.append([b.x0/fig.bbox.width,b.y0/fig.bbox.height,b.x1/fig.bbox.width,b.y1/fig.bbox.height])
    for leg in legends:
        b=leg.get_window_extent(renderer)
        protected.append([b.x0/fig.bbox.width,b.y0/fig.bbox.height,b.x1/fig.bbox.width,b.y1/fig.bbox.height])
    for t,_ in candidates: t.remove()
    for ax in titles: ax.set_title('',loc='left')
    layers=out/'annotations';layers.mkdir(exist_ok=True)
    body=out/'body';body.mkdir(exist_ok=True)
    spec_path=layers/(name+'.json')
    # Existing independent edits are preserved; fresh layout seeds are separate.
    specification={'size_inches':[9,h],'annotations':specs,'protected_rectangles':protected,
       'instructions':'独立后期标注层。x/y 为从画布左下角起算的归一化坐标；修改后单独运行 compose_annotations.py。'}
    spec_path.write_text(json.dumps(specification,ensure_ascii=False,indent=2),encoding='utf-8')
    export_figure(fig,body/name,dpi=400,svg=True,strict_layout=True,strict_design=False)
    result=compose(body/name,spec_path,out/name)
    return result
