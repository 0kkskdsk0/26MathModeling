"""Apply separately edited annotations without rerunning data plots."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'svg.fonttype':'none','axes.unicode_minus':False,'figure.constrained_layout.use':False})
from publication_layout import compose
for path in sorted((root/'annotations').glob('*.json')):
    print(path.stem,compose(root/'body'/path.stem,path,root/path.stem))
