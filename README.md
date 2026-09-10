# 26MathModeling
本项目用于我们团队合作求解26年国赛题目，并同时产出最终论文

# Project Architecture
.agents\skills 用于存放常用的AI skills，如让AI使用nbformat编辑notebook，让AI以notebook为工具分析数据，或将论文 Markdown 草稿转换为国赛规范 LaTeX 并修复编译问题  

当前已有项目 skills：
- `edit-notebook-with-nbformat`：使用 nbformat 安全编辑 Jupyter Notebook
- `analyze-data-using-notebook`：以 Notebook 为叙事工具开展数据分析
- `cumcm-paper-polisher`：默认仅做论文 Markdown/TeX 到国赛规范 LaTeX 的转换、排版和编译修复；写作修改意见与优秀论文参考为显式启用模式  
  
assets 文件夹存放队员产出的将用于论文叙事的资产，如图片，实验结果等，后续可视解题具体情况在文件夹下进一步创立目录  
  
paper 文件夹分节拆分了论文章节，使用md文档书写；每个章节专人专笔更改，最后将统一综合转化成latex文档，编译为pdf  

problem1 用于存放解决第一问的相关文件，负责求解的队员可进一步创建文件夹；当前已有：
- NewProject.md 以“识别问题-提出假设-建立模型-算法求解-模型验证”为设计思路的项目文档，希望开发者能以文档驱动开发的方式推进项目；“文档驱动开发”意味着开发者负责把握开发方向，文档负责记录研究路径，AI负责落实项目的工程细节。文档既作为开发者的思维工具，又为AI提供项目上下文，还为后续的论文编写提供原材料  
- task1.md 在指挥AI进行开发前需要编写的任务文档，后续将作为提示词交给AI；旨在引导开发者全面思考当前任务，写出全面的提示词，以防AI产出结果牛头不对马嘴。  

ProblemC 用于存放官方发放的原始题目，其中C题.md存放的是C题.pdf的markdown转化版本，若有不明晰的地方应该查阅C题.pdf这个官方原版文件做判别
