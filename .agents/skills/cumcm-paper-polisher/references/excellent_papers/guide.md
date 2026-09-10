# 优秀论文参考库使用说明

本文件夹是官方国赛优秀论文的本地参考库。只有用户明确要求参考优秀论文、对照优秀论文写法、提分建议，或要求增加/更新参考库时，才使用本文件夹。

纯 Markdown 转 LaTeX、格式排版或编译修复时，不要查阅本文件夹。

## 文件夹职责

- `index.json`：机器可读的官方论文索引，包含年份、题型、编号、标题和来源 URL。
- `cards/`：压缩后的写法卡片。普通写作审稿时优先读取这些卡片。
- `patterns/`：按主题或题型汇总的写法模式。
- `ocr_text/`：可选的本地 OCR 文本。只有需要精读对照时才使用。
- `raw/`：可选的下载页面图像或重构论文文件，用于本地私有分析。

## 刷新索引

在项目根目录运行：

```bash
python3 .agents/skills/cumcm-paper-polisher/scripts/fetch_excellent_papers.py --out .agents/skills/cumcm-paper-polisher/references/excellent_papers
```

如果还需要下载页面图片供本地 OCR 或查看：

```bash
python3 .agents/skills/cumcm-paper-polisher/scripts/fetch_excellent_papers.py --out .agents/skills/cumcm-paper-polisher/references/excellent_papers --download-pages
```

官方详情页通常提供页面图片，而不是直接 PDF。因此脚本会记录页面图片 URL，也可以下载到 `raw/<year>/<code>/pages/`。

## 添加用户提供的优秀论文

如果用户额外提供优秀论文，将 PDF 放到 `raw/user_added/`，OCR 后的文本放到 `ocr_text/user_added/`，并在 `cards/` 下创建写法卡片。卡片至少记录：

- 来源。
- 年份和题型，如果已知。
- 建模类型。
- 值得借鉴的写法动作。
- 图表使用方式。
- 验证方式。
- OCR 不确定性提示。

## 使用规则

优先使用卡片和汇总模式。只有用户要求详细对照，或卡片信息不足时，才读取 OCR 文本。面向用户回答时，不要长段引用来源论文。
