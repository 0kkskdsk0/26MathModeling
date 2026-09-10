# 国赛分章节 LaTeX 写法指南

生成或修复基于 `cumcmthesis` / `cumcm2026` 的分章节论文项目时，使用本参考。

## 章节文件

`sections/*.tex` 内不要包含：

```tex
\documentclass{...}
\begin{document}
\end{document}
```

使用目标文件已有的章节层级。例如 `sections/02_problem_analysis.tex` 已经以 `\section{问题分析}` 开头，则插入内容通常从 `\subsection{...}` 开始。

## 标签

使用稳定、可读、能表达含义的标签：

```tex
\label{fig:q1-cost-compare}
\label{tab:q2-param-sensitivity}
\label{eq:q3-objective}
\label{alg:q1-search}
```

推荐前缀：

- `fig:` 表示图片。
- `tab:` 表示表格。
- `eq:` 表示公式。
- `alg:` 表示算法。
- `sec:` 表示章节。

模板支持 `cleveref` 时优先使用 `\cref{...}`；如果只引用公式且更清楚，也可以使用 `\eqref{...}`。

## 图片

单张图片：

```tex
\begin{figure}[!htbp]
  \centering
  \includegraphics[width=0.78\textwidth]{figures/q1_cost_compare.png}
  \caption{不同策略下的成本对比}
  \label{fig:q1-cost-compare}
\end{figure}

由图~\cref{fig:q1-cost-compare} 可见，...
```

并排图片：

```tex
\begin{figure}[!htbp]
  \centering
  \begin{minipage}[c]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/q2_result_a.png}
    \subcaption{方案一}
    \label{fig:q2-result-a}
  \end{minipage}
  \hfill
  \begin{minipage}[c]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/q2_result_b.png}
    \subcaption{方案二}
    \label{fig:q2-result-b}
  \end{minipage}
  \caption{不同方案的结果对比}
  \label{fig:q2-result-compare}
\end{figure}
```

规则：

- 图片文件放在 `figures/` 下。
- 位图优先使用 PNG/JPG，矢量图优先使用 PDF。
- 避免文件名包含空格或中文标点。
- `\caption` 必须放在 `\label` 前。
- 除非是明确的横向页面或附录大图，不要使用超过 `1\textwidth` 的宽度。

## 表格

优先使用 `booktabs` 风格三线表：

```tex
\begin{table}[!htbp]
  \centering
  \caption{不同模型的评价指标}
  \label{tab:q1-model-metrics}
  \begin{tabularx}{\textwidth}{lYYY}
    \toprule
    模型 & 指标一 & 指标二 & 说明 \\
    \midrule
    基准模型 & 0.912 & 12.5 & 作为对照 \\
    改进模型 & \textbf{0.936} & 18.7 & 综合表现较好 \\
    \bottomrule
  \end{tabularx}
\end{table}
```

规则：

- 文字较多、需要自动换行的列优先使用 `tabularx`。
- 只有模板支持 `siunitx` 且确实需要数字对齐时，才使用 `S` 列。
- 同一列中的数值精度要一致。
- 单位尽量放在表头中。
- 除非用户明确要求，不使用竖线。
- 很大的表格放到附录，正文只保留关键行和结论。

## 公式

后文会引用的公式应编号：

```tex
\begin{equation}
  \min Z = \sum_{i=1}^{n} c_i x_i
  \label{eq:q1-objective}
\end{equation}
```

多行对齐公式：

```tex
\begin{align}
  s_i &= x_i - \bar{x},\\
  \sigma &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}s_i^2}.
  \label{eq:q1-standardization}
\end{align}
```

优化模型标准写法：

```tex
\begin{equation}
\begin{aligned}
  \min \quad & Z(\bm{x}) \\
  \text{s.t.}\quad
  & g_j(\bm{x}) \le 0,\quad j=1,2,\ldots,m,\\
  & \bm{x} \in \Omega .
\end{aligned}
\label{eq:q1-optimization}
\end{equation}
```

规则：

- 行内公式使用 `$...$`，展示公式使用 `equation` 或 `align`。
- 避免使用 `$$...$$`。
- 重要符号必须在使用前或使用后立即解释。
- 数学环境里不要直接写中文标点；确有文字说明时使用 `\text{...}`。
- 如果模板支持 `bm`，向量和矩阵可使用 `\bm{x}`。

## 特殊字符

正文中的以下字符需要转义：

```tex
\% \& \# \_ \{ \}
```

文件路径、标签名或已经合法的 LaTeX 命令中不要误转义。

常见错误：

- 正文里的 `20%` 应写成 `20\%`。
- 正文里的 `A_B` 应写成 `A\_B`，除非它是数学公式 `$A_B$`。
- 不要在论文正文中出现类似 `C:\Users\...` 的本机路径。

## 引用和参考文献

手写参考文献示例：

```tex
灰色预测模型常用于小样本时间序列预测~\cite{ref:grey}.

\begin{thebibliography}{99}
  \bibitem{ref:grey}
  邓聚龙. 灰色系统理论教程[M]. 武汉: 华中理工大学出版社, 1990.
\end{thebibliography}
```

规则：

- 每个 `\cite{key}` 必须有匹配的 `\bibitem{key}`。
- 参考文献列表中的条目应尽量都在正文中被引用。
- 不要编造文献信息；缺失信息要标为需要确认。

## 常见编译修复

- `Undefined control sequence`：检查是否缺少宏包、命令拼写错误，或从网页复制了类似反斜杠的 Unicode 字符。
- `Missing $ inserted`：通常是文本模式中直接写了数学符号，或 `$` 没有成对出现。
- `File not found`：检查 `figures/` 路径和文件扩展名。
- `Runaway argument`：通常是 `{...}` 不匹配，或未转义的 `%` 把行尾注释掉。
- `Misplaced alignment tab character &`：正文中的 `&` 要写成 `\&`，表格或对齐环境中的 `&` 才是分隔符。
- 不可见 Unicode 字符：替换异常空格、全角反斜杠和复制带来的控制字符。
