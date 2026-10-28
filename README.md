# Kalshi Macro Signal Pipeline

> 从预测市场原始成交数据，到可解释的宏观事件信号，再到利率资产的择时回测——一条**全链路、可复现**的研究流水线。

![python](https://img.shields.io/badge/python-3.10%2B-3776ab) ![steps](https://img.shields.io/badge/pipeline-42%20scripts-orange) ![status](https://img.shields.io/badge/status-research-blue)

---

## 设计哲学

这套 pipeline 不追求"预测单个 market 是否命中"，而是回答三个对交易真正有用的问题：

1. **市场在某一时刻是否出现了宏观信息冲击？**
2. **这个冲击是"方向性跳跃"，还是"阈值反转"？**
3. **这些冲击能否转化为对利率资产未来收益的统计优势？**

对应地，整条链路被拆成六个阶段，每个阶段都留下可审计的中间产物：

```
数据标准化 → 主题归因 → 单因子验证 → 策略信号 → 回测 → 图表报告
```

---

## 目录结构

```
kalshi-macro-signal-pipeline/
└── pipeline/
    ├── 00_build_topic_jsons.py            # 主题市场构建
    ├── 00b_build_official_event_tag_map.py
    ├── 01_load_markets.py                 # 落地标准市场表
    ├── 02_ai_event_parse.py               # LLM 解析事件方向/强度
    ├── 02b_event_asset_map.py
    ├── 02c_build_core_market_pool.py
    ├── 03_build_market_prob.py            # 概率时间序列
    ├── 03b_single_factor_test.py          # ⭐ 单因子检验（先验有效性）
    ├── 03c_build_kalshi_distributions.py  # CDF → PDF 分布还原
    ├── 03c_market_quality_filter.py
    ├── 03d_build_kalshi_research_features.py
    ├── 03e_validate_kalshi_research_features.py
    ├── 03f_kalshi_single_factor_evaluation.py
    ├── 03g_kalshi_ffr_event_study.py      # 事件研究
    ├── 03h_kalshi_factor_validation_pack.py
    ├── 03i_kalshi_factor_predictiveness_quickscan.py
    ├── 03j_build_kalshi_factor_backtest_table.py
    ├── 04_build_event_signal.py           # ⭐ 双策略信号（jump / reversal）
    ├── 05_predict_returns.py              # 信号 → 未来收益
    ├── 05b_monthly_asset_rotation.py
    ├── 06_build_report_charts.py          # 分析师图表包
    ├── 06c_plot_paper_style_kalshi_figures.py
    ├── 07_build_paper_report_bundle.py
    ├── 07_export_report_docs.py
    ├── 07b_plot_manager_dashboard.py
    ├── 07c_build_kalshi_cross_asset_pack.py
    ├── 08_build_single_asset_timing_pack.py
    ├── 08b_build_zt_timing_backtest.py    # ZT 国债期货择时
    ├── 08c_build_zt_timing_signal_backtest.py
    ├── 08d_plot_zt_timing_equity_curve.py
    ├── 08e_build_yield_proxy_timing_backtest.py
    ├── 08f_plot_yield_proxy_timing.py
    ├── 08g_build_yield_proxy_timing_tdd.py
    ├── 08h_plot_yield_proxy_timing_tdd.py
    ├── 08i_build_etf_timing_backtest.py
    ├── 08j_plot_etf_timing.py
    ├── 09_build_mode_only_macro_scorecard.py
    ├── 09a_grid_search_kalshi_signal.py   # ⭐ 严格提前 N 天网格搜索
    ├── 10_mode_final_report.py
    ├── 11_rebuild_kalshi_mode_from_markets_trades.py
    └── 11_rebuild_kalshi_modes_from_raw.py
```

> 脚本内部用 `Path(__file__).resolve().parents[1]` 定位项目根目录，因此**必须保持在 `pipeline/` 子目录下**才能正确解析 `output/`、`out_csv/` 路径。

---

## 六个阶段详解

### 阶段 A｜主题与事件（00 – 02c）

把散乱的市场问题文本，归到宏观主题（`central_banks`、`inflation`、`labor` …），并给每个市场打上方向与强度标签。

- 数据源优先本地 parquet，缺失时回退到 Gamma API 并缓存；
- `02_ai_event_parse.py` 在有 `N1N_API_KEY` 时调用 LLM，无 key 时**自动降级为本地启发式规则**，保证离线可跑。

### 阶段 B｜概率与分布（03 – 03c）

- 由成交记录聚合出分钟级 `prob / volume / trades`；
- 对多 strike 的利率合约，用**累计概率差分**还原日频概率密度（PDF），得到 `mean / median / mode / variance / skewness` 六个矩特征。

### 阶段 C｜单因子检验（03b、03d – 03j）

**在合成策略之前**先验证每个因子本身是否有效，这是整套框架最关键的一步。

检验的因子包括：

| 因子 | 含义 |
|---|---|
| `dprob` | 概率一阶变化 |
| `dlogit` | logit 空间变化（对极端概率更稳健） |
| `factor_jump` | 短窗口概率跳跃 |
| `factor_reversal` | 穿越阈值后的反转 |
| `factor_vol_jump` | 波动率突增 |

评估口径：IC、多空价差（spread）、命中率，并按主题分别给出**动态阈值**——避免用全局固定阈值掩盖主题间的量纲差异。

### 阶段 D｜双策略信号（04 – 05b）

**Jump 策略**（量价跳跃）

```
score = direction × magnitude × sign(dlogit) × sqrt(|dlogit|)
```

过滤条件 `|dprob| > topic_specific_threshold`，主题级聚合输出 `signal_sum / signal_mean / signal_volume_weighted` 三档。

**Reversal 策略**（阈值反转）

反转中枢不是固定 50%，而是按同题多选项组动态计算：

```
pivot = 1 / group_size      # 单市场时退化为 0.5
```

只有当概率**从 pivot 一侧穿越到另一侧**、且穿越幅度超过主题阈值时才触发。

两条策略输出结构完全一致，便于横向对比。

### 阶段 E｜图表与报告（06 – 07c）

自动生成分析师风格图表包：单因子 IC 热力图、价差排名、分 horizon 胜率、Profit Factor、主题信号时序与分布、Top 信号案例表。

### 阶段 F｜择时回测（08x）

把宏观信号落到具体标的：

- ZT 国债期货（分钟级对齐）
- 收益率代理组合
- 长债 ETF

每种标的都同时输出净值曲线、回撤、换手与分年度指标。

### 阶段 G｜mode 点预测（09 – 11）

用分布的**众数**作为点预测，做**严格提前 N 天**的评估（`window ∈ {1,3,5,7,10,15,...,50}`），避免"临到期才预测"造成的信息泄漏。

`09a_grid_search_kalshi_signal.py` 在这套口径下搜索 `topN / stack / selective reject / quantize` 的组合，输出 MAE / RMSE 矩阵与信号选择矩阵。

---

## 快速开始

```bash
pip install pandas numpy matplotlib pyarrow scipy scikit-learn requests
```

```bash
export N1N_API_KEY="your_key"   # 可选；不设则走启发式降级

python pipeline/00_build_topic_jsons.py
python pipeline/01_load_markets.py
python pipeline/02_ai_event_parse.py
python pipeline/03_build_market_prob.py
python pipeline/03b_single_factor_test.py
python pipeline/04_build_event_signal.py
python pipeline/05_predict_returns.py
python pipeline/06_build_report_charts.py
```

---

## 结果解读顺序

建议按这个顺序读，不要一上来只看收益率：

1. **`single_factor_summary.csv`** — 哪些因子在多数主题上 IC 为正、spread 稳定；
2. **`event_backtest_results.csv`** — `jump` 与 `reversal` 在不同 horizon 的胜率与 PF；
3. **`output/report/*case_top_signals.csv`** — 高信号窗口是否与真实宏观事件叙事对得上。

至少联合观察 `n_trades`、`win_rate`、`profit_factor`、`avg_strat_ret` 四项，单看任何一项都会误导。

---

## 工程边界

- 本框架定位是**研究与策略筛选**，不是生产交易系统；未纳入交易成本与滑点。
- LLM 语义归因对提示词与模型版本敏感，正式实验应固定版本并做回溯记录。
- Reversal 的同题组识别目前基于**文本规则**，后续可升级为显式事件图谱匹配。
- 部分脚本依赖外部宏观数据（FRED、FOMC 决议），需自行准备并核对时点。
