# 司法信息来源敏感性实验

本项目配套论文 **Judicial Source Sensitivity in Large Language Models: A Controlled Study of Procedural Role Misattribution in Legal Decision-Making**，研究大语言模型如何利用原告、被告与法院的信息来源，以及来源缺失或错误归属如何改变裁判结果重建。

实验使用 400 个中国劳动争议案件、六种输入条件和三个模型，共 7,200 个模型–案件–条件实例。输入包含法院已经查明的事实和法律规则，因此任务是**受控裁判结果重建**，不代表未审案件的前瞻性预测能力。

## 实验设计

每案包含原告请求 P、被告抗辩 D、法院查明事实 F 和法律规则 R。六条件共用同一套冻结中性化 P/D/F 文本，减少直接暴露原始来源的表达；R 和标准答案始终不变。

| 条件 | 来源处理 | 正式数据文件 |
| --- | --- | --- |
| S1 | 中性 P/D/F 正确归属 | `ccf_400_S1_correct_roles.json` |
| S2 | 去除角色标题，按 P/D/F/R 顺序拼接 | `ccf_400_S2_no_roles.json` |
| S3 | P/D/F 严格循环错配，无角色保持原位 | `ccf_400_S3_mismatched_roles.json` |
| S4 | 交换 P 与 D，F 不变 | `ccf_400_S4_P_D_swap.json` |
| S5 | 交换 P 与 F，D 不变 | `ccf_400_S5_P_F_swap.json` |
| S6 | 交换 D 与 F，P 不变 | `ccf_400_S6_D_F_swap.json` |

S3 的两种循环置换按案件提前冻结，所有模型共用同一映射。模型程序直接使用条件文件，不再次交换字段。中性化不保证消除所有语义或风格线索。

模型只输出一个 A–D 标签，判断对象是原告最终保留的实体请求。撤回或放弃的请求不计入，诉讼费用通常不作为独立实体请求。

| 标签 | 裁判结果 | 案件数 |
| --- | --- | --- |
| A | 全部实体请求获支持 | 122 |
| B | 实体请求部分获支持 | 112 |
| C | 全部实体请求被驳回 | 160 |
| D | 因程序性处置未进行实体裁判 | 6 |

标准答案位于 `data/raw/ccf_400_verified_gold.json`，来自既有人工标注。gold 文件保留原始来源文本，仅用于本地评估，不加入模型提示词。

## 仓库结构

```text
config/.env.example                  接口配置示例
data/raw/                           冻结的 S1–S6 和 gold
data/source/                        候选、冻结来源、中性文本和 S3 映射
scripts/build_labor_candidates.py    原始文书规则筛选及 P/D/F 提取
scripts/build_s_conditions.py        500→400 去重、六条件构造及完整性检查
scripts/build_legacy_s_conditions.py 历史字段组合实验，不用于本论文
src/main.py                         模型调用、断点续跑及结果导出
src/evaluator.py                    六条件指标与配对统计
tests/test_role_conditions.py        构造测试及完整数据回归
results/                            归档预测与原始日志
results/metrics/all_models_S1_S6/    归档的六条件统计结果
docs/data_preparation.md             数据处理细节及可追溯范围
```

原始语料来自 ModelScope 的 KLGR123/wenshu_dataset，完整原始语料不随仓库提供。详见 [数据准备说明](docs/data_preparation.md)。

## 安装

以下命令使用 Windows PowerShell。已有仓库可跳过克隆，在项目根目录运行其余命令。

```powershell
git clone https://github.com/yangyanghaoren/ccf_role_experiment.git
cd ccf_role_experiment
python -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r requirements.txt
```

依赖包括 numpy、pandas、scikit-learn、scipy、openai 和 python-dotenv，尚未锁定版本。归档复现时应记录实际 Python 和依赖版本。

## 复现数据构造

论文数据链为：643 案候选，经完整性和任务适用性审核保留 500 案，再经近重复检测保留 400 案。候选池脚本负责规则筛选及 P/D/F 提取，不替代 643→500 的人工审核、R 整理或人工标签复核。正式构造复用冻结的 500 案材料。

去重拼接 P/D/F/R，执行 Unicode NFKC 规范化、连续数字替换为 `<NUM>` 和空白删除；使用字符级 TF-IDF 3–5 gram 和余弦相似度。相似度 ≥0.75 时建立连接，以连通分量为重复簇，每簇保留原始编号最小的记录。规范化仅用于去重，不改写实验文本。

```powershell
# 完整去重与六条件构造；输出目录必须不存在或为空
.\.venv-local\Scripts\python.exe scripts/build_s_conditions.py build --output-dir data/processed/role_experiment_S1_S6

# 仅生成去重审计
.\.venv-local\Scripts\python.exe scripts/build_s_conditions.py build --dedup-only --output-dir data/processed/dedup_audit

# 构造测试与完整 500 案回归，无需接口密钥
.\.venv-local\Scripts\python.exe -m unittest discover -s tests -v
```

冻结数据得到 38 个重复簇，剔除 100 案、保留 400 案。数量不符时程序保存审计并停止，不强行删样。生成前核验来源编号、材料哈希和既有标签；生成后逐案检查六条件的编号集合、Category、R、文本映射和答案隔离。

输出包含去重报告、重复簇、相似度边、剔除清单、冻结来源、七个正式 JSON 和 manifest。回归测试将 S1–S6、gold 和冻结来源与仓库数据逐条比较。

中性文本来自 `data/source/ccf_400_role_texts.json`，不在构造时重新改写。需要从现有 S2/S3 恢复标注时，输出到新文件：

```powershell
.\.venv-local\Scripts\python.exe scripts/build_s_conditions.py prepare-role-texts --output data/processed/recovered_role_texts.json
```

若需从原始语料重新生成规则候选池：

```powershell
.\.venv-local\Scripts\python.exe scripts/build_labor_candidates.py --input-dir D:\path\to\train --output-dir data/processed/labor_candidate_pool
```

将输入路径替换为实际语料目录。该步骤不保证自动得到论文人工审核后的 500 案样本。

## 复现归档结果

统计分析仅使用本地结果，不调用模型，也不需要 API 密钥：

```powershell
.\.venv-local\Scripts\python.exe src/evaluator.py
```

默认读取 GLM-5.3、GPT-5.6、Qwen3.8-Max 各自的 `S1_S2_S3_results.json` 和 `S4_S5_S6_results.json`，合并六条件结果。输出到 `results/metrics/all_models_S1_S6/`，会更新该目录中的同名文件。

评估要求每个模型、每种条件有 400 个唯一案件，并验证有效标签、固定 gold 和一致案件集合。无效标签、重复记录或案件集合不一致会直接报错，不静默删除。

| 输出 | 内容 |
| --- | --- |
| `01_basic_metrics.csv` | Accuracy、Macro-F1、Weighted-F1 |
| `02_classwise_metrics.csv` | 各类别 Precision、Recall、F1、Support |
| `03_confusion_*.csv` | 混淆矩阵及按真实类别归一化的矩阵 |
| `04_paired_comparisons.csv` | 翻转、有害/有益变化、精确 McNemar 和 Holm 校正 |
| `05_bootstrap_ci.csv` | 配对 Accuracy、Macro-F1 差值的 95% 置信区间 |
| `06_transition_*.csv` | 同一案件在不同条件下的预测转移矩阵 |
| `07_efficiency_metrics.csv` | 延迟及 token 使用统计 |
| `08_overall_summary.csv` | 六条件汇总 |
| `09_accuracy_table_percent.csv` | Accuracy 百分比表 |
| `10_macro_f1_table.csv`、`11_weighted_f1_table.csv` | F1 汇总表 |

九项主比较为 S1–S2、S1–S3、S2–S3、S1–S4、S1–S5、S1–S6、S4–S5、S4–S6、S5–S6。每个模型内对九个精确 McNemar p 值进行 Holm 校正，显著性阈值为校正后 p<0.05。

配对 Bootstrap 重采样 10,000 次，固定随机种子 20260901，两条件共用相同案件索引，采用 2.5% 和 97.5% 百分位区间。区间不作多重比较校正；比较 a→b 时，`delta_accuracy` 为 Accuracy(b)−Accuracy(a)。方向性转移分析属于探索性描述。

## 重新调用模型

将 `config/.env.example` 复制为本地 `config/.env`，填写实际接口配置：

```dotenv
CLOSEAI_API_KEY=your_api_key_here
CLOSEAI_BASE_URL=https://your-api-endpoint.example/v1
MODEL=gpt-5.6
```

`config/.env` 被 Git 忽略，不应提交密钥。接口需要支持请求的模型标识和参数。

```powershell
.\.venv-local\Scripts\python.exe -m src.main
```

当前程序每次运行一个模型的全部六条件，共 2,400 个实例；三个模型需分别配置并运行。系统提示词见 `src/main.py` 的 `SYSTEM_PROMPT`。请求使用 `reasoning_effort=low`，其他解码参数沿用接口默认设置。

| 配置 | 当前值 |
| --- | --- |
| TEST_LIMIT | None，全部案件 |
| MAX_WORKERS | 8 |
| REQUEST_TIMEOUT | 90 秒 |
| MAX_RETRIES | 最多 5 次尝试 |
| RETRY_WAIT | 失败后按 3、6、9、12 秒等待 |

日志保存请求及返回模型标识、预测、延迟和 token 信息。有效预测按 `(case_id, setting, model)` 跳过，失败任务继续重试。模型调用会发送案件材料并产生接口费用。

当前输出为 `<model>_raw_results.jsonl`、`<model>_S1_S2_S3_S4_S5_S6_results.json` 和同名 CSV。合并文件与归档的两批结果命名不同：分析新结果时，在评估入口的 `result_paths` 中改为对应合并文件，不同时传入新旧结果造成重复。

修改输入、提示词或推理设置后，应在独立仓库副本运行，或先归档对应旧日志，避免断点续跑复用旧预测。构造输出位于指定目录，模型程序仍读取 `data/raw/`；使用新输入前需明确切换数据路径并核验版本。

## 复现范围

代码能复现冻结材料上的去重、条件构造和统计分析，不重建历史人工审核或中性化改写过程。商业模型版本和接口行为可能变化，重新调用不保证得到相同预测。

论文归档应保留数据及提示词版本、请求与返回模型标识、推理设置、运行日期、依赖版本和不含密钥的接口信息。类别 D 仅有 6 案，类别指标需结合样本量解释；当前样本及三个模型的行为不能直接推广到其他法律领域。
