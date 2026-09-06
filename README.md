# CCF 劳动争议案件司法角色实验

本项目用于论文中的大语言模型裁判结果分类实验：在同一批 400 个劳动争议案件上，比较正确角色、无角色和错配角色三种输入条件下的预测表现。项目包含正式实验数据、模型调用程序、统计评估程序，以及数据清洗与条件构造代码。

## 1. 实验任务

模型根据案件材料输出一个标签：

| 标签 | 含义 |
| --- | --- |
| A | 全部支持 |
| B | 部分支持 |
| C | 全部不支持 |
| D | 因程序性原因未进入实体裁判 |

判断对象是当事人在庭审中最终保留的实体诉讼请求；撤回或放弃的请求不计入，诉讼费一般不作为独立实体请求。具体提示词见 `src/main.py` 的 `SYSTEM_PROMPT`。

| 正式条件 | 文件 | 模型接收的材料 |
| --- | --- | --- |
| S1：正确角色 | `ccf_400_S1_correct_roles.json` | 按原告诉称 P、被告抗辩 D、法院查明事实 F、法律规则 R 分块 |
| S2：无角色 | `ccf_400_S2_no_roles.json` | `text` 字段，统一放在“案件材料”标题下 |
| S3：错配角色 | `ccf_400_S3_mismatched_roles.json` | 使用文件中预先处理的 P/D/F/R，并添加与 S1 相同的标题 |

S3 使用与 S2 相同的中性化 P/D/F 段落，按逐案固定的完全错配置换重新分配角色，R 保持不变。S1 使用原始角色文本。模型调用程序不会再次交换角色；中性文本和逐案置换已从现有材料恢复到 `data/source/ccf_400_role_texts.json`，生成时进行来源哈希核验。恢复已有文本不等同于重建历史中性化改写算法。

标准答案位于 `data/raw/ccf_400_verified_gold.json`，仅用于本地比较，不加入模型提示词。当前标签分布为 A=122、B=112、C=160、D=6。

## 2. 目录结构

```text
ccf_role_experiment/
├── README.md
├── requirements.txt
├── config/
│   ├── .env                  # 本地接口配置
│   └── .env.example          # 不含密钥的配置示例
├── scripts/
│   ├── build_labor_candidates.py  # 原始案件清洗及 P/D/F 提取
│   └── build_s_conditions.py      # 正式 500→400 去重及 S1/S2/S3 生成
├── data/
│   ├── raw/                  # 正式 S1/S2/S3 和 gold，共四个 JSON
│   ├── source/               # 500 案输入、400 案来源及角色文本标注
│   └── processed/            # 清洗输出，运行脚本后生成
├── src/
│   ├── main.py               # 并发调用模型、断点续跑、结果导出
│   └── evaluator.py          # 指标、配对检验和置信区间
└── results/
    ├── <model>_raw_results.jsonl
    ├── <model>_S1_S2_S3_results.json
    ├── <model>_S1_S2_S3_results.csv
    └── metrics/all_models/   # 已保存的统计结果
```

原始 `train` 数据位于 `E:\PythonProject\train`，未复制到本项目。清洗脚本可通过参数直接读取该目录。

## 3. 环境准备

以下命令使用 Windows PowerShell。进入项目根目录后创建新环境：

```powershell
cd E:\ccf_role_experiment
py -3.13 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r requirements.txt
```

现有 `.venv` 指向的 `E:\Anaconda\python.exe` 在本次检查中不可用，因此示例使用新建的 `.venv-local`。在 PyCharm 中也可将解释器设置为该环境的 `Scripts\python.exe`。

依赖包含 numpy、pandas、scikit-learn、scipy、openai 和 python-dotenv。现有依赖文件没有锁定版本；正式复现实验时应另外保存实际使用的依赖版本。初筛脚本仅使用 Python 标准库；正式条件脚本使用 numpy、scipy 和 scikit-learn。

## 4. 数据清洗

### 4.1 生成候选池

`scripts/build_labor_candidates.py` 用于案件筛选、去重和 P/D/F 提取，支持命令行路径参数、输入检查及 QC 统计导出。

```powershell
py -3.13 scripts/build_labor_candidates.py --input-dir E:\PythonProject\train
```

默认输出到本项目的 `data/processed/labor_candidate_pool/`。也可通过 `--output-dir` 指定其他目录；省略 `--input-dir` 时读取本项目的 `data/train/`。

输入为递归目录中的 JSON 文件，顶层包含 `ctxs` 字典，每个案件包含 `CaseId`、`Case`、`Category`、`JudgeAccusation`、`JudgeReason`、`JudgeResult` 等字段。

主要处理步骤：

1. 筛选 `cat_1=劳动人事` 的案件，按 CaseId 和规范化文本哈希去重。
2. 限定二级类别为工资福利、劳动合同、劳动争议，文书类型为判决书或裁定书。
3. 排除必要文本缺失，以及诉辩文本或裁判理由长度恰好为 700 字符的疑似截断记录。
4. 通过规则提取原告诉称 P、被告抗辩 D、法院查明事实 F，并判断抗辩类型、边界及潜在结果泄漏。
5. 分为 `CLEAR`、`REVIEW`、`EXCLUDED`，输出候选池及复核记录。`CLEAR` 是规则筛选状态，不等同于人工核验通过。

| 输出文件 | 内容 |
| --- | --- |
| `candidate_clear.json` | 精简候选记录，含 P/D/F 和 JudgeResult |
| `candidate_clear_full.json` | 完整候选记录，含来源、原文和 QC 信息 |
| `candidate_review.json` | 待人工复核记录 |
| `excluded_cases.json` | 被排除记录及原因 |
| `qc_statistics.json` | 扫描、去重、QC 与类别分布 |
| `qc_review.csv` | 方便人工查看的复核清单 |

### 4.2 正式 500→400 案去重与角色条件生成

`scripts/build_s_conditions.py` 默认读取 `data/source/ccf_500_candidates.json`，该文件包含 500 条 P/D/F/R 均非空的待去重材料。

去重时依次拼接 P、D、F、R，实施 Unicode NFKC 规范化，将连续数字（正则 `\d+`）替换为 `<NUM>`，删除全部空白。使用字符级 TF-IDF（3–5 gram；小写转换、平滑 IDF、L2 归一化），计算余弦相似度。相似度 ≥0.75 的文书之间建立边，以连通分量作为重复簇，每簇保留原始编号最小的文书，保留样本按输入顺序排列。规范化仅用于去重，不修改正式提示词材料。

本次完整运行得到 **38 个重复簇、361 条阈值边、剔除 100 份、保留 400 份**。保留的原始编号与已有正式来源文件完全一致。代码不会为凑足 400 条而随机删样；数量断言不满足时保存审计并停止条件生成。

```powershell
# 从项目根目录运行；依赖已安装到 .venv-local
.\.venv-local\Scripts\python.exe scripts/build_s_conditions.py build
# 已有输出时使用新的空目录；程序拒绝覆盖非空目录
.\.venv-local\Scripts\python.exe scripts/build_s_conditions.py build --output-dir data/processed/role_experiment_rerun
# 只运行去重与审计
.\.venv-local\Scripts\python.exe scripts/build_s_conditions.py build --dedup-only --output-dir data/processed/dedup_audit
```

默认输出在 `data/processed/role_experiment/`。本项目已保存一次完整运行，因此再次运行默认命令时需改用新的输出目录。

| 输出 | 内容 |
| --- | --- |
| `dedup_report.json` | 参数、环境版本、输入哈希、实测数量及保留编号 |
| `duplicate_clusters.json` | 每个重复簇的成员、代表与剔除编号 |
| `similarity_edges.json` | 所有达到阈值的样本对及相似度 |
| `removed_cases.json` | 剔除编号、对应代表与簇编号 |
| `deduplicated_candidates.json` | 保留原始编号的 400 条候选 |
| `source/ccf_400_full_source.json` | 冻结样本、原始编号、既有 gold 与来源信息 |
| `raw/ccf_400_*.json` | 正式 S1、S2、S3 与 gold 四个文件 |
| `manifest.json` | 条件定义、标注来源哈希、标签分布及输出哈希 |

生成条件前，脚本严格匹配 `ccf_400_full_source.json` 中的原始编号、材料和 gold，以及 `ccf_400_role_texts.json` 中的材料哈希。S1 保留原始 P/D/F；S2 按 P、D、F 顺序拼接已有中性文本；S3 将同一套中性文本按已冻结的逐案映射完全错配。所有条件保留相同 R，gold 不写入三组模型输入。新生成的四个正式数据文件及来源文件均与现有文件 JSON 内容完全一致。

如需从已有 S2/S3 重新恢复角色文本标注，可运行以下命令生成新文件。该步骤复用历史文本，不执行新的自动中性化或人工核验：

```powershell
.\.venv-local\Scripts\python.exe scripts/build_s_conditions.py prepare-role-texts --output data/processed/recovered_role_texts.json
```

验证命令：

```powershell
.\.venv-local\Scripts\python.exe -m unittest discover -s tests -v
```

详细方法、数据来源与论文措辞见 [docs/data_preparation.md](docs/data_preparation.md)。

## 5. 运行正式模型实验

已有 `config/.env` 时直接编辑；首次配置可从 `.env.example` 复制。配置项为：

```dotenv
CLOSEAI_API_KEY=your_api_key_here
CLOSEAI_BASE_URL=https://your-api-endpoint.example/v1
MODEL=qwen3.8-max
```

填写实际可用的 OpenAI 兼容接口及模型标识，然后在项目根目录运行：

```powershell
.\.venv-local\Scripts\python.exe -m src.main
```

当前程序每次运行一个模型。现有结果的模型标识为 `glm-5.3`、`gpt-5.6`、`qwen3.8-max`，每个模型各 1,200 条结果，即 400 案 × 3 条件；这些名称是本项目保存的请求标识，接口实际返回的模型另记在 `returned_model`。

`src/main.py` 中的主要配置：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| TEST_LIMIT | None | 使用全部案件；小规模试跑可改为 5 |
| MAX_WORKERS | 5 | 并发线程数 |
| REQUEST_TIMEOUT | 90.0 | 单次请求超时秒数 |
| MAX_RETRIES | 5 | 最多尝试次数 |
| RETRY_WAIT | 3 | 失败后按 3、6、9、12 秒等待 |

程序当前统一发送 `extra_body={"reasoning_effort": "low"}`，更换模型时需按所用接口支持情况调整。调用会实际发送案件材料并产生接口请求。

每完成一条任务立即追加 JSONL 日志。重跑时按照 `(case_id, setting, model)` 跳过已有有效预测，失败任务会重试；最终整理为 JSON 和 CSV。**已有完整日志时，同名模型通常不会重新请求。** 更改数据、提示词或推理参数后，应先将对应模型的已有结果移入单独的备份目录，再开始新一轮，避免混用旧预测。

## 6. 统计评估

评估仅读取本地结果，无需调用模型。现有评估入口使用相对于 `src` 的路径，因此按以下方式运行：

```powershell
Push-Location src
..\.venv-local\Scripts\python.exe evaluator.py
Pop-Location
```

默认评估上述三个模型，Bootstrap 重采样 10,000 次，随机种子为 20260901。输出会更新 `results/metrics/all_models/` 中的同名文件。

| 输出 | 内容 |
| --- | --- |
| `01_basic_metrics.csv` | Accuracy、Macro-F1、Weighted-F1 |
| `02_classwise_metrics.csv` | 各类别 Precision、Recall、F1、Support |
| `03_confusion_*.csv` | 各模型、条件的混淆矩阵 |
| `04_paired_comparisons.csv` | 预测翻转率、有害/有益翻转、精确 McNemar 检验 |
| `05_bootstrap_ci.csv` | 配对指标差值的 95% Bootstrap 置信区间 |
| `06_transition_*.csv` | 条件之间的预测转移矩阵 |
| `07_efficiency_metrics.csv` | 延迟和 token 使用量的均值、中位数、P95 |

对条件 a→b，有害翻转表示由正确变为错误，有益翻转表示由错误变为正确；`delta_accuracy` 按 a−b 计算。评估会排除无效预测，配对统计使用 S1/S2/S3 均有有效预测的案件，因此论文中应同时报告有效样本量。评估器对重复记录只给出警告，建议使用主程序整理后的 JSON，避免直接传入包含重复尝试的原始 JSONL。

## 7. 复现记录

论文归档时建议保存数据版本、提示词、模型请求及返回标识、接口配置（不含密钥）、推理参数、依赖版本和运行日期。当前项目提供已有实验结果，但 README 不将其扩展为未经核验的论文结论。
