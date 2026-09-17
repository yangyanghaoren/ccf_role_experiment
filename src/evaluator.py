# src/evaluator.py
# -*- coding: utf-8 -*-

"""
CCF 司法角色实验统计分析（S1-S6）
================================

输出：
1. Basic Metrics: Accuracy / Macro-F1 / Weighted-F1
2. Classwise Metrics: Precision / Recall / F1 / Support
3. Confusion Matrices: 原始计数 + 按真实类别归一化
4. Paired Comparisons: Flip / Harmful / Helpful / McNemar Exact Test
   + Holm 多重比较校正
5. Paired Bootstrap 95% CI
6. Prediction Transition Matrices
7. Latency / Token / Reasoning Token 汇总
8. S1-S6 Overall Summary
9. Accuracy Table (%)
10. Macro-F1 Table
11. Weighted-F1 Table

说明：
- S1-S6 都必须存在。
- 默认每个 model × setting 应有 400 个唯一 case。
- 无效 gold/pred、重复记录、case 集不一致会直接报错，而不是静默删除。
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)

from scipy.stats import binomtest


class ExperimentEvaluator:
    LABELS = ["A", "B", "C", "D"]
    SETTINGS = ["S1", "S2", "S3", "S4", "S5", "S6"]

    # 论文主比较：
    # 1) S1/S2/S3：正确角色、无角色、全面错配
    # 2) S1 vs S4/S5/S6：每一种局部错配相对正确角色的损失
    # 3) S4/S5/S6：机制差异
    COMPARISONS = [
        ("S1", "S2"),
        ("S1", "S3"),
        ("S2", "S3"),
        ("S1", "S4"),
        ("S1", "S5"),
        ("S1", "S6"),
        ("S4", "S5"),
        ("S4", "S6"),
        ("S5", "S6"),
    ]

    def __init__(
        self,
        result_paths,
        output_dir="results/metrics",
        bootstrap_n=10000,
        random_seed=20260901,
        expected_n=400,
        print_matrices=True,
    ):
        self.result_paths = [Path(p) for p in result_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bootstrap_n = int(bootstrap_n)
        self.random_seed = int(random_seed)
        self.expected_n = expected_n
        self.print_matrices = bool(print_matrices)

        self.df = self._load_results()
        self._validate()

    # =========================================================
    # 1. 数据读取
    # =========================================================

    def _load_results(self):
        all_records = []

        for result_path in self.result_paths:
            if not result_path.exists():
                raise FileNotFoundError(
                    f"结果文件不存在：{result_path.resolve()}"
                )

            suffix = result_path.suffix.lower()

            if suffix == ".json":
                with open(result_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    raise ValueError(
                        f"{result_path} 最外层必须是 list"
                    )
                all_records.extend(data)

            elif suffix in {".jsonl", ".ndjson"}:
                with open(result_path, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            all_records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"{result_path} 第 {line_no} 行 JSON 解析失败：{e}"
                            ) from e

            elif suffix == ".csv":
                temp_df = pd.read_csv(result_path)
                all_records.extend(
                    temp_df.to_dict(orient="records")
                )

            else:
                raise ValueError(
                    f"不支持文件格式：{result_path}"
                )

        if not all_records:
            raise ValueError("没有读取到任何实验结果。")

        return pd.DataFrame(all_records)

    # =========================================================
    # 2. 数据完整性检查
    # =========================================================

    def _validate(self):
        required = {
            "case_id",
            "model",
            "setting",
            "gold",
            "pred",
        }

        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"缺少必要字段：{sorted(missing)}"
            )

        # 统一关键字段格式，避免 23 与 "23" 被当成不同案件
        self.df["case_id"] = (
            self.df["case_id"].astype(str).str.strip()
        )
        for col in ["model", "setting", "gold", "pred"]:
            self.df[col] = (
                self.df[col].astype(str).str.strip()
            )

        # 空值/空字符串
        for col in required:
            bad = (
                self.df[col].isna()
                | self.df[col].astype(str).str.strip().eq("")
                | self.df[col].astype(str).str.lower().eq("nan")
            )
            if bad.any():
                raise ValueError(
                    f"字段 {col} 存在 {int(bad.sum())} 条空值/非法空字符串。"
                )

        # gold 必须合法
        invalid_gold = ~self.df["gold"].isin(self.LABELS)
        if invalid_gold.any():
            bad_values = sorted(
                self.df.loc[invalid_gold, "gold"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"发现 {int(invalid_gold.sum())} 条非法 gold：{bad_values}"
            )

        # pred 必须合法：不允许直接删除，否则会虚高 Accuracy
        invalid_pred = ~self.df["pred"].isin(self.LABELS)
        if invalid_pred.any():
            preview = (
                self.df.loc[
                    invalid_pred,
                    ["case_id", "model", "setting", "gold", "pred"],
                ]
                .head(20)
                .to_string(index=False)
            )
            raise ValueError(
                "发现非法预测。非法预测不应直接从统计样本中删除。\n"
                f"数量：{int(invalid_pred.sum())}\n"
                f"示例：\n{preview}"
            )

        # setting 必须在 S1-S6
        unknown_settings = sorted(
            set(self.df["setting"].dropna().unique())
            - set(self.SETTINGS)
        )
        if unknown_settings:
            raise ValueError(
                f"发现未定义 setting：{unknown_settings}"
            )

        # 重复记录直接报错，避免 basic 与 paired 统计口径不一致
        dup = self.df.duplicated(
            subset=["case_id", "model", "setting"],
            keep=False,
        )
        if dup.any():
            preview = (
                self.df.loc[
                    dup,
                    ["case_id", "model", "setting", "gold", "pred"],
                ]
                .sort_values(["model", "case_id", "setting"])
                .head(30)
                .to_string(index=False)
            )
            raise ValueError(
                "发现重复的 (case_id, model, setting) 记录。\n"
                f"重复行数量：{int(dup.sum())}\n"
                f"示例：\n{preview}"
            )

        # 同一 case_id 在所有模型、所有 setting 下 gold 必须一致
        gold_count = (
            self.df.groupby("case_id")["gold"].nunique()
        )
        inconsistent_gold = gold_count > 1
        if inconsistent_gold.any():
            bad_ids = (
                inconsistent_gold[
                    inconsistent_gold
                ].index.tolist()[:20]
            )
            raise ValueError(
                "发现同一 case_id 在不同模型/Setting 下 Gold 不一致。"
                f" 示例 case_id：{bad_ids}"
            )

        # 每个模型都必须包含 S1-S6
        for model, g in self.df.groupby("model"):
            present = set(g["setting"].unique())
            missing_settings = set(self.SETTINGS) - present
            if missing_settings:
                raise ValueError(
                    f"{model} 缺少 Setting："
                    f"{sorted(missing_settings)}"
                )

        # 每个 model × setting 的 case_id 集合必须完全相同
        grouped = list(
            self.df.groupby(["model", "setting"], sort=True)
        )
        if not grouped:
            raise ValueError("没有可用的 model × setting 分组。")

        ref_key, ref_group = grouped[0]
        ref_ids = set(ref_group["case_id"])

        for key, g in grouped:
            ids = set(g["case_id"])
            if ids != ref_ids:
                missing_ids = sorted(ref_ids - ids)[:20]
                extra_ids = sorted(ids - ref_ids)[:20]
                raise ValueError(
                    f"{key} 的 case_id 集合与参考组 {ref_key} 不一致。\n"
                    f"缺少示例：{missing_ids}\n"
                    f"多出示例：{extra_ids}"
                )

        # 当前研究固定为 400 案件；可传 expected_n=None 关闭
        if self.expected_n is not None:
            for key, g in grouped:
                n_unique = g["case_id"].nunique()
                if n_unique != self.expected_n:
                    raise ValueError(
                        f"{key} 唯一 case 数为 {n_unique}，"
                        f"预期应为 {self.expected_n}。"
                    )

        print("\n===== 数据检查 =====")
        print(f"总记录数：{len(self.df)}")

        print("\n各模型 / Setting 数量：")
        print(
            self.df
            .groupby(["model", "setting"])
            .size()
            .to_string()
        )

        completeness = (
            self.df
            .groupby(["model", "setting"])["case_id"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(columns=self.SETTINGS)
        )

        print("\n各 Setting 唯一 case 数：")
        print(completeness.to_string())

        print("\n数据完整性检查：通过")

    # =========================================================
    # 3. 基础指标
    # =========================================================

    def basic_metrics(self):
        rows = []

        for (model, setting), g in self.df.groupby(
            ["model", "setting"]
        ):
            y_true = g["gold"]
            y_pred = g["pred"]

            rows.append({
                "model": model,
                "setting": setting,
                "n": len(g),
                "correct": int((y_true == y_pred).sum()),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    labels=self.LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
                    y_true,
                    y_pred,
                    labels=self.LABELS,
                    average="weighted",
                    zero_division=0,
                ),
            })

        result = pd.DataFrame(rows)
        result["_setting_order"] = result["setting"].map(
            {s: i for i, s in enumerate(self.SETTINGS)}
        )
        result = (
            result
            .sort_values(["model", "_setting_order"])
            .drop(columns=["_setting_order"])
            .reset_index(drop=True)
        )

        result.to_csv(
            self.output_dir / "01_basic_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return result

    # =========================================================
    # 4. 各类别 Precision / Recall / F1
    # =========================================================

    def classwise_metrics(self):
        rows = []

        for (model, setting), g in self.df.groupby(
            ["model", "setting"]
        ):
            precision, recall, f1, support = (
                precision_recall_fscore_support(
                    g["gold"],
                    g["pred"],
                    labels=self.LABELS,
                    zero_division=0,
                )
            )

            for i, label in enumerate(self.LABELS):
                rows.append({
                    "model": model,
                    "setting": setting,
                    "class": label,
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1": f1[i],
                    "support": int(support[i]),
                })

        result = pd.DataFrame(rows)
        result.to_csv(
            self.output_dir / "02_classwise_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return result

    # =========================================================
    # 5. 混淆矩阵：count + row-normalized
    # =========================================================

    def confusion_matrices(self):
        result = {}

        for (model, setting), g in self.df.groupby(
            ["model", "setting"]
        ):
            cm_count = confusion_matrix(
                g["gold"],
                g["pred"],
                labels=self.LABELS,
            )

            cm_norm = confusion_matrix(
                g["gold"],
                g["pred"],
                labels=self.LABELS,
                normalize="true",
            )

            index = [f"Gold_{x}" for x in self.LABELS]
            columns = [f"Pred_{x}" for x in self.LABELS]

            count_df = pd.DataFrame(
                cm_count,
                index=index,
                columns=columns,
            )

            norm_df = pd.DataFrame(
                cm_norm,
                index=index,
                columns=columns,
            )

            count_df.to_csv(
                self.output_dir
                / f"03_confusion_count_{model}_{setting}.csv",
                encoding="utf-8-sig",
            )

            norm_df.to_csv(
                self.output_dir
                / f"03_confusion_normalized_{model}_{setting}.csv",
                encoding="utf-8-sig",
            )

            result[f"{model}_{setting}"] = {
                "count": count_df,
                "normalized": norm_df,
            }

        return result

    # =========================================================
    # 6. 把所有 Setting 拉到同一行
    # =========================================================

    def _paired_predictions(self, model):
        g = self.df[
            self.df["model"] == model
        ].copy()

        # validate 已保证无重复，可直接 pivot
        pivot = g.pivot(
            index="case_id",
            columns="setting",
            values="pred",
        )

        gold = (
            g.groupby("case_id")["gold"]
            .first()
        )
        pivot["gold"] = gold

        return pivot

    # =========================================================
    # 7. 取某一对 Setting 的有效配对
    # =========================================================

    def _get_pair(
        self,
        model,
        setting_a,
        setting_b,
    ):
        p = self._paired_predictions(model)

        missing_columns = [
            s
            for s in [setting_a, setting_b]
            if s not in p.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{model} 缺少 Setting：{missing_columns}"
            )

        p = p[
            [setting_a, setting_b, "gold"]
        ].dropna()

        if self.expected_n is not None and len(p) != self.expected_n:
            raise ValueError(
                f"{model} {setting_a} vs {setting_b} "
                f"有效配对只有 {len(p)}，预期 {self.expected_n}。"
            )

        return p

    # =========================================================
    # 8. Flip + McNemar Exact Test
    # =========================================================

    def paired_comparison(
        self,
        model,
        setting_a,
        setting_b,
    ):
        p = self._get_pair(
            model,
            setting_a,
            setting_b,
        )

        pred_a = p[setting_a]
        pred_b = p[setting_b]
        gold = p["gold"]

        correct_a = pred_a == gold
        correct_b = pred_b == gold

        # 预测标签本身是否变化
        prediction_flip = pred_a != pred_b

        # A 正确 -> B 错误
        harmful = correct_a & ~correct_b

        # A 错误 -> B 正确
        helpful = ~correct_a & correct_b

        # A 错 -> B 仍错，但标签发生变化
        wrong_to_wrong = (
            ~correct_a
            & ~correct_b
            & prediction_flip
        )

        n_harmful = int(harmful.sum())
        n_helpful = int(helpful.sum())
        n_wrong_to_wrong = int(wrong_to_wrong.sum())

        discordant = n_harmful + n_helpful

        # Exact McNemar = 对不一致对做双侧二项精确检验
        if discordant > 0:
            p_value = binomtest(
                min(n_harmful, n_helpful),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        else:
            p_value = 1.0

        accuracy_a = correct_a.mean()
        accuracy_b = correct_b.mean()

        return {
            "model": model,
            "comparison": f"{setting_a}_vs_{setting_b}",
            "setting_a": setting_a,
            "setting_b": setting_b,
            "n": len(p),
            "accuracy_a": accuracy_a,
            "accuracy_b": accuracy_b,

            # B - A；正数 = 后一个 setting 更好
            "delta_accuracy": accuracy_b - accuracy_a,

            "prediction_flip_n": int(prediction_flip.sum()),
            "prediction_flip_rate": prediction_flip.mean(),

            "harmful_flip_n": n_harmful,
            "harmful_flip_rate": harmful.mean(),

            "helpful_flip_n": n_helpful,
            "helpful_flip_rate": helpful.mean(),

            "wrong_to_wrong_flip_n": n_wrong_to_wrong,
            "wrong_to_wrong_flip_rate": wrong_to_wrong.mean(),

            # Helpful - Harmful；正数 = 净增加正确案件
            "net_correctness_gain_n": n_helpful - n_harmful,

            # 暂存 raw p；后续统一做 Holm
            "mcnemar_p_raw": p_value,
        }

    @staticmethod
    def _holm_adjust(p_values):
        """
        Holm-Bonferroni adjusted p-values.
        返回顺序与输入 p_values 一致。
        """
        p_values = np.asarray(p_values, dtype=float)
        m = len(p_values)

        if m == 0:
            return np.array([], dtype=float)

        order = np.argsort(p_values)
        sorted_p = p_values[order]

        adjusted_sorted = np.empty(m, dtype=float)

        running_max = 0.0
        for i, p in enumerate(sorted_p):
            adjusted = (m - i) * p
            running_max = max(running_max, adjusted)
            adjusted_sorted[i] = min(running_max, 1.0)

        adjusted = np.empty(m, dtype=float)
        adjusted[order] = adjusted_sorted
        return adjusted

    # =========================================================
    # 9. 所有主要配对比较 + Holm 校正
    # =========================================================

    def all_paired_comparisons(self):
        rows = []

        for model in sorted(
            self.df["model"].unique()
        ):
            for setting_a, setting_b in self.COMPARISONS:
                rows.append(
                    self.paired_comparison(
                        model,
                        setting_a,
                        setting_b,
                    )
                )

        result = pd.DataFrame(rows)

        # 每个模型内部，对该模型全部预设主比较做 Holm 校正
        result["mcnemar_p_holm"] = np.nan

        for model, idx in result.groupby("model").groups.items():
            idx = list(idx)
            adjusted = self._holm_adjust(
                result.loc[idx, "mcnemar_p_raw"].to_numpy()
            )
            result.loc[idx, "mcnemar_p_holm"] = adjusted

        result["significant_raw_005"] = (
            result["mcnemar_p_raw"] < 0.05
        )
        result["significant_holm_005"] = (
            result["mcnemar_p_holm"] < 0.05
        )

        # 兼容旧代码/旧表命名
        result["mcnemar_p"] = result["mcnemar_p_raw"]
        result["significant_005"] = (
            result["significant_raw_005"]
        )

        result.to_csv(
            self.output_dir / "04_paired_comparisons.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return result

    # =========================================================
    # 10. Paired Bootstrap 95% CI
    # =========================================================

    def bootstrap_difference(
        self,
        model,
        setting_a,
        setting_b,
    ):
        p = self._get_pair(
            model,
            setting_a,
            setting_b,
        )

        n = len(p)

        # 每个比较使用相同 seed，保证可复现
        rng = np.random.default_rng(
            self.random_seed
        )

        gold = p["gold"].to_numpy()
        pred_a = p[setting_a].to_numpy()
        pred_b = p[setting_b].to_numpy()

        acc_diff = np.empty(
            self.bootstrap_n,
            dtype=float,
        )
        macro_diff = np.empty(
            self.bootstrap_n,
            dtype=float,
        )

        for i in range(self.bootstrap_n):
            # paired bootstrap：
            # gold / A / B 使用完全同一个案件索引
            idx = rng.integers(
                0,
                n,
                size=n,
            )

            y = gold[idx]
            a = pred_a[idx]
            b = pred_b[idx]

            acc_diff[i] = (
                accuracy_score(y, b)
                - accuracy_score(y, a)
            )

            macro_diff[i] = (
                f1_score(
                    y,
                    b,
                    labels=self.LABELS,
                    average="macro",
                    zero_division=0,
                )
                -
                f1_score(
                    y,
                    a,
                    labels=self.LABELS,
                    average="macro",
                    zero_division=0,
                )
            )

        observed_acc_diff = (
            accuracy_score(gold, pred_b)
            - accuracy_score(gold, pred_a)
        )

        observed_macro_diff = (
            f1_score(
                gold,
                pred_b,
                labels=self.LABELS,
                average="macro",
                zero_division=0,
            )
            -
            f1_score(
                gold,
                pred_a,
                labels=self.LABELS,
                average="macro",
                zero_division=0,
            )
        )

        acc_ci_low, acc_ci_high = np.percentile(
            acc_diff,
            [2.5, 97.5],
        )

        macro_ci_low, macro_ci_high = np.percentile(
            macro_diff,
            [2.5, 97.5],
        )

        return {
            "model": model,
            "comparison": f"{setting_a}_vs_{setting_b}",
            "setting_a": setting_a,
            "setting_b": setting_b,
            "n": n,

            # B - A
            "delta_accuracy": observed_acc_diff,
            "acc_ci_low": acc_ci_low,
            "acc_ci_high": acc_ci_high,
            "acc_ci_excludes_zero": (
                acc_ci_low > 0
                or acc_ci_high < 0
            ),

            # B - A
            "delta_macro_f1": observed_macro_diff,
            "macro_f1_ci_low": macro_ci_low,
            "macro_f1_ci_high": macro_ci_high,
            "macro_ci_excludes_zero": (
                macro_ci_low > 0
                or macro_ci_high < 0
            ),
        }

    # =========================================================
    # 11. 所有 Bootstrap
    # =========================================================

    def all_bootstrap(self):
        rows = []

        models = sorted(
            self.df["model"].unique()
        )

        total = (
                len(models)
                * len(self.COMPARISONS)
        )

        current = 0

        for model in models:

            for setting_a, setting_b in self.COMPARISONS:
                current += 1

                print(
                    f"\n[Bootstrap {current}/{total}] "
                    f"{model}: "
                    f"{setting_a} vs {setting_b}",
                    flush=True
                )

                start = time.time()

                row = self.bootstrap_difference(
                    model,
                    setting_a,
                    setting_b,
                )

                rows.append(row)

                print(
                    f"[完成] "
                    f"{model}: "
                    f"{setting_a} vs {setting_b} | "
                    f"{time.time() - start:.2f}s",
                    flush=True
                )

        result = pd.DataFrame(rows)

        result.to_csv(
            self.output_dir / "05_bootstrap_ci.csv",
            index=False,
            encoding="utf-8-sig",
        )

        return result

    # =========================================================
    # 12. 标签转移矩阵
    # =========================================================

    def transition_matrix(
        self,
        model,
        setting_a,
        setting_b,
    ):
        p = self._get_pair(
            model,
            setting_a,
            setting_b,
        )

        matrix = pd.crosstab(
            p[setting_a],
            p[setting_b],
            dropna=False,
        )

        matrix = matrix.reindex(
            index=self.LABELS,
            columns=self.LABELS,
            fill_value=0,
        )

        matrix.index.name = f"{setting_a}_pred"
        matrix.columns.name = f"{setting_b}_pred"

        matrix.to_csv(
            self.output_dir
            / (
                f"06_transition_"
                f"{model}_"
                f"{setting_a}_to_{setting_b}.csv"
            ),
            encoding="utf-8-sig",
        )

        return matrix

    # =========================================================
    # 13. 所有转移矩阵
    # =========================================================

    def all_transition_matrices(self):
        result = {}

        for model in sorted(
            self.df["model"].unique()
        ):
            for setting_a, setting_b in self.COMPARISONS:
                matrix = self.transition_matrix(
                    model,
                    setting_a,
                    setting_b,
                )

                key = (
                    f"{model}_"
                    f"{setting_a}_"
                    f"{setting_b}"
                )
                result[key] = matrix

        return result

    # =========================================================
    # 14. 时间 / Token / Reasoning Token
    # =========================================================

    def efficiency_metrics(self):
        wanted = [
            "latency",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
        ]

        available = [
            x for x in wanted
            if x in self.df.columns
        ]

        if not available:
            return None

        rows = []

        for (model, setting), g in self.df.groupby(
            ["model", "setting"]
        ):
            row = {
                "model": model,
                "setting": setting,
                "n": len(g),
            }

            for col in available:
                values = pd.to_numeric(
                    g[col],
                    errors="coerce",
                )

                row[f"{col}_mean"] = values.mean()
                row[f"{col}_median"] = values.median()
                row[f"{col}_p95"] = values.quantile(0.95)
                row[f"{col}_sum"] = values.sum()

            if "reasoning_tokens" in available:
                reasoning_values = pd.to_numeric(
                    g["reasoning_tokens"],
                    errors="coerce",
                )
                row["reasoning_zero_n"] = int(
                    (reasoning_values == 0).sum()
                )
                row["reasoning_zero_rate"] = (
                    (reasoning_values == 0).mean()
                )

            rows.append(row)

        result = pd.DataFrame(rows)

        # 固定 S1-S6 排序
        result["_setting_order"] = result["setting"].map(
            {s: i for i, s in enumerate(self.SETTINGS)}
        )
        result = (
            result
            .sort_values(["model", "_setting_order"])
            .drop(columns=["_setting_order"])
            .reset_index(drop=True)
        )

        result.to_csv(
            self.output_dir / "07_efficiency_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        return result

    # =========================================================
    # 15. 论文总体结果表
    # =========================================================

    def overall_summary(self, basic_metrics):
        acc = basic_metrics.pivot(
            index="model",
            columns="setting",
            values="accuracy",
        ).reindex(columns=self.SETTINGS)
        acc.columns = [
            f"{x}_accuracy" for x in acc.columns
        ]

        macro = basic_metrics.pivot(
            index="model",
            columns="setting",
            values="macro_f1",
        ).reindex(columns=self.SETTINGS)
        macro.columns = [
            f"{x}_macro_f1" for x in macro.columns
        ]

        weighted = basic_metrics.pivot(
            index="model",
            columns="setting",
            values="weighted_f1",
        ).reindex(columns=self.SETTINGS)
        weighted.columns = [
            f"{x}_weighted_f1" for x in weighted.columns
        ]

        summary = (
            acc
            .join(macro)
            .join(weighted)
            .reset_index()
        )

        summary.to_csv(
            self.output_dir / "08_overall_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        return summary

    # =========================================================
    # 16. Accuracy
    # =========================================================

    def accuracy_table(self, basic_metrics):
        table = basic_metrics.pivot(
            index="model",
            columns="setting",
            values="accuracy",
        ).reindex(columns=self.SETTINGS)

        table = (table * 100).round(2)
        table = table.reset_index()

        table.to_csv(
            self.output_dir / "09_accuracy_table_percent.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return table

    # =========================================================
    # 17. Macro-F1 专用论文表
    # =========================================================

    def macro_f1_table(self, basic_metrics):
        table = basic_metrics.pivot(
            index="model",
            columns="setting",
            values="macro_f1",
        ).reindex(columns=self.SETTINGS)

        table = table.round(4).reset_index()

        table.to_csv(
            self.output_dir / "10_macro_f1_table.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return table

    # =========================================================
    # 18. Weighted-F1 专用论文表
    # =========================================================

    def weighted_f1_table(self, basic_metrics):
        table = basic_metrics.pivot(
            index="model",
            columns="setting",
            values="weighted_f1",
        ).reindex(columns=self.SETTINGS)

        table = table.round(4).reset_index()

        table.to_csv(
            self.output_dir / "11_weighted_f1_table.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return table

    # =========================================================
    # 19. 总运行入口
    # =========================================================

    def run_all(self):
        print("\n===== 1. Basic Metrics =====")
        basic = self.basic_metrics()
        print(basic.to_string(index=False))

        print("\n===== 2. Classwise Metrics =====")
        classwise = self.classwise_metrics()
        print(classwise.to_string(index=False))

        print("\n===== 3. Confusion Matrices =====")
        cms = self.confusion_matrices()

        if self.print_matrices:
            for name, matrices in cms.items():
                print(f"\n--- {name} | Count ---")
                print(
                    matrices["count"].to_string()
                )

                print(
                    f"\n--- {name} | Row-normalized ---"
                )
                print(
                    matrices["normalized"]
                    .round(4)
                    .to_string()
                )
        else:
            print(
                "混淆矩阵已保存为 CSV；"
                "print_matrices=False，控制台不展开。"
            )

        print("\n===== 4. Paired Comparisons =====")
        paired = self.all_paired_comparisons()
        print(paired.to_string(index=False))

        print("\n===== 5. Bootstrap 95% CI =====")
        bootstrap = self.all_bootstrap()
        print(bootstrap.to_string(index=False))

        print("\n===== 6. Transition Matrices =====")
        transitions = self.all_transition_matrices()

        if self.print_matrices:
            for name, matrix in transitions.items():
                print(f"\n--- {name} ---")
                print(matrix.to_string())
        else:
            print(
                "转移矩阵已保存为 CSV；"
                "print_matrices=False，控制台不展开。"
            )

        print("\n===== 7. Efficiency Metrics =====")
        efficiency = self.efficiency_metrics()
        if efficiency is not None:
            print(efficiency.to_string(index=False))
        else:
            print("结果文件中没有可用的时间/Token字段。")

        print("\n===== 8. Overall Summary =====")
        summary = self.overall_summary(basic)
        print(summary.to_string(index=False))

        print("\n===== 9. Accuracy Table (%) =====")
        accuracy_table = self.accuracy_table(basic)
        print(accuracy_table.to_string(index=False))

        print("\n===== 10. Macro-F1 Table =====")
        macro_table = self.macro_f1_table(basic)
        print(macro_table.to_string(index=False))

        print("\n===== 11. Weighted-F1 Table =====")
        weighted_table = self.weighted_f1_table(basic)
        print(weighted_table.to_string(index=False))

        print(
            "\n全部统计结果已输出至："
            f"{self.output_dir.resolve()}"
        )


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    # 如果文件放在项目 src/ 目录：
    #   ROOT_DIR = 项目根目录
    # 如果文件直接放项目根目录：
    #   ROOT_DIR = 当前目录
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = (
        SCRIPT_DIR.parent
        if SCRIPT_DIR.name.lower() == "src"
        else SCRIPT_DIR
    )

    RESULT_DIR = ROOT_DIR / "results"

    evaluator = ExperimentEvaluator(
        result_paths=[
            RESULT_DIR
            / "glm-5.3_S1_S2_S3_results.json",

            RESULT_DIR
            / "gpt-5.6_S1_S2_S3_results.json",

            RESULT_DIR
            / "qwen3.8-max_S1_S2_S3_results.json",

            RESULT_DIR
            / "glm-5.3_S4_S5_S6_results.json",

            RESULT_DIR
            / "gpt-5.6_S4_S5_S6_results.json",

            RESULT_DIR
            / "qwen3.8-max_S4_S5_S6_results.json",
        ],

        output_dir=(
            RESULT_DIR
            / "metrics"
            / "all_models_S1_S6"
        ),

        bootstrap_n=10000,
        random_seed=20260901,
        expected_n=400,

        # True：控制台打印全部 confusion / transition matrix
        # False：只保存 CSV，不在控制台展开
        print_matrices=True,
    )

    evaluator.run_all()
