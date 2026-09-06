# src/evaluator.py

import json
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
    """
    CCF 司法角色实验统计分析

    支持：
    1. Accuracy
    2. Macro-F1
    3. Weighted-F1
    4. 各类别 Precision / Recall / F1 / Support
    5. Confusion Matrix
    6. Prediction Flip Rate
    7. Harmful / Helpful Flip
    8. McNemar Exact Test
    9. Paired Bootstrap 95% CI
    10. S1/S2/S3 Prediction Transition Matrix
    11. Latency / Token / Reasoning Token 汇总
    """

    LABELS = ["A", "B", "C", "D"]
    SETTINGS = ["S1", "S2", "S3"]

    def __init__(
            self,
            result_paths,
            output_dir="results/metrics",
            bootstrap_n=10000,
            random_seed=20260901,
    ):
        self.result_paths = [Path(p) for p in result_paths]

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bootstrap_n = bootstrap_n
        self.random_seed = random_seed

        self.df = self._load_results()

        self._validate()

    # =========================================================
    # 1. 数据读取
    # =========================================================

    def _load_results(self):

        all_records = []

        for result_path in self.result_paths:

            suffix = result_path.suffix.lower()

            if suffix == ".json":

                with open(
                        result_path,
                        "r",
                        encoding="utf-8"
                ) as f:

                    data = json.load(f)

                if not isinstance(data, list):
                    raise ValueError(
                        f"{result_path} 最外层必须是 list"
                    )

                all_records.extend(data)

            elif suffix in [".jsonl", ".ndjson"]:

                with open(
                        result_path,
                        "r",
                        encoding="utf-8"
                ) as f:

                    for line in f:

                        line = line.strip()

                        if not line:
                            continue

                        all_records.append(
                            json.loads(line)
                        )

            else:
                raise ValueError(
                    f"不支持文件格式：{result_path}"
                )

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
                f"缺少必要字段：{missing}"
            )

        # 只保留有效预测
        invalid = ~self.df["pred"].isin(self.LABELS)

        if invalid.any():
            print(
                f"警告：发现 {invalid.sum()} 条无效预测，"
                f"将从统计中排除。"
            )

            self.df = self.df[
                ~invalid
            ].copy()

        # 检查重复
        dup = self.df.duplicated(
            subset=[
                "case_id",
                "model",
                "setting"
            ],
            keep=False
        )

        if dup.any():
            print(
                f"警告：发现 {dup.sum()} 条重复记录。"
            )

        print("\n===== 数据检查 =====")
        print(f"总记录数：{len(self.df)}")
        print(
            self.df.groupby(
                ["model", "setting"]
            ).size()
        )

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

                "correct": int(
                    (y_true == y_pred).sum()
                ),

                "accuracy": accuracy_score(
                    y_true,
                    y_pred
                ),

                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    labels=self.LABELS,
                    average="macro",
                    zero_division=0
                ),

                "weighted_f1": f1_score(
                    y_true,
                    y_pred,
                    labels=self.LABELS,
                    average="weighted",
                    zero_division=0
                ),
            })

        result = pd.DataFrame(rows)

        result.to_csv(
            self.output_dir /
            "01_basic_metrics.csv",
            index=False,
            encoding="utf-8-sig"
        )

        return result

    # =========================================================
    # 4. 各类别 P / R / F1
    # =========================================================

    def classwise_metrics(self):

        rows = []

        for (model, setting), g in self.df.groupby(
            ["model", "setting"]
        ):

            p, r, f1, support = (
                precision_recall_fscore_support(
                    g["gold"],
                    g["pred"],
                    labels=self.LABELS,
                    zero_division=0
                )
            )

            for i, label in enumerate(self.LABELS):

                rows.append({
                    "model": model,
                    "setting": setting,
                    "class": label,
                    "precision": p[i],
                    "recall": r[i],
                    "f1": f1[i],
                    "support": support[i],
                })

        result = pd.DataFrame(rows)

        result.to_csv(
            self.output_dir /
            "02_classwise_metrics.csv",
            index=False,
            encoding="utf-8-sig"
        )

        return result

    # =========================================================
    # 5. 混淆矩阵
    # =========================================================

    def confusion_matrices(self):

        result = {}

        for (model, setting), g in self.df.groupby(
            ["model", "setting"]
        ):

            cm = confusion_matrix(
                g["gold"],
                g["pred"],
                labels=self.LABELS
            )

            cm_df = pd.DataFrame(
                cm,
                index=[
                    f"Gold_{x}"
                    for x in self.LABELS
                ],
                columns=[
                    f"Pred_{x}"
                    for x in self.LABELS
                ]
            )

            filename = (
                f"03_confusion_"
                f"{model}_{setting}.csv"
            )

            cm_df.to_csv(
                self.output_dir / filename,
                encoding="utf-8-sig"
            )

            result[
                f"{model}_{setting}"
            ] = cm_df

        return result

    # =========================================================
    # 6. 把 S1/S2/S3 拉成同一行
    # =========================================================

    def _paired_predictions(self, model):

        g = self.df[
            self.df["model"] == model
        ].copy()

        pivot = g.pivot_table(
            index="case_id",
            columns="setting",
            values="pred",
            aggfunc="first"
        )

        gold = (
            g.groupby("case_id")["gold"]
            .first()
        )

        pivot["gold"] = gold

        return pivot.dropna()

    # =========================================================
    # 7. Flip + McNemar
    # =========================================================

    def paired_comparison(
        self,
        model,
        setting_a,
        setting_b
    ):

        p = self._paired_predictions(model)

        pred_a = p[setting_a]
        pred_b = p[setting_b]
        gold = p["gold"]

        correct_a = pred_a == gold
        correct_b = pred_b == gold

        # 预测本身是否变化
        prediction_flip = (
            pred_a != pred_b
        )

        # A正确 -> B错误
        harmful = (
            correct_a & ~correct_b
        )

        # A错误 -> B正确
        helpful = (
            ~correct_a & correct_b
        )

        n_harmful = int(harmful.sum())
        n_helpful = int(helpful.sum())

        discordant = (
            n_harmful + n_helpful
        )

        # Exact McNemar
        if discordant > 0:

            p_value = binomtest(
                min(
                    n_harmful,
                    n_helpful
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided"
            ).pvalue

        else:
            p_value = 1.0

        return {
            "model": model,
            "comparison":
                f"{setting_a}_vs_{setting_b}",

            "n": len(p),

            "accuracy_a":
                correct_a.mean(),

            "accuracy_b":
                correct_b.mean(),

            "delta_accuracy":
                correct_a.mean()
                - correct_b.mean(),

            "prediction_flip_n":
                int(prediction_flip.sum()),

            "prediction_flip_rate":
                prediction_flip.mean(),

            "harmful_flip_n":
                n_harmful,

            "harmful_flip_rate":
                harmful.mean(),

            "helpful_flip_n":
                n_helpful,

            "helpful_flip_rate":
                helpful.mean(),

            "mcnemar_p":
                p_value,
        }

    def all_paired_comparisons(self):

        comparisons = [
            ("S1", "S2"),
            ("S1", "S3"),
            ("S2", "S3"),
        ]

        rows = []

        for model in sorted(
            self.df["model"].unique()
        ):

            for a, b in comparisons:

                rows.append(
                    self.paired_comparison(
                        model,
                        a,
                        b
                    )
                )

        result = pd.DataFrame(rows)

        result.to_csv(
            self.output_dir /
            "04_paired_comparisons.csv",
            index=False,
            encoding="utf-8-sig"
        )

        return result

    # =========================================================
    # 8. Paired Bootstrap
    # =========================================================

    def bootstrap_difference(
        self,
        model,
        setting_a,
        setting_b
    ):

        p = self._paired_predictions(model)

        n = len(p)

        rng = np.random.default_rng(
            self.random_seed
        )

        gold = p["gold"].to_numpy()
        pred_a = p[setting_a].to_numpy()
        pred_b = p[setting_b].to_numpy()

        acc_diff = []
        macro_diff = []

        for _ in range(self.bootstrap_n):

            idx = rng.integers(
                0,
                n,
                size=n
            )

            y = gold[idx]
            a = pred_a[idx]
            b = pred_b[idx]

            acc_diff.append(
                accuracy_score(y, a)
                -
                accuracy_score(y, b)
            )

            macro_diff.append(
                f1_score(
                    y,
                    a,
                    labels=self.LABELS,
                    average="macro",
                    zero_division=0
                )
                -
                f1_score(
                    y,
                    b,
                    labels=self.LABELS,
                    average="macro",
                    zero_division=0
                )
            )

        acc_diff = np.array(acc_diff)
        macro_diff = np.array(macro_diff)

        return {
            "model": model,
            "comparison":
                f"{setting_a}_vs_{setting_b}",

            "delta_accuracy":
                accuracy_score(gold, pred_a)
                -
                accuracy_score(gold, pred_b),

            "acc_ci_low":
                np.percentile(
                    acc_diff,
                    2.5
                ),

            "acc_ci_high":
                np.percentile(
                    acc_diff,
                    97.5
                ),

            "delta_macro_f1":
                f1_score(
                    gold,
                    pred_a,
                    labels=self.LABELS,
                    average="macro",
                    zero_division=0
                )
                -
                f1_score(
                    gold,
                    pred_b,
                    labels=self.LABELS,
                    average="macro",
                    zero_division=0
                ),

            "macro_f1_ci_low":
                np.percentile(
                    macro_diff,
                    2.5
                ),

            "macro_f1_ci_high":
                np.percentile(
                    macro_diff,
                    97.5
                ),
        }

    def all_bootstrap(self):

        rows = []

        comparisons = [
            ("S1", "S2"),
            ("S1", "S3"),
            ("S2", "S3"),
        ]

        for model in sorted(
            self.df["model"].unique()
        ):

            for a, b in comparisons:

                rows.append(
                    self.bootstrap_difference(
                        model,
                        a,
                        b
                    )
                )

        result = pd.DataFrame(rows)

        result.to_csv(
            self.output_dir /
            "05_bootstrap_ci.csv",
            index=False,
            encoding="utf-8-sig"
        )

        return result

    # =========================================================
    # 9. 标签转移矩阵
    # =========================================================

    def transition_matrix(
        self,
        model,
        setting_a,
        setting_b
    ):

        p = self._paired_predictions(model)

        matrix = pd.crosstab(
            p[setting_a],
            p[setting_b],
            dropna=False
        )

        matrix = matrix.reindex(
            index=self.LABELS,
            columns=self.LABELS,
            fill_value=0
        )

        filename = (
            f"06_transition_"
            f"{model}_"
            f"{setting_a}_to_"
            f"{setting_b}.csv"
        )

        matrix.to_csv(
            self.output_dir / filename,
            encoding="utf-8-sig"
        )

        return matrix

    def all_transition_matrices(self):

        for model in sorted(
            self.df["model"].unique()
        ):

            self.transition_matrix(
                model,
                "S1",
                "S2"
            )

            self.transition_matrix(
                model,
                "S1",
                "S3"
            )

            self.transition_matrix(
                model,
                "S2",
                "S3"
            )

    # =========================================================
    # 10. 时间 / token
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
                    errors="coerce"
                )

                row[
                    f"{col}_mean"
                ] = values.mean()

                row[
                    f"{col}_median"
                ] = values.median()

                row[
                    f"{col}_p95"
                ] = values.quantile(0.95)

            rows.append(row)

        result = pd.DataFrame(rows)

        result.to_csv(
            self.output_dir /
            "07_efficiency_metrics.csv",
            index=False,
            encoding="utf-8-sig"
        )

        return result

    # =========================================================
    # 11. 总运行入口
    # =========================================================

    def run_all(self):

        print("\n===== 1. Basic Metrics =====")
        basic = self.basic_metrics()
        print(basic.to_string(index=False))

        print("\n===== 2. Classwise Metrics =====")
        classwise = self.classwise_metrics()

        print("\n===== 3. Confusion Matrices =====")
        self.confusion_matrices()

        print("\n===== 4. Paired Comparisons =====")
        paired = self.all_paired_comparisons()
        print(paired.to_string(index=False))

        print("\n===== 5. Bootstrap 95% CI =====")
        bootstrap = self.all_bootstrap()
        print(bootstrap.to_string(index=False))

        print("\n===== 6. Transition Matrices =====")
        self.all_transition_matrices()

        print("\n===== 7. Efficiency Metrics =====")
        efficiency = self.efficiency_metrics()

        if efficiency is not None:
            print(
                efficiency.to_string(
                    index=False
                )
            )

        print(
            f"\n全部统计结果已输出至："
            f"{self.output_dir.resolve()}"
        )
if __name__ == "__main__":
    evaluator = ExperimentEvaluator(
        result_paths=[
            "../results/glm-5.3_S1_S2_S3_results.json",
            "../results/gpt-5.6_S1_S2_S3_results.json",
            "../results/qwen3.8-max_S1_S2_S3_results.json"
        ],
        output_dir="../results/metrics/all_models",
        bootstrap_n=10000,
        random_seed=20260901
    )

    evaluator.run_all()
