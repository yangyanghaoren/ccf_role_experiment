"""语义测试与现有 500→400 数据回归；不调用 API。"""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts import build_s_conditions as pipeline


def sample(record_id=1):
    return {"id": record_id, "Category": {"cat_1": "劳动人事", "cat_2": "劳动合同"},
            "P": "原告诉请工资100元", "D": "被告认为已支付工资", "F": "法院查明劳动关系", "R": "劳动合同法"}


class RoleConditionsTests(unittest.TestCase):
    def test_normalization_and_preservation(self):
        record = sample()
        record.update(P="ＡＢＣ １２３\n", D="工资456", F="事实\t", R="规则")
        before = copy.deepcopy(record)
        self.assertEqual(pipeline.normalize_for_dedup(record), "ABC<NUM>工资<NUM>事实规则")
        self.assertEqual(record, before)

    def test_transitive_components_include_exact_threshold_and_keep_min_id(self):
        scores = np.array([[1, .75, .1, .0], [.75, 1, .8, .0], [.1, .8, 1, .0], [.0, .0, .0, 1]])
        groups, edges = pipeline.cluster_similarities([9, 3, 7, 20], scores, .75)
        self.assertEqual(groups, [[3, 7, 9], [20]])
        self.assertEqual(len(edges), 2)

    def test_incomplete_and_duplicate_inputs_fail(self):
        missing = sample()
        missing["R"] = None
        with self.assertRaisesRegex(ValueError, "R"):
            pipeline.deduplicate([missing])
        with self.assertRaisesRegex(ValueError, "重复"):
            pipeline.deduplicate([sample(), sample()])

    def test_min_id_kept_without_changing_input(self):
        records = [sample(8), sample(2)]
        before = copy.deepcopy(records)
        kept, clusters, removed, _, _ = pipeline.deduplicate(records)
        self.assertEqual([r["id"] for r in kept], [2])
        self.assertEqual(clusters[0]["member_ids"], [2, 8])
        self.assertEqual(removed[0]["original_id"], 8)
        self.assertEqual(records, before)

    def test_build_validation_and_no_gold_in_conditions(self):
        record = sample()
        verified = [{**record, "original_id": 1, "gold": "B"}]
        annotation = {"original_id": 1, "source_fields_sha256": pipeline.fields_hash(record),
                      "neutral": {"P": "诉请工资", "D": "已经支付", "F": "存在劳动关系"},
                      "s3_assignment": {"P": "F", "D": "P", "F": "D"}}
        outputs, frozen = pipeline.build_conditions([record], verified, [annotation])
        self.assertEqual(outputs["S3"][0]["P"], "存在劳动关系")
        self.assertEqual(outputs["S2"][0]["text"], "诉请工资\n\n已经支付\n\n存在劳动关系\n\n劳动合同法")
        self.assertEqual(frozen[0]["gold"], "B")
        for setting in ["S1", "S2", "S3"]:
            self.assertNotIn("gold", outputs[setting][0])
            self.assertNotIn("JudgeResult", outputs[setting][0])
        bad = {**annotation, "source_fields_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "哈希"):
            pipeline.build_conditions([record], verified, [bad])
        with self.assertRaisesRegex(ValueError, "gold"):
            pipeline.build_conditions([record], [{**verified[0], "gold": None}], [annotation])

    def test_count_mismatch_writes_audit_but_no_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            pipeline.write_json(source, [sample(1), sample(2)])
            args = Namespace(input=source, threshold=.75, output_dir=root / "out",
                             expected_input_count=2, expected_output_count=400,
                             expected_cluster_count=38, dedup_only=False)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "实测数量"):
                pipeline.run_build(args)
            self.assertTrue((args.output_dir / "dedup_report.json").exists())
            self.assertFalse((args.output_dir / "raw").exists())
            with self.assertRaisesRegex(ValueError, "输出目录"):
                pipeline.new_output_dir(args.output_dir)

    def test_full_regression_against_saved_400_cases(self):
        root = pipeline.ROOT
        records = pipeline.read_records(root / "data/source/ccf_500_candidates.json")
        kept, clusters, removed, _, report = pipeline.deduplicate(records)
        self.assertEqual((len(clusters), len(removed), len(kept)), (38, 100, 400))
        verified = pipeline.read_records(root / "data/source/ccf_400_full_source.json")
        self.assertEqual([r["id"] for r in kept], [r["original_id"] for r in verified])
        annotations = pipeline.read_records(root / "data/source/ccf_400_role_texts.json")
        outputs, frozen = pipeline.build_conditions(kept, verified, annotations)
        self.assertEqual(frozen, verified)
        for setting, filename in pipeline.FILENAMES.items():
            self.assertEqual(outputs[setting], pipeline.read_records(root / "data/raw" / filename))
        self.assertEqual(report["normalization"]["digit_replacement"], "<NUM>")
        recovered = pipeline.prepare_role_texts(verified, outputs["S2"], outputs["S3"])
        self.assertEqual(recovered, annotations)


if __name__ == "__main__":
    unittest.main()
