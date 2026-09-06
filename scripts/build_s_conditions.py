"""500 案近重复审计、冻结样本及 S1/S2/S3 构造；不调用模型或推断 gold。"""
import argparse
import hashlib
import itertools
import json
import platform
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("P", "D", "F", "R")
LABELS = {"A", "B", "C", "D"}
FILENAMES = {
    "S1": "ccf_400_S1_correct_roles.json", "S2": "ccf_400_S2_no_roles.json",
    "S3": "ccf_400_S3_mismatched_roles.json", "gold": "ccf_400_verified_gold.json",
}


def read_records(path):
    records = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"输入必须是非空对象数组：{path}")
    return records


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fields_hash(record):
    return content_hash({field: record[field] for field in FIELDS})


def index_records(records, key="id"):
    result = {}
    for record in records:
        value = record.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} 必须是正整数：{value!r}")
        if value in result:
            raise ValueError(f"重复的 {key}：{value}")
        result[value] = record
    return result


def validate_fields(records):
    for record in records:
        for field in FIELDS:
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"样本 {record.get('id')} 缺少非空文本字段 {field}")
        if not isinstance(record.get("Category"), dict):
            raise ValueError(f"样本 {record.get('id')} 缺少 Category 对象")


def normalize_for_dedup(record):
    """仅用于相似度，不修改模型输入；连续数字统一为 <NUM>。"""
    text = unicodedata.normalize("NFKC", "".join(record[field] for field in FIELDS))
    text = re.sub(r"\d+", "<NUM>", text)
    return re.sub(r"\s+", "", text)


def cluster_similarities(ids, similarities, threshold):
    """阈值图的连通分量；传递连接不要求簇内两两达到阈值。"""
    parents = list(range(len(ids)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            score = float(similarities[i, j])
            if score >= threshold:
                a, b = find(i), find(j)
                parents[b] = a
                edges.append({"id_a": ids[i], "id_b": ids[j], "similarity": score})
    components = defaultdict(list)
    for i, record_id in enumerate(ids):
        components[find(i)].append(record_id)
    groups = sorted((sorted(g) for g in components.values()), key=lambda g: g[0])
    return groups, edges


def deduplicate(records, threshold=0.75):
    if not 0 < threshold <= 1:
        raise ValueError("相似度阈值必须位于 (0, 1]")
    index_records(records)
    validate_fields(records)
    texts = [normalize_for_dedup(r) for r in records]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=True,
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
        min_df=1, max_df=1.0, dtype=np.float64,
    )
    similarities = cosine_similarity(vectorizer.fit_transform(texts))
    groups, edges = cluster_similarities([r["id"] for r in records], similarities, threshold)
    keep_ids = {g[0] for g in groups}
    kept = [r.copy() for r in records if r["id"] in keep_ids]
    clusters = [
        {"cluster_id": n, "kept_id": g[0], "member_ids": g, "removed_ids": g[1:]}
        for n, g in enumerate((g for g in groups if len(g) > 1), start=1)
    ]
    removed = [
        {"original_id": i, "kept_id": c["kept_id"], "cluster_id": c["cluster_id"],
         "reason": "near_duplicate_component"}
        for c in clusters for i in c["removed_ids"]
    ]
    report = {
        "input_count": len(records), "duplicate_cluster_count": len(clusters),
        "removed_count": len(removed), "retained_count": len(kept),
        "threshold_edge_count": len(edges),
        "normalization": {"fields": list(FIELDS), "join_separator": "", "unicode": "NFKC",
                          "digit_pattern": r"\d+", "digit_replacement": "<NUM>", "whitespace": "remove"},
        "tfidf": {"analyzer": "char", "ngram_range": [3, 5], "lowercase": True,
                  "norm": "l2", "use_idf": True, "smooth_idf": True, "sublinear_tf": False,
                  "min_df": 1, "max_df": 1.0, "dtype": "float64", "fit_scope": "all_input_records"},
        "threshold": threshold, "comparison": ">=", "clustering": "connected_components",
        "representative": "minimum_original_id", "output_order": "input_order",
        "retained_original_ids": [r["id"] for r in kept],
        "retained_content_sha256": content_hash(kept), "normalized_texts_sha256": content_hash(texts),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                     "unicode": unicodedata.unidata_version},
        "qualification": "Near-duplicate screening does not establish legal case independence.",
    }
    return kept, clusters, removed, edges, report


def prepare_role_texts(source, s2, s3):
    """恢复已有中性文本和置换关系，不重建历史改写算法或执行人工核验。"""
    source_index, s2_index, s3_index = map(index_records, (source, s2, s3))
    index_records(source, "original_id")
    if source_index.keys() != s2_index.keys() or source_index.keys() != s3_index.keys():
        raise ValueError("来源数据与 S2/S3 的案件编号集合不一致")
    validate_fields(source)
    validate_fields(s3)
    annotations = []
    for case_id, record in source_index.items():
        condition2, condition3 = s2_index[case_id], s3_index[case_id]
        if condition3["R"] != record["R"]:
            raise ValueError(f"案件 {case_id} 的 S3 法律规则与来源不一致")
        if any(c.get("Category") != record["Category"] for c in (condition2, condition3)):
            raise ValueError(f"案件 {case_id} 的 Category 不一致")
        matches = [
            order for order in itertools.permutations("PDF")
            if condition2.get("text") == "\n\n".join(
                [condition3[f] for f in order] + [condition3["R"]])
        ]
        if len(matches) != 1:
            raise ValueError(f"案件 {case_id} 无法唯一恢复中性段落与置换关系")
        order = matches[0]
        assignment = {output: field for field, output in zip("PDF", order)}
        if any(output == field for output, field in assignment.items()):
            raise ValueError(f"案件 {case_id} 的置换含未错配字段")
        annotations.append({
            "original_id": record["original_id"], "source_fields_sha256": fields_hash(record),
            "neutral": {field: condition3[output] for field, output in zip("PDF", order)},
            "s3_assignment": assignment,
            "provenance": "Recovered from existing S2/S3 artifacts; no new text rewriting.",
        })
    return annotations


def build_conditions(kept, verified_source, annotations):
    index_records(kept)
    validate_fields(kept)
    verified = index_records(verified_source, "original_id")
    roles = index_records(annotations, "original_id")
    ids = {r["id"] for r in kept}
    if ids != verified.keys() or ids != roles.keys():
        raise ValueError("保留样本、核验来源和角色文本的 original_id 集合必须完全一致")
    outputs = {name: [] for name in FILENAMES}
    frozen_source = []
    for new_id, record in enumerate(kept, start=1):
        original_id = record["id"]
        checked, role = verified[original_id], roles[original_id]
        if any(record[f] != checked.get(f) for f in (*FIELDS, "Category")):
            raise ValueError(f"样本 {original_id} 与核验来源的材料不一致")
        if role.get("source_fields_sha256") != fields_hash(record):
            raise ValueError(f"样本 {original_id} 的角色标注来源哈希不一致")
        if checked.get("gold") not in LABELS:
            raise ValueError(f"样本 {original_id} 缺少有效 gold；不能自动推断标准答案")
        neutral = role.get("neutral", {})
        if any(not isinstance(neutral.get(f), str) or not neutral[f].strip() for f in "PDF"):
            raise ValueError(f"样本 {original_id} 缺少中性文本")
        assignment = role.get("s3_assignment", {})
        if (set(assignment) != set("PDF") or set(assignment.values()) != set("PDF")
                or any(k == v for k, v in assignment.items())):
            raise ValueError(f"样本 {original_id} 的 S3 必须为 P/D/F 完全错配置换")
        base = {"id": new_id, "Category": record["Category"]}
        s1 = {**base, **{f: record[f] for f in FIELDS}}
        outputs["S1"].append(s1)
        outputs["S2"].append({**base, "text": "\n\n".join([neutral[f] for f in "PDF"] + [record["R"]])})
        outputs["S3"].append({**base, **{f: neutral[assignment[f]] for f in "PDF"}, "R": record["R"]})
        outputs["gold"].append({**s1, "gold": checked["gold"]})
        frozen_source.append({**checked, "id": new_id, "original_id": original_id})
    return outputs, frozen_source


def new_output_dir(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"输出目录必须不存在或为空，以保留已有实验：{path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_build(args):
    records = read_records(args.input)
    print(f"正在对 {len(records)} 份文书构建 TF-IDF 相似度图……", flush=True)
    kept, clusters, removed, edges, report = deduplicate(records, args.threshold)
    report["input"] = {"path": str(args.input.resolve()), "sha256": file_hash(args.input)}
    expectations = {"input_count": args.expected_input_count, "retained_count": args.expected_output_count,
                    "duplicate_cluster_count": args.expected_cluster_count}
    report["expected_counts"] = expectations
    report["counts_match"] = all(report[k] == v for k, v in expectations.items())
    out = new_output_dir(args.output_dir)
    for name, value in [("dedup_report", report), ("duplicate_clusters", clusters),
                        ("removed_cases", removed), ("similarity_edges", edges),
                        ("deduplicated_candidates", kept)]:
        write_json(out / f"{name}.json", value)
    print(f"重复簇 {len(clusters)}；剔除 {len(removed)}；保留 {len(kept)}", flush=True)
    if not report["counts_match"]:
        raise ValueError(f"实测数量与预期不符；审计已保存至 {out}，未生成条件。不会强行删样。")
    if args.dedup_only:
        return
    outputs, frozen = build_conditions(kept, read_records(args.verified_source), read_records(args.role_texts))
    source_dir, raw_dir = out / "source", out / "raw"
    source_dir.mkdir()
    raw_dir.mkdir()
    # 先冻结统一样本与标签，再写三个条件。
    write_json(source_dir / "ccf_400_full_source.json", frozen)
    for condition, filename in FILENAMES.items():
        write_json(raw_dir / filename, outputs[condition])
    manifest = {
        "sample_count": len(frozen), "frozen_source_sha256": content_hash(frozen),
        "verified_source": {"path": str(args.verified_source.resolve()), "sha256": file_hash(args.verified_source)},
        "role_texts": {"path": str(args.role_texts.resolve()), "sha256": file_hash(args.role_texts)},
        "gold_distribution": dict(Counter(r["gold"] for r in frozen)),
        "conditions": {"S1": "Original P/D/F with correct headings; original R.",
                       "S2": "Annotated neutral P/D/F, then R, joined by two newlines.",
                       "S3": "Same neutral blocks as S2; per-case frozen derangement; R unchanged."},
        "outputs": {filename: file_hash(raw_dir / filename) for filename in FILENAMES.values()},
        "manual_review_note": "Uses existing gold and role annotations; does not perform human review.",
    }
    write_json(out / "manifest.json", manifest)
    print(f"冻结样本及 S1/S2/S3 已生成：{out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="对 500 案去重并生成正式条件")
    build.add_argument("--input", type=Path, default=ROOT / "data/source/ccf_500_candidates.json")
    build.add_argument("--verified-source", type=Path, default=ROOT / "data/source/ccf_400_full_source.json")
    build.add_argument("--role-texts", type=Path, default=ROOT / "data/source/ccf_400_role_texts.json")
    build.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/role_experiment")
    build.add_argument("--threshold", type=float, default=0.75)
    build.add_argument("--expected-input-count", type=int, default=500)
    build.add_argument("--expected-output-count", type=int, default=400)
    build.add_argument("--expected-cluster-count", type=int, default=38)
    build.add_argument("--dedup-only", action="store_true")
    prepare = commands.add_parser("prepare-role-texts", help="从已有 S2/S3 恢复中性文本及逐案置换")
    prepare.add_argument("--source", type=Path, default=ROOT / "data/source/ccf_400_full_source.json")
    prepare.add_argument("--s2", type=Path, default=ROOT / "data/raw" / FILENAMES["S2"])
    prepare.add_argument("--s3", type=Path, default=ROOT / "data/raw" / FILENAMES["S3"])
    prepare.add_argument("--output", type=Path, default=ROOT / "data/source/ccf_400_role_texts.json")
    args = parser.parse_args()
    try:
        if args.command == "build":
            run_build(args)
        else:
            if args.output.exists():
                raise ValueError(f"输出文件已存在：{args.output}")
            annotations = prepare_role_texts(read_records(args.source), read_records(args.s2), read_records(args.s3))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, annotations)
            print(f"已恢复 {len(annotations)} 条角色文本标注：{args.output}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
