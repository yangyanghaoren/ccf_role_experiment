import argparse
import json
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "labor_candidate_pool" / "candidate_clear.json"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "legacy_s_conditions"
SEED = 42
REMOVE_COUNT = 143


BASE_FIELDS = [
    "id",
    "Category",
]

CONDITIONS = {
    "S1": ["P"],
    "S2": ["P", "D"],
    "S3": ["P", "F"],
    "S4": ["P", "D", "F"],
    "S5": ["P", "R"],
    "S6": ["P", "D", "R"],
    "S7": ["P", "D", "F", "R"],
}


def select_fields(record, condition_fields, record_id):
    output = {
        "id": record_id,
    }

    for field in BASE_FIELDS[1:] + condition_fields:
        output[field] = record.get(field)

    return output


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def remove_same_random_cases(records):
    if REMOVE_COUNT > len(records):
        raise ValueError(
            f"REMOVE_COUNT={REMOVE_COUNT} 大于样本总数 {len(records)}"
        )

    rng = random.Random(SEED)
    removed_indices = set(rng.sample(range(len(records)), REMOVE_COUNT))

    kept_records = [
        record
        for index, record in enumerate(records)
        if index not in removed_indices
    ]

    removed_records = [
        {
            "CaseId": record.get("CaseId"),
            "Case": record.get("Case"),
        }
        for index, record in enumerate(records)
        if index in removed_indices
    ]

    return kept_records, removed_records


def main():
    global INPUT_PATH, OUTPUT_DIR, SEED, REMOVE_COUNT
    parser = argparse.ArgumentParser(description="生成旧版 S1–S7 字段组合条件，不是正式角色实验 S1–S3。")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--remove-count", type=int, default=REMOVE_COUNT)
    args = parser.parse_args()
    INPUT_PATH, OUTPUT_DIR = args.input, args.output_dir
    SEED, REMOVE_COUNT = args.seed, args.remove_count
    if not INPUT_PATH.is_file():
        parser.error(f"找不到输入文件：{INPUT_PATH}")
    if REMOVE_COUNT < 0:
        parser.error("--remove-count 不能为负数")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kept_records, removed_records = remove_same_random_cases(records)

    manifest = {
        "input": str(INPUT_PATH),
        "source_sample_size": len(records),
        "seed": SEED,
        "removed_count": len(removed_records),
        "final_sample_size": len(kept_records),
        "removed_cases_output": str(OUTPUT_DIR / "removed_cases.json"),
        "base_fields": BASE_FIELDS,
        "conditions": {},
    }

    for condition, fields in CONDITIONS.items():
        output_records = [
            select_fields(record, fields, record_id)
            for record_id, record in enumerate(kept_records, start=1)
        ]

        output_path = OUTPUT_DIR / f"{condition}.json"
        write_json(output_path, output_records)

        manifest["conditions"][condition] = {
            "fields": BASE_FIELDS + fields,
            "output": str(output_path),
            "sample_size": len(output_records),
        }

    write_json(OUTPUT_DIR / "removed_cases.json", removed_records)

    manifest_path = OUTPUT_DIR / "manifest.json"
    write_json(manifest_path, manifest)

    print("=" * 70)
    print("S1-S7 实验条件文件生成完成")
    print("=" * 70)
    print(f"输入: {INPUT_PATH.resolve()}")
    print(f"原始样本数: {len(records)}")
    print(f"随机删除数: {len(removed_records)}")
    print(f"最终样本数: {len(kept_records)}")
    print(f"随机种子: {SEED}")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")
    for condition in CONDITIONS:
        print(f"{condition}: {(OUTPUT_DIR / f'{condition}.json').resolve()}")
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
