import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# =========================================================
# 配置
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "data" / "train"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "labor_candidate_pool"

TARGET_CAT2 = {
    "工资福利",
    "劳动合同",
    "劳动争议",
}

TARGET_DOCUMENT_TYPES = {
    "判决书",
    "裁定书",
}

EXCLUDE_700_TRUNCATED = True


# =========================================================
# 基础工具
# =========================================================

def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(text):
    text = safe_text(text)
    return re.sub(r"\s+", "", text)


def content_hash(case):
    normalized_text = (
        normalize_text(case.get("JudgeAccusation"))
        + normalize_text(case.get("JudgeReason"))
        + normalize_text(case.get("JudgeResult"))
    )
    return hashlib.md5(normalized_text.encode("utf-8")).hexdigest()


def get_document_type(case_title):
    title = safe_text(case_title)

    if "判决书" in title:
        return "判决书"
    if "裁定书" in title:
        return "裁定书"
    if "调解书" in title:
        return "调解书"
    if "决定书" in title:
        return "决定书"

    return "其他"


def iter_json_cases(input_dir, scan_stats=None):
    files = sorted(input_dir.rglob("*.json"))
    if scan_stats is not None:
        scan_stats["total_source_files"] = len(files)

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if scan_stats is not None:
                scan_stats["success_files"] += 1
        except Exception as exc:
            if scan_stats is not None:
                scan_stats["failed_files"] += 1
            yield {
                "file_path": file_path,
                "load_error": repr(exc),
                "data": None,
                "ctx_id": None,
                "case": None,
            }
            continue

        ctxs = data.get("ctxs")
        if not isinstance(ctxs, dict):
            continue

        for ctx_id, case in ctxs.items():
            if not isinstance(case, dict):
                continue

            yield {
                "file_path": file_path,
                "load_error": None,
                "data": data,
                "ctx_id": ctx_id,
                "case": case,
            }


def first_match_start(patterns, text):
    starts = []

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            starts.append(m.start())

    return min(starts) if starts else None


def earliest_non_none(values):
    values = [v for v in values if v is not None and v >= 0]
    return min(values) if values else None


def make_base_record(file_path, input_dir, data, ctx_id, case, chash=None):
    category = case.get("Category") or {}
    if not isinstance(category, dict):
        category = {}

    record = {
        "source_file": str(file_path.relative_to(input_dir)),
        "source_q_id": data.get("q_id") if isinstance(data, dict) else None,
        "source_ctx_id": str(ctx_id),
        "CaseId": safe_text(case.get("CaseId")),
        "Case": safe_text(case.get("Case")),
        "CaseProc": safe_text(case.get("CaseProc")),
        "CaseType": safe_text(case.get("CaseType")),
        "Category": {
            "cat_1": safe_text(category.get("cat_1")),
            "cat_2": safe_text(category.get("cat_2")),
        },
        "document_type": get_document_type(case.get("Case")),
        "JudgeAccusation": safe_text(case.get("JudgeAccusation")),
        "JudgeReason": safe_text(case.get("JudgeReason")),
        "JudgeResult": safe_text(case.get("JudgeResult")),
        "content_hash": chash or content_hash(case),
        "qc_status": "REVIEW",
        "qc_flags": [],
    }

    return record


# =========================================================
# P/D/F 边界模式
# =========================================================

PLAINTIFF_START_PATTERNS = [
    r"原告[^。\n]{0,80}?诉称[：:，,]?",
    r"原告[^。\n]{0,80}?向本院提出(?:如下)?诉讼请求[：:，,]?",
    r"原告[^。\n]{0,80}?提出(?:如下)?诉讼请求[：:，,]?",
    r"原告[^。\n]{0,80}?请求(?:依法)?判令[：:，,]?",
    r"原告[^。\n]{0,80}?诉请[：:，,]?",
    r"[^。\n]{1,60}?向本院提出(?:如下)?诉讼请求[：:，,]?",
]

DEFENSE_START_PATTERNS = [
    r"(?:被告|二被告|两被告|三被告|各被告)[^。\n]{0,100}?辩称[：:，,]?",
    r"(?:被告|二被告|两被告|三被告|各被告)[^。\n]{0,100}?答辩称[：:，,]?",
    r"(?:被告|二被告|两被告|三被告|各被告)[^。\n]{0,100}?答辩如下[：:，,]?",
    r"(?:被告|二被告|两被告|三被告|各被告)[^。\n]{0,100}?答辩意见如下[：:，,]?",
    r"(?:被告|二被告|两被告|三被告|各被告)[^。\n]{0,100}?答辩意见为[：:，,]?",
]

CAUTIOUS_DEFENSE_PATTERNS = [
    r"(?:被告|二被告|两被告|三被告|各被告)[^。\n]{0,80}?称[：:]",
]

NO_DEFENSE_PATTERNS = [
    r"未作答辩",
    r"未做答辩",
    r"没有答辩",
    r"未答辩",
    r"未提交书面答辩状",
    r"未提交书面答辩",
    r"未提交答辩状",
    r"未提交答辩意见",
    r"未提交书面答辩意见",
    r"未向本院提交书面答辩",
    r"未向本院提交答辩意见",
    r"未提供书面答辩意见",
    r"亦未提交书面答辩",
    r"也未提交书面答辩",
    r"均未提交书面答辩",
]

EVIDENCE_STAGE_PATTERNS = [
    r"原告[^。\n]{0,60}?为证明",
    r"原告[^。\n]{0,60}?向本院提交",
    r"被告[^。\n]{0,60}?为证明",
    r"被告[^。\n]{0,60}?向本院提交",
    r"当事人围绕[^。\n]{0,80}?提交了证据",
    r"当事人围绕[^。\n]{0,80}?提供了证据",
    r"(?:原、被告|原被告|双方|当事人)[^。\n]{0,80}?(?:提交|提供)了?证据",
    r"(?:原、被告|原被告|双方)[^。\n]{0,80}?未[^。\n]{0,40}?提供证据",
    r"本院组织[^。\n]{0,80}?质证",
    r"经庭审质证",
]

FACT_START_PATTERNS = [
    r"经审理查明[：:]?",
    r"经本院审理查明[：:]?",
    r"本院经审理查明[：:]?",
    r"本院经审理认定事实如下[：:]?",
    r"本院认定事实如下[：:]?",
    r"本院认定如下事实[：:]?",
    r"本院查明[：:]?",
    r"本院审理查明[：:]?",
    r"经审理认定[：:]?",
    r"本院经审理认定如下事实[：:]?",
    r"本院审理后查明[：:]?",
    r"经庭审查明[：:]?",
    r"本院确认以下事实[：:]?",
    r"本院确认如下事实[：:]?",
    r"本院结合庭审[^。\n]{0,80}?认定如下事实[：:]?",
    r"根据当事人陈述及[^。\n]{0,80}?证据[^。\n]{0,80}?本院确认[^。\n]{0,20}?事实[：:]?",
]

REASON_START_PATTERNS = [
    r"本院认为",
    r"本院经审理认为",
    r"法院认为",
    r"本院认为本案争议焦点为",
    r"综上，本院认为",
    r"综上所述",
    r"依照",
    r"判决如下",
    r"裁定如下",
]

FACT_TRAILING_EVIDENCE_PATTERNS = [
    r"上述事实[，,]?有",
    r"以上事实[，,]?有",
    r"上述事实有",
    r"以上事实有",
]

LEAKAGE_PATTERNS = [
    r"判决如下",
    r"裁定如下",
    r"本院予以支持",
    r"本院不予支持",
    r"本院不予采信",
    r"原告诉请成立",
    r"驳回原告",
    r"应当支付",
    r"因此被告",
    r"故被告",
]


# =========================================================
# P 提取与 claim_count
# =========================================================

def extract_plaintiff_text(judge_accusation):
    text = safe_text(judge_accusation)

    if not text:
        return "", "REVIEW", ["PLAINTIFF_EMPTY"]

    start = None
    matched = None

    for pattern in PLAINTIFF_START_PATTERNS:
        m = re.search(pattern, text)
        if m and (start is None or m.start() < start):
            start = m.start()
            matched = m

    if matched is None:
        return "", "REVIEW", ["PLAINTIFF_BOUNDARY_UNCERTAIN"]

    d_start = first_match_start(DEFENSE_START_PATTERNS, text[matched.end():])
    if d_start is not None:
        d_start += matched.end()

    evidence_start = first_match_start(EVIDENCE_STAGE_PATTERNS, text[matched.end():])
    if evidence_start is not None:
        evidence_start += matched.end()

    fact_start = first_match_start(FACT_START_PATTERNS, text[matched.end():])
    if fact_start is not None:
        fact_start += matched.end()

    end = earliest_non_none([d_start, evidence_start, fact_start])
    if end is None:
        end = len(text)

    plaintiff_text = text[start:end].strip()
    flags = []

    if not plaintiff_text:
        flags.append("PLAINTIFF_EMPTY")

    if any(re.search(p, plaintiff_text) for p in DEFENSE_START_PATTERNS):
        flags.append("PLAINTIFF_CONTAINS_DEFENSE")

    if any(re.search(p, plaintiff_text) for p in FACT_START_PATTERNS):
        flags.append("PLAINTIFF_CONTAINS_COURT_FACT")

    status = "CLEAR" if plaintiff_text and not flags else "REVIEW"
    return plaintiff_text, status, flags


CHINESE_NUMERAL_MAP = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def chinese_numeral_to_int(text):
    text = safe_text(text)

    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in CHINESE_NUMERAL_MAP:
        return CHINESE_NUMERAL_MAP[text]
    if text.startswith("十") and len(text) == 2:
        tail = CHINESE_NUMERAL_MAP.get(text[1])
        return 10 + tail if tail else None
    if text.endswith("十") and len(text) == 2:
        head = CHINESE_NUMERAL_MAP.get(text[0])
        return head * 10 if head else None
    if len(text) == 3 and text[1] == "十":
        head = CHINESE_NUMERAL_MAP.get(text[0])
        tail = CHINESE_NUMERAL_MAP.get(text[2])
        if head and tail:
            return head * 10 + tail
    return None


def estimate_claim_count_from_p(plaintiff_text):
    text = safe_text(plaintiff_text)

    if not text:
        return None, "REVIEW", ["CLAIM_COUNT_UNCERTAIN"]

    start_markers = [
        "诉讼请求",
        "请求判令",
        "请求人民法院判令",
        "请求依法判令",
        "要求判令",
        "诉请",
    ]

    positions = [
        text.find(marker)
        for marker in start_markers
        if text.find(marker) != -1
    ]

    if not positions:
        return None, "REVIEW", ["CLAIM_COUNT_UNCERTAIN"]

    start = min(positions)
    claim_text = text[start:]

    reason_markers = [
        "事实与理由",
        "事实和理由",
        "事实及理由",
        "事实理由",
        "理由如下",
    ]

    reason_positions = [
        claim_text.find(marker)
        for marker in reason_markers
        if claim_text.find(marker) != -1
    ]

    if reason_positions:
        claim_text = claim_text[:min(reason_positions)]

    raw_nums = re.findall(
        r"(?:^|[：:；;。\n])\s*([1-9]\d*|[一二三四五六七八九十]{1,3})[、.．）)]",
        claim_text,
    )

    nums = []
    suspicious = []

    for raw in raw_nums:
        value = chinese_numeral_to_int(raw)
        if value is None:
            continue
        if 1 <= value <= 20:
            nums.append(value)
        else:
            suspicious.append(value)

    if nums:
        if 1 not in nums:
            return None, "REVIEW", ["CLAIM_COUNT_UNCERTAIN"]
        return max(nums), "CLEAR", []

    flags = []
    if suspicious:
        flags.append("CLAIM_COUNT_UNCERTAIN")

    # 没有编号但存在明确请求句时，按一个请求处理。
    if any(marker in claim_text for marker in ["请求", "诉请", "判令"]):
        return 1, "CLEAR" if not flags else "REVIEW", flags

    return None, "REVIEW", ["CLAIM_COUNT_UNCERTAIN"]


# =========================================================
# D 提取与分类
# =========================================================

MINIMAL_DEFENSE_PATTERNS = [
    r"^不同意原告",
    r"^不同意.*诉讼请求",
    r"^不认可",
    r"^不予认可",
    r"^不应支持",
    r"^不予支持",
    r"^请求驳回",
    r"^无异议",
    r"^没有异议",
    r"^无意见",
    r"^没有意见",
    r"^以法院判决为准",
    r"^由法院依法判决",
    r"^依法判决",
]

SUBSTANTIVE_SIGNALS = [
    "劳动关系",
    "劳务关系",
    "劳动合同",
    "劳动报酬",
    "工资",
    "加班费",
    "已支付",
    "已经支付",
    "没有拖欠",
    "已结清",
    "已经结清",
    "离职",
    "辞职",
    "解除",
    "终止",
    "经济补偿",
    "赔偿",
    "社会保险",
    "社保",
    "仲裁时效",
    "诉讼时效",
    "主体不适格",
    "管辖",
    "仲裁前置",
    "证据",
    "真实性",
    "举证",
    "退休",
    "法律依据",
    "金额",
    "数额",
]

SUBSTANTIVE_DEFENSE_PATTERNS = [
    r"不存在[^。；;，,]{0,30}?劳动关系",
    r"不属于[^。；;，,]{0,30}?劳动关系",
    r"劳务关系[^。；;，,]{0,30}?劳动关系",
    r"(?:已经|已|均已)[^。；;，,]{0,20}?支付",
    r"没有拖欠",
    r"不拖欠",
    r"(?:已经|已)[^。；;，,]{0,20}?结清",
    r"(?:超过|已过)[^。；;，,]{0,30}?时效",
    r"仲裁时效",
    r"诉讼时效",
    r"未经[^。；;，,]{0,30}?仲裁",
    r"仲裁[^。；;，,]{0,20}?前置",
    r"主体不适格",
    r"无管辖权",
    r"不认可[^。；;，,]{0,30}?证据",
    r"证据[^。；;，,]{0,30}?真实性",
    r"证据不足",
    r"不能证明",
    r"自行离职",
    r"主动离职",
    r"自动离职",
    r"未提供劳动",
    r"没有提供劳动",
    r"达到[^。；;，,]{0,20}?退休年龄",
    r"不符合[^。；;，,]{0,30}?(?:经济补偿|赔偿|补偿)[^。；;，,]{0,20}?条件",
    r"无法律依据",
    r"没有法律依据",
    r"(?:金额|数额|工资|加班费|劳动报酬)[^。；;，,]{0,40}?(?:不符|不同|错误|过高|过低|应按)",
    r"(?:解除|终止)[^。；;，,]{0,30}?合同",
    r"合同[^。；;，,]{0,30}?(?:解除|终止|未签订)",
    r"欠条[^。；;，,]{0,20}?(?:不是|并非|未)",
    r"原告[^。；;，,]{0,30}?未[^。；;，,]{0,30}?举证",
]


def strip_defense_heading(segment):
    text = safe_text(segment)

    for pattern in DEFENSE_START_PATTERNS:
        m = re.match(pattern, text)
        if m:
            text = text[m.end():]
            break

    return re.sub(r"\s+", "", text).strip("：:，,；;。")


def segment_is_only_no_defense(segment, body):
    text = normalize_text(segment)
    body_text = normalize_text(body)

    if not any(re.search(pattern, text) for pattern in NO_DEFENSE_PATTERNS):
        return False

    if len(body_text) <= 24:
        return True

    return not any(
        re.search(pattern, body_text)
        for pattern in SUBSTANTIVE_DEFENSE_PATTERNS
    )


def classify_single_defense(body):
    text = normalize_text(body)

    if not text:
        return "MINIMAL_DEFENSE"

    if any(
        re.search(pattern, text)
        for pattern in SUBSTANTIVE_DEFENSE_PATTERNS
    ):
        return "SUBSTANTIVE_DEFENSE"

    signal_count = sum(
        1 for signal in SUBSTANTIVE_SIGNALS
        if signal in text
    )

    if len(text) >= 80 and signal_count >= 2:
        return "SUBSTANTIVE_DEFENSE"

    if any(
        re.search(pattern, text)
        for pattern in MINIMAL_DEFENSE_PATTERNS
    ):
        return "MINIMAL_DEFENSE"

    if len(text) < 40:
        return "MINIMAL_DEFENSE"

    return "REVIEW"


def extract_defense_segments(judge_accusation):
    text = safe_text(judge_accusation)
    fact_start = first_match_start(FACT_START_PATTERNS, text)
    evidence_start = first_match_start(EVIDENCE_STAGE_PATTERNS, text)
    hard_end = earliest_non_none([fact_start, evidence_start])

    litigation_text = text[:hard_end] if hard_end is not None else text
    matches = []

    for pattern in DEFENSE_START_PATTERNS:
        for match in re.finditer(pattern, litigation_text):
            # 防止从“本院认为，被告...”之类法院说理里误抽。
            prefix = litigation_text[max(0, match.start() - 12):match.start()]
            if "本院认为" in prefix or "法院认为" in prefix:
                continue
            matches.append(match)

    matches.sort(key=lambda m: m.start())

    if not matches:
        return []

    segments = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(litigation_text)
        segment = litigation_text[start:end].strip()

        if segment:
            segments.append(segment)

    return segments


def classify_defense(judge_accusation):
    text = safe_text(judge_accusation)

    if not text:
        return {
            "D": "",
            "defense_type": "REVIEW",
            "defense_segments": [],
            "defendant_substantive_defense_count": 0,
            "defendant_minimal_defense_count": 0,
            "defendant_no_defense_count": 0,
            "defendant_review_defense_count": 0,
            "flags": ["DEFENSE_BOUNDARY_UNCERTAIN"],
        }

    segments = extract_defense_segments(text)
    flags = []

    if not segments:
        litigation_end = first_match_start(FACT_START_PATTERNS, text)
        litigation_text = text[:litigation_end] if litigation_end is not None else text

        if any(re.search(pattern, litigation_text) for pattern in NO_DEFENSE_PATTERNS):
            defense_type = "NO_DEFENSE"
            flags.append("NO_DEFENSE")
        elif any(re.search(pattern, litigation_text) for pattern in CAUTIOUS_DEFENSE_PATTERNS):
            defense_type = "REVIEW"
            flags.append("DEFENSE_BOUNDARY_UNCERTAIN")
        else:
            defense_type = "REVIEW"
            flags.append("DEFENSE_BOUNDARY_UNCERTAIN")

        return {
            "D": "",
            "defense_type": defense_type,
            "defense_segments": [],
            "defendant_substantive_defense_count": 0,
            "defendant_minimal_defense_count": 0,
            "defendant_no_defense_count": 1 if defense_type == "NO_DEFENSE" else 0,
            "defendant_review_defense_count": 1 if defense_type == "REVIEW" else 0,
            "flags": flags,
        }

    defense_records = []
    counts = Counter()
    d_parts = []

    for segment in segments:
        body = strip_defense_heading(segment)

        if segment_is_only_no_defense(segment, body):
            dtype = "NO_DEFENSE"
        else:
            dtype = classify_single_defense(body)
            if dtype != "NO_DEFENSE" and body:
                d_parts.append(segment.strip())

        counts[dtype] += 1
        defense_records.append({
            "text": segment,
            "body": body,
            "defense_type": dtype,
        })

    if counts["SUBSTANTIVE_DEFENSE"]:
        defense_type = "SUBSTANTIVE_DEFENSE"
    elif counts["REVIEW"]:
        defense_type = "REVIEW"
        flags.append("DEFENSE_QUALITY_UNCERTAIN")
    elif counts["MINIMAL_DEFENSE"]:
        defense_type = "MINIMAL_DEFENSE"
    elif counts["NO_DEFENSE"]:
        defense_type = "NO_DEFENSE"
    else:
        defense_type = "REVIEW"
        flags.append("DEFENSE_BOUNDARY_UNCERTAIN")

    return {
        "D": "\n".join(d_parts).strip(),
        "defense_type": defense_type,
        "defense_segments": defense_records,
        "defendant_substantive_defense_count": counts["SUBSTANTIVE_DEFENSE"],
        "defendant_minimal_defense_count": counts["MINIMAL_DEFENSE"],
        "defendant_no_defense_count": counts["NO_DEFENSE"],
        "defendant_review_defense_count": counts["REVIEW"],
        "flags": flags,
    }


# =========================================================
# F 提取
# =========================================================

def extract_fact_text(judge_accusation):
    text = safe_text(judge_accusation)

    if not text:
        return "", "REVIEW", ["COURT_FACT_BOUNDARY_UNCERTAIN"]

    start_match = None

    for pattern in FACT_START_PATTERNS:
        m = re.search(pattern, text)
        if m and (start_match is None or m.start() < start_match.start()):
            start_match = m

    if start_match is None:
        return "", "REVIEW", ["COURT_FACT_BOUNDARY_UNCERTAIN"]

    tail = text[start_match.start():]
    end_relative = first_match_start(REASON_START_PATTERNS, tail[start_match.end() - start_match.start():])

    if end_relative is not None:
        end = start_match.end() + end_relative
        fact_text = text[start_match.start():end].strip()
    else:
        fact_text = tail.strip()

    trailing_start = first_match_start(FACT_TRAILING_EVIDENCE_PATTERNS, fact_text)
    if trailing_start is not None and trailing_start > 40:
        fact_text = fact_text[:trailing_start].strip()

    flags = []

    if len(normalize_text(fact_text)) < 30:
        flags.append("COURT_FACT_TOO_SHORT")

    if any(re.search(pattern, fact_text) for pattern in LEAKAGE_PATTERNS):
        flags.append("FACT_POTENTIAL_LEAKAGE")

    if any(re.search(pattern, fact_text) for pattern in DEFENSE_START_PATTERNS):
        flags.append("FACT_CONTAINS_DEFENSE_TEXT")

    status = "CLEAR" if fact_text and not flags else "REVIEW"
    return fact_text, status, flags


# =========================================================
# 样本 QC
# =========================================================

def get_basic_exclusion_flags(case, document_type):
    flags = []
    category = case.get("Category") or {}
    if not isinstance(category, dict):
        category = {}

    if safe_text(category.get("cat_1")) != "劳动人事":
        flags.append("NOT_LABOR_CASE")

    if safe_text(category.get("cat_2")) not in TARGET_CAT2:
        flags.append("CAT2_NOT_TARGET")

    if document_type not in TARGET_DOCUMENT_TYPES:
        flags.append("DOCUMENT_TYPE_NOT_TARGET")

    accusation = safe_text(case.get("JudgeAccusation"))
    reason = safe_text(case.get("JudgeReason"))
    result = safe_text(case.get("JudgeResult"))

    if not accusation:
        flags.append("MISSING_JUDGE_ACCUSATION")
    if not reason:
        flags.append("MISSING_JUDGE_REASON")
    if not result:
        flags.append("MISSING_JUDGE_RESULT")

    if EXCLUDE_700_TRUNCATED:
        if len(accusation) == 700:
            flags.append("JUDGE_ACCUSATION_700_TRUNCATED")
        if len(reason) == 700:
            flags.append("JUDGE_REASON_700_TRUNCATED")

    return flags


def detect_document_type_conflict(record):
    doc = record["document_type"]
    reason = record["JudgeReason"]
    result = record["JudgeResult"]
    flags = []

    if doc == "裁定书":
        if "判决如下" in reason or "本判决" in result:
            flags.append("DOCUMENT_TYPE_CONFLICT")
    elif doc == "判决书":
        if "裁定如下" in reason or "本裁定" in result:
            flags.append("DOCUMENT_TYPE_CONFLICT")

    return flags


def process_case(file_path, input_dir, data, ctx_id, case, chash):
    record = make_base_record(file_path, input_dir, data, ctx_id, case, chash)
    flags = []

    basic_flags = get_basic_exclusion_flags(case, record["document_type"])
    flags.extend(basic_flags)

    if basic_flags:
        record["qc_status"] = "EXCLUDED"
        record["qc_flags"] = flags
        return record

    doc_flags = detect_document_type_conflict(record)
    flags.extend(doc_flags)

    p_text, p_status, p_flags = extract_plaintiff_text(record["JudgeAccusation"])
    claim_count, claim_status, claim_flags = estimate_claim_count_from_p(p_text)
    defense_info = classify_defense(record["JudgeAccusation"])
    f_text, f_status, f_flags = extract_fact_text(record["JudgeAccusation"])

    flags.extend(p_flags)
    flags.extend(claim_flags)
    flags.extend(defense_info["flags"])
    flags.extend(f_flags)

    record.update({
        "P": p_text,
        "D": defense_info["D"],
        "F": f_text,
        "defense_type": defense_info["defense_type"],
        "defense_segments": defense_info["defense_segments"],
        "defendant_substantive_defense_count": defense_info["defendant_substantive_defense_count"],
        "defendant_minimal_defense_count": defense_info["defendant_minimal_defense_count"],
        "defendant_no_defense_count": defense_info["defendant_no_defense_count"],
        "defendant_review_defense_count": defense_info["defendant_review_defense_count"],
        "claim_count": claim_count,
        "claim_count_status": claim_status,
        "plaintiff_status": p_status,
        "court_fact_status": f_status,
    })

    if doc_flags:
        record["qc_status"] = "REVIEW"
        flags.append("DOCUMENT_TYPE_REVIEW_REQUIRED")
    elif p_status != "CLEAR":
        record["qc_status"] = "REVIEW"
    elif defense_info["defense_type"] == "SUBSTANTIVE_DEFENSE" and f_status == "CLEAR":
        record["qc_status"] = "CLEAR"
    elif defense_info["defense_type"] in {"NO_DEFENSE", "MINIMAL_DEFENSE"}:
        record["qc_status"] = "EXCLUDED"
    else:
        record["qc_status"] = "REVIEW"

    record["qc_flags"] = sorted(set(flags))
    return record



def make_clear_export_record(record):
    return {
        "CaseId": record.get("CaseId"),
        "Case": record.get("Case"),
        "CaseProc": record.get("CaseProc"),
        "CaseType": record.get("CaseType"),
        "Category": record.get("Category"),
        "document_type": record.get("document_type"),
        "JudgeResult": record.get("JudgeResult"),
        "P": record.get("P"),
        "D": record.get("D"),
        "F": record.get("F"),
    }


def write_review_csv(path, records):
    fields = [
        "qc_status",
        "qc_flags",
        "defense_type",
        "plaintiff_status",
        "court_fact_status",
        "claim_count_status",
        "CaseId",
        "Case",
        "cat_2",
        "document_type",
        "source_file",
        "source_ctx_id",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in records:
            writer.writerow({
                "qc_status": r.get("qc_status", ""),
                "qc_flags": ";".join(r.get("qc_flags", [])),
                "defense_type": r.get("defense_type", ""),
                "plaintiff_status": r.get("plaintiff_status", ""),
                "court_fact_status": r.get("court_fact_status", ""),
                "claim_count_status": r.get("claim_count_status", ""),
                "CaseId": r.get("CaseId", ""),
                "Case": r.get("Case", ""),
                "cat_2": r.get("Category", {}).get("cat_2", ""),
                "document_type": r.get("document_type", ""),
                "source_file": r.get("source_file", ""),
                "source_ctx_id": r.get("source_ctx_id", ""),
            })


def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_statistics(records, scan_stats):
    """汇总 QC 记录；去重时排除的记录也计入 QC 分布。"""
    return {
        "scan_stats": dict(scan_stats),
        "qc_status_distribution": dict(Counter(r["qc_status"] for r in records)),
        "defense_type_distribution": dict(Counter(
            r["defense_type"] for r in records if r.get("defense_type")
        )),
        "court_fact_status_distribution": dict(Counter(
            r["court_fact_status"] for r in records if r.get("court_fact_status")
        )),
        "cat2_x_document_type_x_qc_status": dict(Counter(
            f"{r.get('Category', {}).get('cat_2', '')}__"
            f"{r.get('document_type', '')}__{r['qc_status']}"
            for r in records
        )),
        "qc_flags": dict(Counter(
            flag for r in records for flag in r.get("qc_flags", [])
        )),
    }


def write_outputs(records, scan_stats, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_records = [r for r in records if r["qc_status"] == "CLEAR"]
    review_records = [r for r in records if r["qc_status"] == "REVIEW"]
    excluded_records = [r for r in records if r["qc_status"] == "EXCLUDED"]
    clear_export_records = [
        make_clear_export_record(r)
        for r in clear_records
    ]

    stats = build_statistics(records, scan_stats)

    write_json(output_dir / "candidate_clear.json", clear_export_records)
    write_json(output_dir / "candidate_clear_full.json", clear_records)
    write_json(output_dir / "candidate_review.json", review_records)
    write_json(output_dir / "excluded_cases.json", excluded_records)
    write_json(output_dir / "qc_statistics.json", stats)
    write_review_csv(output_dir / "qc_review.csv", review_records)

    return {
        "candidate_clear": output_dir / "candidate_clear.json",
        "candidate_clear_full": output_dir / "candidate_clear_full.json",
        "candidate_review": output_dir / "candidate_review.json",
        "excluded_cases": output_dir / "excluded_cases.json",
        "qc_statistics": output_dir / "qc_statistics.json",
        "qc_review": output_dir / "qc_review.csv",
    }


# =========================================================
# 主程序
# =========================================================

def main():
    global INPUT_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="清洗劳动争议案件并提取 P/D/F 候选池（源自 E:/PythonProject）。")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    INPUT_DIR, OUTPUT_DIR = args.input_dir, args.output_dir
    if not INPUT_DIR.is_dir() or next(INPUT_DIR.rglob("*.json"), None) is None:
        parser.error(f"输入目录不存在或没有 JSON 文件：{INPUT_DIR}")

    scan_stats = Counter()
    seen_case_ids = set()
    seen_content_hashes = set()
    records = []

    for item in iter_json_cases(INPUT_DIR, scan_stats):
        file_path = item["file_path"]
        data = item["data"]
        ctx_id = item["ctx_id"]
        case = item["case"]

        if item["load_error"]:
            continue

        scan_stats["total_ctxs"] += 1

        category = case.get("Category") or {}
        if not isinstance(category, dict):
            category = {}

        if safe_text(category.get("cat_1")) != "劳动人事":
            continue

        scan_stats["labor_raw"] += 1

        case_id = safe_text(case.get("CaseId"))
        if case_id:
            if case_id in seen_case_ids:
                scan_stats["case_id_duplicates"] += 1
                record = make_base_record(file_path, INPUT_DIR, data, ctx_id, case)
                record["qc_status"] = "EXCLUDED"
                record["duplicate_reason"] = "CASE_ID_DUPLICATE"
                record["qc_flags"] = ["CASE_ID_DUPLICATE"]
                records.append(record)
                continue
            seen_case_ids.add(case_id)

        scan_stats["after_case_id_dedup"] += 1

        chash = content_hash(case)
        if chash in seen_content_hashes:
            scan_stats["content_duplicates"] += 1
            record = make_base_record(file_path, INPUT_DIR, data, ctx_id, case, chash)
            record["qc_status"] = "EXCLUDED"
            record["duplicate_reason"] = "CONTENT_DUPLICATE"
            record["qc_flags"] = ["CONTENT_DUPLICATE"]
            records.append(record)
            continue
        seen_content_hashes.add(chash)

        scan_stats["after_content_dedup"] += 1

        record = process_case(file_path, INPUT_DIR, data, ctx_id, case, chash)

        if not get_basic_exclusion_flags(case, record["document_type"]):
            scan_stats["base_eligible"] += 1

        records.append(record)

    output_paths = write_outputs(records, scan_stats, OUTPUT_DIR)

    print("=" * 70)
    print("劳动争议 P/D/F 候选池构建完成")
    print("=" * 70)
    print(f"劳动案件原始数量: {scan_stats.get('labor_raw', 0)}")
    print(f"CaseId 去重后数量: {scan_stats.get('after_case_id_dedup', 0)}")
    print(f"全文去重后数量: {scan_stats.get('after_content_dedup', 0)}")
    print(f"基础合格数量: {scan_stats.get('base_eligible', 0)}")

    qc_dist = Counter(r["qc_status"] for r in records)
    print("\nQC 状态:")
    for key, value in qc_dist.most_common():
        print(f"{key}: {value}")

    defense_dist = Counter(
        r.get("defense_type", "NOT_EVALUATED")
        for r in records
        if r["qc_status"] != "EXCLUDED" or r.get("defense_type")
    )
    print("\n抗辩类型:")
    for key, value in defense_dist.most_common():
        print(f"{key}: {value}")

    print("\n输出文件:")
    for name, path in output_paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
