import os
import csv
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 1. 项目路径
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"/"raw"
RESULT_DIR = ROOT_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 加载 .env
# =========================================================

load_dotenv(CONFIG_DIR / ".env")

API_KEY = os.getenv("CLOSEAI_API_KEY")
BASE_URL = os.getenv(
    "CLOSEAI_BASE_URL",
    "https://api.openai-proxy.org/v1"
)
MODEL = os.getenv(
    "MODEL",
    #  三个模型是:
    # qwen3.8-max
    # gpt-5.6
    # qwen3.8-max

    "qwen3.8-max"
)

if not API_KEY:
    raise ValueError("没有读取到 CLOSEAI_API_KEY，请检查 config/.env")


# =========================================================
# 3. 实验配置
# =========================================================

# 正式400案：
# TEST_LIMIT = None
TEST_LIMIT = None

# 并发线程数
# 建议先 5，稳定后可以改成 8
MAX_WORKERS = 5

# 单次 API 请求超时（秒）
REQUEST_TIMEOUT = 90.0

# 我们自己控制重试，SDK 不额外重试
MAX_RETRIES = 5

# 第一次失败后的等待基数
# 实际等待：3s、6s、9s、12s...
RETRY_WAIT = 3

VALID_LABELS = {"A", "B", "C", "D"}


# =========================================================
# 4. 数据文件
# =========================================================

S1_PATH = DATA_DIR / "ccf_400_S1_correct_roles.json"
S2_PATH = DATA_DIR / "ccf_400_S2_no_roles.json"
S3_PATH = DATA_DIR / "ccf_400_S3_mismatched_roles.json"
GOLD_PATH = DATA_DIR / "ccf_400_verified_gold.json"


# =========================================================
# 5. 结果文件
# =========================================================

RAW_RESULT_PATH = RESULT_DIR / f"{MODEL}_raw_results.jsonl"

FINAL_JSON_PATH = (
    RESULT_DIR
    / f"{MODEL}_S1_S2_S3_results.json"
)

FINAL_CSV_PATH = (
    RESULT_DIR
    / f"{MODEL}_S1_S2_S3_results.csv"
)


# =========================================================
# 6. 线程安全
# =========================================================

write_lock = threading.Lock()

# 每个线程单独创建自己的 OpenAI Client
# 避免多个线程共享同一 client 的潜在状态问题
thread_local = threading.local()


def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )

    return thread_local.client


# =========================================================
# 7. System Prompt
# =========================================================

SYSTEM_PROMPT = """
你是一名进行中国劳动争议案件裁判判断的法律分析模型。

你的任务是根据提供的案件材料，判断法院对当事人最终实体诉讼请求的支持情况。

标签定义：

A：全部支持
B：部分支持
C：全部不支持
D：因程序性原因未进入实体裁判

判断规则：

1. 以当事人庭审中最终保留的实体诉讼请求为判断对象；
2. 明确撤回、放弃的诉讼请求不再计入；
3. 对多个独立实体请求仅支持部分的，判断为 B；
4. 对金额、期间等仅部分支持的，判断为 B；
5. 诉讼费等程序性负担一般不作为独立实体请求；
6. 仅根据当前提供的案件材料进行判断，不假设存在未提供的信息。

最终只输出一个大写字母：

A、B、C 或 D。

不要解释，不要输出其他内容。
""".strip()


# =========================================================
# 8. 读取 JSON
# =========================================================

def load_json(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


s1_data = load_json(S1_PATH)
s2_data = load_json(S2_PATH)
s3_data = load_json(S3_PATH)
gold_data = load_json(GOLD_PATH)


# =========================================================
# 9. Gold
#
# Gold 永远不会发送给模型
# =========================================================

gold_map = {
    int(item["id"]):
        str(item["gold"]).strip().upper()
    for item in gold_data
}


# =========================================================
# 10. 检查案件 ID
# =========================================================

s1_ids = [
    int(x["id"])
    for x in s1_data
]

s2_ids = [
    int(x["id"])
    for x in s2_data
]

s3_ids = [
    int(x["id"])
    for x in s3_data
]

if not (
    s1_ids
    == s2_ids
    == s3_ids
):

    raise ValueError(
        "S1、S2、S3 的案件 ID 或顺序不一致"
    )


# =========================================================
# 11. 建立 ID 映射
# =========================================================

datasets = {

    "S1": {
        int(x["id"]): x
        for x in s1_data
    },

    "S2": {
        int(x["id"]): x
        for x in s2_data
    },

    "S3": {
        int(x["id"]): x
        for x in s3_data
    },

}


# =========================================================
# 12. 构造 Prompt
# =========================================================

def build_user_prompt(
    setting: str,
    case: dict
):

    # S1 = 正确角色
    # S3 = 错配角色
    #
    # S3 数据文件已经完成角色错配，
    # 所以这里只按当前 P/D/F 字段加标题。
    if setting in {
        "S1",
        "S3"
    }:

        return f"""
【原告诉称】
{case["P"]}

【被告抗辩】
{case["D"]}

【法院查明事实】
{case["F"]}

【法律规则】
{case["R"]}

请判断该案裁判结果。
只输出 A、B、C 或 D。
""".strip()

    # S2 = 无角色
    if setting == "S2":

        return f"""
【案件材料】
{case["text"]}

请判断该案裁判结果。
只输出 A、B、C 或 D。
""".strip()

    raise ValueError(
        f"未知实验条件：{setting}"
    )


# =========================================================
# 13. 解析标签
# =========================================================

def parse_label(content):

    if content is None:
        return None

    text = (
        str(content)
        .strip()
        .upper()
    )

    if text in VALID_LABELS:
        return text

    # 容错：
    # "答案：A"
    # "A."
    for label in [
        "A",
        "B",
        "C",
        "D"
    ]:

        if label in text:
            return label

    return None


# =========================================================
# 14. 调用模型
# =========================================================

def call_model(
    setting: str,
    case: dict
):

    user_prompt = build_user_prompt(
        setting,
        case
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            client = get_client()

            start_time = time.time()

            response = (
                client
                .chat
                .completions
                .create(
                    model=MODEL,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                SYSTEM_PROMPT
                        },
                        {
                            "role":
                                "user",
                            "content":
                                user_prompt
                        }
                    ],
                    #     "glm-5.3": extra_body = {
                    #         "thinking": {
                    #             "type": "enabled"
                    #         },
                    #         "reasoning_effort": "low"
                    #     }
                    #     "qwen3.8-max": extra_body ={
                    #         "reasoning_effort": "low"
                    #     }
                    #     # GPT-5.6：不启用额外显式推理
                    #     "gpt-5.6": {},

                    extra_body={
                        "reasoning_effort": "low"
                    }
                )
            )

            latency = round(
                time.time()
                - start_time,
                3
            )

            message = (
                response
                .choices[0]
                .message
            )

            raw_output = (
                message.content
            )

            pred = parse_label(
                raw_output
            )

            usage = getattr(
                response,
                "usage",
                None
            )

            prompt_tokens = None
            completion_tokens = None
            reasoning_tokens = None
            total_tokens = None

            if usage is not None:

                prompt_tokens = getattr(
                    usage,
                    "prompt_tokens",
                    None
                )

                completion_tokens = getattr(
                    usage,
                    "completion_tokens",
                    None
                )

                total_tokens = getattr(
                    usage,
                    "total_tokens",
                    None
                )

                details = getattr(
                    usage,
                    "completion_tokens_details",
                    None
                )

                if details is not None:

                    reasoning_tokens = getattr(
                        details,
                        "reasoning_tokens",
                        None
                    )

            return {

                "pred":
                    pred,

                "raw_output":
                    raw_output,

                "requested_model":
                    MODEL,

                "returned_model":
                    getattr(
                        response,
                        "model",
                        None
                    ),

                "latency":
                    latency,

                "prompt_tokens":
                    prompt_tokens,

                "completion_tokens":
                    completion_tokens,

                "reasoning_tokens":
                    reasoning_tokens,

                "total_tokens":
                    total_tokens,

                "attempts":
                    attempt,

                "error":
                    None,
            }

        except Exception as e:

            last_error = (
                f"{type(e).__name__}: {e}"
            )

            print(
                f"[失败] "
                f"{MODEL} / "
                f"{setting} / "
                f"Case {case['id']} / "
                f"第 {attempt}/{MAX_RETRIES} 次\n"
                f"{last_error}",
                flush=True
            )

            if attempt < MAX_RETRIES:

                wait_time = (
                    RETRY_WAIT
                    * attempt
                )

                print(
                    f"[重试] "
                    f"{wait_time} 秒后再次调用 "
                    f"{setting} / "
                    f"Case {case['id']}",
                    flush=True
                )

                time.sleep(
                    wait_time
                )

    return {

        "pred":
            None,

        "raw_output":
            None,

        "requested_model":
            MODEL,

        "returned_model":
            None,

        "latency":
            None,

        "prompt_tokens":
            None,

        "completion_tokens":
            None,

        "reasoning_tokens":
            None,

        "total_tokens":
            None,

        "attempts":
            MAX_RETRIES,

        "error":
            last_error,
    }


# =========================================================
# 15. 单个并发任务
# =========================================================

def run_one_task(
    case_id: int,
    setting: str
):

    case = datasets[
        setting
    ][case_id]

    api_result = call_model(
        setting,
        case
    )

    pred = api_result[
        "pred"
    ]

    gold = gold_map.get(
        case_id
    )

    correct = (
        pred == gold
        if pred in VALID_LABELS
        else False
    )

    return {

        "case_id":
            case_id,

        "model":
            MODEL,

        "setting":
            setting,

        "gold":
            gold,

        "pred":
            pred,

        "correct":
            correct,

        "requested_model":
            api_result[
                "requested_model"
            ],

        "returned_model":
            api_result[
                "returned_model"
            ],

        "raw_output":
            api_result[
                "raw_output"
            ],

        "latency":
            api_result[
                "latency"
            ],

        "prompt_tokens":
            api_result[
                "prompt_tokens"
            ],

        "completion_tokens":
            api_result[
                "completion_tokens"
            ],

        "reasoning_tokens":
            api_result[
                "reasoning_tokens"
            ],

        "total_tokens":
            api_result[
                "total_tokens"
            ],

        "attempts":
            api_result[
                "attempts"
            ],

        "error":
            api_result[
                "error"
            ],
    }


# =========================================================
# 16. 写 JSONL
#
# 加锁，防止多个线程同时写乱文件
# =========================================================

def save_result_immediately(
    row: dict
):

    line = (
        json.dumps(
            row,
            ensure_ascii=False
        )
        + "\n"
    )

    with write_lock:

        with RAW_RESULT_PATH.open(
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                line
            )

            f.flush()

            os.fsync(
                f.fileno()
            )


# =========================================================
# 17. 读取已完成任务
# =========================================================

def load_completed_tasks():

    completed = set()

    if not RAW_RESULT_PATH.exists():
        return completed

    with RAW_RESULT_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = (
                line
                .strip()
            )

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

                pred = row.get(
                    "pred"
                )

                if pred in VALID_LABELS:

                    completed.add(
                        (
                            int(
                                row[
                                    "case_id"
                                ]
                            ),

                            str(
                                row[
                                    "setting"
                                ]
                            ),

                            str(
                                row[
                                    "model"
                                ]
                            ),
                        )
                    )

            except Exception:
                continue

    return completed


# =========================================================
# 18. 整理 JSONL
# =========================================================

def load_final_rows_from_jsonl():

    rows = {}

    if not RAW_RESULT_PATH.exists():
        return []

    with RAW_RESULT_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = (
                line
                .strip()
            )

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

                key = (
                    int(
                        row[
                            "case_id"
                        ]
                    ),
                    str(
                        row[
                            "setting"
                        ]
                    ),
                    str(
                        row[
                            "model"
                        ]
                    ),
                )

                old = rows.get(
                    key
                )

                if old is None:

                    rows[
                        key
                    ] = row

                elif (
                    row.get("pred")
                    in VALID_LABELS
                ):

                    rows[
                        key
                    ] = row

                elif (
                    old.get("pred")
                    not in VALID_LABELS
                ):

                    rows[
                        key
                    ] = row

            except Exception:
                continue

    setting_order = {
        "S1": 1,
        "S2": 2,
        "S3": 3,
    }

    final_rows = list(
        rows.values()
    )

    final_rows.sort(
        key=lambda x: (
            int(
                x[
                    "case_id"
                ]
            ),
            setting_order.get(
                x[
                    "setting"
                ],
                99
            ),
        )
    )

    return final_rows


# =========================================================
# 19. 导出 JSON / CSV
# =========================================================

def export_final_results():

    rows = (
        load_final_rows_from_jsonl()
    )

    with FINAL_JSON_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            rows,
            f,
            ensure_ascii=False,
            indent=2
        )

    fieldnames = [

        "case_id",
        "model",
        "setting",
        "gold",
        "pred",
        "correct",

        "requested_model",
        "returned_model",

        "raw_output",
        "latency",

        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",

        "attempts",
        "error",
    ]

    with FINAL_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =========================================================
# 20. 主实验
# =========================================================

def run_experiment():

    if TEST_LIMIT is None:

        run_ids = s1_ids

    else:

        run_ids = (
            s1_ids[
                :TEST_LIMIT
            ]
        )

    settings = [
        "S1",
        "S2",
        "S3",
    ]

    completed_tasks = (
        load_completed_tasks()
    )

    tasks = []

    for case_id in run_ids:

        for setting in settings:

            task_key = (
                case_id,
                setting,
                MODEL
            )

            if (
                task_key
                not in completed_tasks
            ):

                tasks.append(
                    (
                        case_id,
                        setting
                    )
                )

    total_tasks = (
        len(run_ids)
        * len(settings)
    )

    already_done = (
        total_tasks
        - len(tasks)
    )

    print(
        "=" * 76
    )

    print(
        "CCF 司法角色信息并发实验"
    )

    print(
        f"模型：{MODEL}"
    )

    print(
        f"案件数：{len(run_ids)}"
    )

    print(
        f"理论任务数：{total_tasks}"
    )

    print(
        f"已完成：{already_done}"
    )

    print(
        f"待运行：{len(tasks)}"
    )

    print(
        f"并发数：{MAX_WORKERS}"
    )

    print(
        f"单请求超时："
        f"{REQUEST_TIMEOUT} 秒"
    )

    print(
        f"最大重试："
        f"{MAX_RETRIES} 次"
    )

    print(
        f"断点文件："
        f"{RAW_RESULT_PATH}"
    )

    print(
        "=" * 76
    )

    if not tasks:

        print(
            "当前范围内所有任务均已完成。"
        )

        export_final_results()

        return

    completed_count = 0
    success_count = 0
    failed_count = 0

    # =====================================================
    # 并发调用
    # =====================================================

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {

            executor.submit(
                run_one_task,
                case_id,
                setting
            ):
            (
                case_id,
                setting
            )

            for (
                case_id,
                setting
            ) in tasks
        }

        for future in as_completed(
            future_map
        ):

            (
                case_id,
                setting
            ) = future_map[
                future
            ]

            try:

                row = (
                    future.result()
                )

            except Exception as e:

                row = {

                    "case_id":
                        case_id,

                    "model":
                        MODEL,

                    "setting":
                        setting,

                    "gold":
                        gold_map.get(
                            case_id
                        ),

                    "pred":
                        None,

                    "correct":
                        False,

                    "requested_model":
                        MODEL,

                    "returned_model":
                        None,

                    "raw_output":
                        None,

                    "latency":
                        None,

                    "prompt_tokens":
                        None,

                    "completion_tokens":
                        None,

                    "reasoning_tokens":
                        None,

                    "total_tokens":
                        None,

                    "attempts":
                        0,

                    "error":
                        (
                            f"UnhandledError: "
                            f"{type(e).__name__}: "
                            f"{e}"
                        )
                }

            # =============================================
            # 立即保存
            # =============================================

            save_result_immediately(
                row
            )

            completed_count += 1

            if (
                row["pred"]
                in VALID_LABELS
            ):

                success_count += 1

            else:

                failed_count += 1

            print(
                f"[{completed_count}/{len(tasks)}] "
                f"Case={row['case_id']} | "
                f"{row['setting']} | "
                f"Pred={row['pred']} | "
                f"Gold={row['gold']} | "
                f"Correct={row['correct']} | "
                f"耗时={row['latency']}s | "
                f"尝试={row['attempts']}",
                flush=True
            )

    # =====================================================
    # 当前批次结束后整理结果
    # =====================================================

    export_final_results()

    print()

    print(
        "=" * 76
    )

    print(
        "本轮并发实验结束"
    )

    print(
        f"本轮成功："
        f"{success_count}"
    )

    print(
        f"本轮失败："
        f"{failed_count}"
    )

    print(
        f"原始断点日志："
        f"{RAW_RESULT_PATH}"
    )

    print(
        f"整理后 JSON："
        f"{FINAL_JSON_PATH}"
    )

    print(
        f"整理后 CSV："
        f"{FINAL_CSV_PATH}"
    )

    print(
        "=" * 76
    )

    if failed_count > 0:

        print(
            "存在失败任务。"
            "重新运行 main.py 即可；"
            "程序会自动跳过已成功任务，"
            "只重跑失败任务。"
        )


# =========================================================
# 21. 程序入口
# =========================================================

if __name__ == "__main__":

    run_experiment()
