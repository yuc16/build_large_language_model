from __future__ import annotations

import argparse
import gc
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer_sft import DEFAULT_ADAPTER_DIR, DEFAULT_MODEL_DIR, pick_device, pick_dtype
from train_sft import format_prompt


@dataclass
class EvalCase:
    name: str
    instruction: str
    input_text: str = ""
    reference: str | None = None


DEFAULT_CASES = [
    EvalCase(
        name="city_only",
        instruction="法国的首都在哪里？请只输出城市名，不要加标点。",
        reference="巴黎",
    ),
    EvalCase(
        name="math_number_only",
        instruction="12 + 35 等于多少？请只输出数字，不要解释。",
        reference="47",
    ),
    EvalCase(
        name="translation_only",
        instruction="把 hello 翻译成中文，只输出译文，不要额外说明。",
        reference="你好",
    ),
    EvalCase(
        name="extract_company",
        instruction="请从输入中提取公司名称，只输出公司名。",
        input_text="苹果公司于2007年发布第一代 iPhone。",
        reference="苹果公司",
    ),
    EvalCase(
        name="extract_date",
        instruction="请从输入中提取日期，只输出日期本身。",
        input_text="会议通知：项目复盘会将于2026年4月12日下午三点举行。",
        reference="2026年4月12日",
    ),
    EvalCase(
        name="extract_amount",
        instruction="请从输入中提取合同金额，只输出金额。",
        input_text="根据合同，本次采购总金额为人民币128万元，已支付30%。",
        reference="人民币128万元",
    ),
    EvalCase(
        name="sentiment_label",
        instruction="判断这句话的情感倾向，只输出“正面”或“负面”，不要解释。",
        input_text="这家餐厅上菜很快，服务也很周到。",
        reference="正面",
    ),
    EvalCase(
        name="intent_classification",
        instruction="识别用户意图，只输出以下标签之一：投诉、咨询、退款。",
        input_text="我买的耳机左边没有声音，想问下怎么处理。",
        reference="投诉",
    ),
    EvalCase(
        name="rewrite_formal",
        instruction="把下面这句话改写得更正式一些，只输出改写后的句子。",
        input_text="这个方案挺不错的，我们明天就开干吧。",
    ),
    EvalCase(
        name="summary_two_lines",
        instruction="请用两句话总结输入内容，每句话不超过20个字。",
        input_text=(
            "监督微调是在预训练模型基础上，使用带标注的指令-回答数据继续训练，"
            "让模型更符合人类期望的回答方式。它通常用于提升模型的指令跟随能力、"
            "输出格式稳定性和任务表现。"
        ),
    ),
    EvalCase(
        name="json_output",
        instruction="根据输入生成 JSON，对象只包含 company 和 action 两个字段，不要输出其他内容。",
        input_text="小米公司今天发布了新款扫地机器人。",
        reference='{"company":"小米公司","action":"发布了新款扫地机器人"}',
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 base model 和 SFT LoRA 在同一组测试题上的生成结果。"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=None,
        help="Optional jsonl file with fields: instruction, input, and optional reference.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_cases(eval_path: Path | None, limit: int | None) -> list[EvalCase]:
    if eval_path is None:
        cases = list(DEFAULT_CASES)
    else:
        rows = [
            json.loads(line)
            for line in eval_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cases = []
        for index, row in enumerate(rows, start=1):
            instruction = row["instruction"]
            input_text = row.get("input", row.get("input_text", ""))
            reference = row.get("reference", row.get("answer", row.get("output")))
            cases.append(
                EvalCase(
                    name=row.get("name", f"case_{index}"),
                    instruction=instruction,
                    input_text=input_text,
                    reference=reference,
                )
            )
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("no eval cases found")
    return cases


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：、“”‘’\"'.,!?;:()（）【】\[\]<>《》\-]", "", text)
    return text


def compare_to_reference(output: str, reference: str | None) -> dict[str, bool] | None:
    if reference is None:
        return None
    normalized_output = normalize_text(output)
    normalized_reference = normalize_text(reference)
    return {
        "exact_match": normalized_output == normalized_reference,
        "contains_reference": normalized_reference in normalized_output,
    }


def generate_text(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore
    do_sample = temperature > 0
    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,  # type: ignore
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
        )
    elapsed = time.perf_counter() - start_time
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)  # type: ignore
    return text, elapsed


def unload_model(model, device: torch.device) -> None:
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def evaluate_base_model(
    tokenizer: AutoTokenizer,
    cases: list[EvalCase],
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)  # type: ignore
    model.eval()
    results: list[dict[str, Any]] = []
    for case in cases:
        prompt = format_prompt(case.instruction, case.input_text)
        output, elapsed = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        results.append(
            {
                "name": case.name,
                "prompt": prompt,
                "output": output,
                "latency_seconds": elapsed,
                "reference_eval": compare_to_reference(output, case.reference),
            }
        )
    unload_model(model, device)
    return results


def evaluate_sft_model(
    tokenizer: AutoTokenizer,
    cases: list[EvalCase],
    model_dir: Path,
    adapter_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    base_model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)  # type: ignore
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    model.eval()
    results: list[dict[str, Any]] = []
    for case in cases:
        prompt = format_prompt(case.instruction, case.input_text)
        output, elapsed = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        results.append(
            {
                "name": case.name,
                "prompt": prompt,
                "output": output,
                "latency_seconds": elapsed,
                "reference_eval": compare_to_reference(output, case.reference),
            }
        )
    unload_model(model, device)
    return results


def summarize(label: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total_latency = sum(item["latency_seconds"] for item in results)
    reference_results = [
        item["reference_eval"] for item in results if item["reference_eval"] is not None
    ]
    exact_matches = sum(
        1 for item in reference_results if item is not None and item["exact_match"]
    )
    contains_matches = sum(
        1
        for item in reference_results
        if item is not None and item["contains_reference"]
    )
    return {
        "label": label,
        "cases": len(results),
        "reference_cases": len(reference_results),
        "exact_match_count": exact_matches,
        "contains_reference_count": contains_matches,
        "avg_latency_seconds": total_latency / max(len(results), 1),
    }


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(device)
    cases = load_cases(args.eval_path, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_results = evaluate_base_model(
        tokenizer=tokenizer,
        cases=cases,
        model_dir=args.model_dir,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    sft_results = evaluate_sft_model(
        tokenizer=tokenizer,
        cases=cases,
        model_dir=args.model_dir,
        adapter_dir=args.adapter_dir,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    merged_results: list[dict[str, Any]] = []
    for case, base_result, sft_result in zip(cases, base_results, sft_results):
        merged_results.append(
            {
                "name": case.name,
                "instruction": case.instruction,
                "input_text": case.input_text,
                "reference": case.reference,
                "prompt": base_result["prompt"],
                "base_output": base_result["output"],
                "sft_output": sft_result["output"],
                "base_latency_seconds": base_result["latency_seconds"],
                "sft_latency_seconds": sft_result["latency_seconds"],
                "base_reference_eval": base_result["reference_eval"],
                "sft_reference_eval": sft_result["reference_eval"],
            }
        )

    base_summary = summarize("base", base_results)
    sft_summary = summarize("sft", sft_results)

    print("mode=ab_eval")
    print(f"device={device}")
    print(f"model_dir={args.model_dir}")
    print(f"adapter_dir={args.adapter_dir}")
    print(f"cases={len(cases)}")
    print(f"temperature={args.temperature}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print("")
    for index, result in enumerate(merged_results, start=1):
        print(f"[{index}] {result['name']}")
        print("instruction:")
        print(result["instruction"])
        if result["input_text"]:
            print("input:")
            print(result["input_text"])
        if result["reference"] is not None:
            print(f"reference: {result['reference']}")
        print("base_output:")
        print(result["base_output"])
        print("sft_output:")
        print(result["sft_output"])
        if result["base_reference_eval"] is not None:
            print(f"base_eval: {result['base_reference_eval']}")
            print(f"sft_eval: {result['sft_reference_eval']}")
        print(
            "latency_seconds: "
            f"base={result['base_latency_seconds']:.3f} "
            f"sft={result['sft_latency_seconds']:.3f}"
        )
        print("")

    print("summary:")
    print(
        json.dumps(
            {"base": base_summary, "sft": sft_summary}, ensure_ascii=False, indent=2
        )
    )

    if args.output_json is not None:
        payload = {
            "config": {
                "device": str(device),
                "model_dir": str(args.model_dir),
                "adapter_dir": str(args.adapter_dir),
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "cases": len(cases),
            },
            "summary": {"base": base_summary, "sft": sft_summary},
            "results": merged_results,
            "eval_cases": [asdict(case) for case in cases],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved_results={args.output_json}")


if __name__ == "__main__":
    main()
