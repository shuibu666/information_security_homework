# coding=utf-8
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LogitsProcessorList,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.prompt_utils import load_prompts, str2bool
from course_project.processors.adaptive_watermark_processor import AdaptiveDeltaWatermarkLogitsProcessor
from course_project.processors.cakl_watermark_processor import CAKLModelAssistedDetector, CAKLWatermarkLogitsProcessor
from watermark_processor import WatermarkDetector, WatermarkLogitsProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline and adaptive watermark experiments.")
    parser.add_argument("--model_name_or_path", type=str, default="./models/opt-125m")
    parser.add_argument("--prompt_source", type=str, choices=["prompts_file", "hf_dataset"], default="hf_dataset")
    parser.add_argument("--prompts_path", type=str, default="course_project/data/prompts.txt")
    parser.add_argument("--save_loaded_prompts_path", type=str, default="course_project/data/prompts_c4_realnewslike_500.txt")
    parser.add_argument("--dataset_name", type=str, default="c4")
    parser.add_argument("--dataset_config_name", type=str, default="realnewslike")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--dataset_streaming", type=str2bool, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dataset_seed", type=int, default=1234)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--dataset_skip_examples", type=int, default=0)
    parser.add_argument("--min_prompt_tokens", type=int, default=50)
    parser.add_argument("--min_source_tokens", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default="course_project/outputs/results.csv")
    parser.add_argument("--prompt_max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--target_new_tokens", type=int, default=None)
    parser.add_argument("--length_tolerance", type=int, default=5)
    parser.add_argument("--paper_style_prompt", type=str2bool, default=False)
    parser.add_argument("--paper_completion_tokens", type=int, default=None)
    parser.add_argument("--generation_seed", type=int, default=123)
    parser.add_argument("--use_sampling", type=str2bool, default=True)
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--n_beams", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--load_fp16", type=str2bool, default=False)
    parser.add_argument("--seed_separately", type=str2bool, default=True)
    parser.add_argument("--seeding_scheme", type=str, default="simple_1")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--detection_z_threshold", type=float, default=4.0)
    parser.add_argument("--normalizers", type=str, default="")
    parser.add_argument("--ignore_repeated_bigrams", type=str2bool, default=False)
    parser.add_argument("--select_green_tokens", type=str2bool, default=True)
    parser.add_argument("--fixed_deltas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--run_baseline", type=str2bool, default=True)
    parser.add_argument("--run_fixed", type=str2bool, default=True)
    parser.add_argument("--run_adaptive", type=str2bool, default=True)
    parser.add_argument("--adaptive_delta_min", type=float, default=0.5)
    parser.add_argument("--adaptive_delta_max", type=float, default=3.0)
    parser.add_argument("--adaptive_entropy_floor", type=float, default=0.20)
    parser.add_argument("--adaptive_delta_exponent", type=float, default=0.5)
    parser.add_argument("--run_cakl", type=str2bool, default=True)
    parser.add_argument("--kl_epsilon", type=float, default=0.02)
    parser.add_argument("--cakl_delta_max", type=float, default=3.0)
    parser.add_argument("--confidence_entropy_threshold", type=float, default=0.35)
    parser.add_argument("--confidence_top1_threshold", type=float, default=0.85)
    parser.add_argument("--candidate_top_p", type=float, default=0.95)
    parser.add_argument("--window_sizes", type=str, default="20,40,80,max")
    parser.add_argument("--use_model_assisted_detector", type=str2bool, default=True)
    parser.add_argument("--limit_prompts", type=int, default=500)
    return parser.parse_args()


def infer_model_family(model_name_or_path: str) -> tuple[bool, bool]:
    model_name = model_name_or_path.lower()
    is_seq2seq = any(tag in model_name for tag in ["t5", "t0"])
    is_decoder_only = any(tag in model_name for tag in ["gpt", "opt", "bloom", "llama", "qwen", "mistral", "gemma"])
    if not is_seq2seq and not is_decoder_only:
        is_decoder_only = True
    return is_seq2seq, is_decoder_only


def load_model_and_tokenizer(args):
    is_seq2seq, is_decoder_only = infer_model_family(args.model_name_or_path)

    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    else:
        model_kwargs = {}
        if args.load_fp16 and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
        if not (args.load_fp16 and not is_seq2seq):
            model = model.to(device)
    else:
        device = "cpu"

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device, is_decoder_only


def resolve_prompt_max_length(args, model) -> int:
    if args.prompt_max_length is not None:
        return args.prompt_max_length

    generation_max_tokens = resolve_generation_window(args)[1]
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_positions, int) and max_positions > generation_max_tokens:
        return max_positions - generation_max_tokens

    max_length = getattr(model.config, "max_length", None)
    if isinstance(max_length, int) and max_length > generation_max_tokens:
        return max_length - generation_max_tokens

    return max(1, 2048 - generation_max_tokens)


def resolve_generation_window(args) -> tuple[int | None, int]:
    if args.target_new_tokens is None:
        return None, args.max_new_tokens

    target_new_tokens = max(1, args.target_new_tokens)
    tolerance = max(0, args.length_tolerance)
    min_new_tokens = max(1, target_new_tokens - tolerance)
    max_new_tokens = target_new_tokens + tolerance
    return min_new_tokens, max_new_tokens


def build_generation_kwargs(args, tokenizer):
    min_new_tokens, max_new_tokens = resolve_generation_window(args)
    generation_kwargs = {"max_new_tokens": max_new_tokens}
    if min_new_tokens is not None:
        generation_kwargs["min_new_tokens"] = min_new_tokens
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if args.use_sampling:
        generation_kwargs.update({"do_sample": True, "top_k": 0, "temperature": args.sampling_temp})
    else:
        generation_kwargs.update({"num_beams": args.n_beams})
    return generation_kwargs


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_completion(prompt, args, model, tokenizer, device, is_decoder_only, logits_processor=None, return_prompt_ids=False):
    prompt_max_length = resolve_prompt_max_length(args, model)
    tokenized_input = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_max_length,
    ).to(device)

    generation_kwargs = build_generation_kwargs(args, tokenizer)
    if logits_processor is not None:
        generation_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

    seed_everything(args.generation_seed)
    with torch.no_grad():
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    if is_decoder_only:
        generated_ids = output_ids[:, tokenized_input["input_ids"].shape[-1]:]
    else:
        generated_ids = output_ids

    decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if return_prompt_ids:
        return decoded_text, generated_ids[0].detach().cpu(), tokenized_input["input_ids"][0].detach().cpu()
    return decoded_text, generated_ids[0].detach().cpu()


def compute_text_metrics(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not words:
        return {"word_count": 0, "distinct_1": 0.0, "distinct_2": 0.0, "repetition_rate": 0.0}

    unique_words = len(set(words))
    bigrams = list(zip(words, words[1:]))
    unique_bigrams = len(set(bigrams)) if bigrams else 0

    return {
        "word_count": len(words),
        "distinct_1": unique_words / len(words),
        "distinct_2": (unique_bigrams / len(bigrams)) if bigrams else 0.0,
        "repetition_rate": 1.0 - (unique_words / len(words)),
    }


def build_detector(args, tokenizer, device, vocab_ids):
    normalizers = args.normalizers.split(",") if args.normalizers else []
    return WatermarkDetector(
        vocab=vocab_ids,
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=torch.device(device),
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=normalizers,
        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
        select_green_tokens=args.select_green_tokens,
    )


def build_model_assisted_detector(args, model, tokenizer, device, vocab_ids, use_candidate_greenlist: bool, use_confidence_gate: bool):
    return CAKLModelAssistedDetector(
        vocab=vocab_ids,
        gamma=args.gamma,
        delta=args.cakl_delta_max,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
        candidate_top_p=args.candidate_top_p,
        use_candidate_greenlist=use_candidate_greenlist,
        use_confidence_gate=use_confidence_gate,
        entropy_threshold=args.confidence_entropy_threshold,
        top1_threshold=args.confidence_top1_threshold,
        window_sizes=args.window_sizes,
        z_threshold=args.detection_z_threshold,
    )


def detect_text(detector, token_ids):
    if token_ids.numel() <= detector.min_prefix_len:
        return {
            "num_tokens_scored": 0,
            "num_green_tokens": 0,
            "green_fraction": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "prediction": False,
            "confidence": 0.0,
        }
    return detector.detect(tokenized_text=token_ids.to(detector.device))


def detect_text_model_assisted(detector, prompt_token_ids, token_ids):
    if detector is None:
        return {}
    return detector.detect(prompt_token_ids=prompt_token_ids, generated_token_ids=token_ids)


def format_prediction(prediction: bool) -> str:
    return "Watermarked" if prediction else "Human/Unwatermarked"


def build_row(prompt_id, prompt, method, delta, delta_min, delta_max, generated_text, token_ids, detection_result, text_metrics, extra_metrics=None):
    extra_metrics = extra_metrics or {}
    confidence = detection_result.get("confidence")
    row = {
        "prompt_id": prompt_id,
        "method": method,
        "delta": delta,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "prompt": prompt,
        "generated_text": generated_text,
        "token_ids": json.dumps(token_ids.detach().cpu().tolist()),
        "tokens_counted": detection_result.get("num_tokens_scored"),
        "num_green_tokens": detection_result.get("num_green_tokens"),
        "green_fraction": detection_result.get("green_fraction"),
        "z_score": detection_result.get("z_score"),
        "p_value": detection_result.get("p_value"),
        "prediction": format_prediction(detection_result.get("prediction", False)),
        "confidence": confidence if confidence is not None else 0.0,
        "word_count": text_metrics["word_count"],
        "distinct_1": text_metrics["distinct_1"],
        "distinct_2": text_metrics["distinct_2"],
        "repetition_rate": text_metrics["repetition_rate"],
        **extra_metrics,
    }
    for field in [
        "avg_step_delta",
        "observed_delta_min",
        "observed_delta_max",
        "weighted_z_score",
        "winmax_weighted_z_score",
        "weighted_p_value",
        "weighted_prediction",
        "weighted_confidence",
        "avg_kl",
        "avg_delta",
        "delta_std",
        "delta_p25",
        "delta_p50",
        "delta_p75",
        "avg_entropy",
        "avg_p_green_mass",
        "gate_pass_rate",
        "avg_candidate_size",
        "candidate_top_p",
        "kl_epsilon",
    ]:
        row.setdefault(field, "")
    return row


def write_results(rows, output_csv: str):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "method",
        "delta",
        "delta_min",
        "delta_max",
        "prompt",
        "generated_text",
        "token_ids",
        "tokens_counted",
        "num_green_tokens",
        "green_fraction",
        "z_score",
        "p_value",
        "prediction",
        "confidence",
        "word_count",
        "distinct_1",
        "distinct_2",
        "repetition_rate",
        "avg_step_delta",
        "observed_delta_min",
        "observed_delta_max",
        "weighted_z_score",
        "winmax_weighted_z_score",
        "weighted_p_value",
        "weighted_prediction",
        "weighted_confidence",
        "avg_kl",
        "avg_delta",
        "delta_std",
        "delta_p25",
        "delta_p50",
        "delta_p75",
        "avg_entropy",
        "avg_p_green_mass",
        "gate_pass_rate",
        "avg_candidate_size",
        "candidate_top_p",
        "kl_epsilon",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    model, tokenizer, device, is_decoder_only = load_model_and_tokenizer(args)
    prompts = load_prompts(args, tokenizer)
    vocab_ids = list(tokenizer.get_vocab().values())
    detector = build_detector(args, tokenizer, device, vocab_ids)
    cakl_weighted_detector = None
    cakl_cg_weighted_detector = None
    if args.run_cakl and args.use_model_assisted_detector and is_decoder_only:
        cakl_weighted_detector = build_model_assisted_detector(
            args,
            model,
            tokenizer,
            device,
            vocab_ids,
            use_candidate_greenlist=False,
            use_confidence_gate=False,
        )
        cakl_cg_weighted_detector = build_model_assisted_detector(
            args,
            model,
            tokenizer,
            device,
            vocab_ids,
            use_candidate_greenlist=True,
            use_confidence_gate=True,
        )

    results = []
    for prompt_id, prompt in enumerate(prompts, start=1):
        if args.run_baseline:
            print(f"[Prompt {prompt_id}/{len(prompts)}] Generating baseline outputs...")

            plain_text, plain_token_ids, plain_prompt_ids = generate_completion(
                prompt,
                args,
                model,
                tokenizer,
                device,
                is_decoder_only,
                return_prompt_ids=True,
            )
            plain_detection = detect_text(detector, plain_token_ids)
            plain_weighted_detection = detect_text_model_assisted(cakl_cg_weighted_detector, plain_prompt_ids, plain_token_ids)
            plain_metrics = compute_text_metrics(plain_text)
            results.append(
                build_row(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    method="No Watermark",
                    delta="",
                    delta_min="",
                    delta_max="",
                    generated_text=plain_text,
                    token_ids=plain_token_ids,
                    detection_result=plain_detection,
                    text_metrics=plain_metrics,
                    extra_metrics=plain_weighted_detection,
                )
            )

        if args.run_fixed:
            for delta in args.fixed_deltas:
                print(f"[Prompt {prompt_id}/{len(prompts)}] Fixed delta={delta}")
                processor = WatermarkLogitsProcessor(
                    vocab=vocab_ids,
                    gamma=args.gamma,
                    delta=delta,
                    seeding_scheme=args.seeding_scheme,
                    select_green_tokens=args.select_green_tokens,
                )
                generated_text, token_ids, prompt_token_ids = generate_completion(
                    prompt,
                    args,
                    model,
                    tokenizer,
                    device,
                    is_decoder_only,
                    logits_processor=processor,
                    return_prompt_ids=True,
                )
                detection_result = detect_text(detector, token_ids)
                weighted_detection = detect_text_model_assisted(cakl_cg_weighted_detector, prompt_token_ids, token_ids)
                text_metrics = compute_text_metrics(generated_text)
                results.append(
                    build_row(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        method=f"Fixed Delta {delta}",
                        delta=delta,
                        delta_min="",
                        delta_max="",
                        generated_text=generated_text,
                        token_ids=token_ids,
                        detection_result=detection_result,
                        text_metrics=text_metrics,
                        extra_metrics={
                            "avg_step_delta": delta,
                            "observed_delta_min": delta,
                            "observed_delta_max": delta,
                            **weighted_detection,
                        },
                    )
                )

        if args.run_adaptive:
            print(f"[Prompt {prompt_id}/{len(prompts)}] Adaptive delta")
            adaptive_processor = AdaptiveDeltaWatermarkLogitsProcessor(
                vocab=vocab_ids,
                gamma=args.gamma,
                delta=args.adaptive_delta_max,
                delta_min=args.adaptive_delta_min,
                delta_max=args.adaptive_delta_max,
                entropy_floor=args.adaptive_entropy_floor,
                delta_exponent=args.adaptive_delta_exponent,
                seeding_scheme=args.seeding_scheme,
                select_green_tokens=args.select_green_tokens,
            )
            adaptive_text, adaptive_token_ids, adaptive_prompt_ids = generate_completion(
                prompt,
                args,
                model,
                tokenizer,
                device,
                is_decoder_only,
                logits_processor=adaptive_processor,
                return_prompt_ids=True,
            )
            adaptive_detection = detect_text(detector, adaptive_token_ids)
            adaptive_weighted_detection = detect_text_model_assisted(cakl_cg_weighted_detector, adaptive_prompt_ids, adaptive_token_ids)
            adaptive_metrics = compute_text_metrics(adaptive_text)
            adaptive_extra = adaptive_processor.get_delta_summary()
            adaptive_extra.update(adaptive_weighted_detection)
            results.append(
                build_row(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    method="Current Adaptive Delta",
                    delta="adaptive",
                    delta_min=args.adaptive_delta_min,
                    delta_max=args.adaptive_delta_max,
                    generated_text=adaptive_text,
                    token_ids=adaptive_token_ids,
                    detection_result=adaptive_detection,
                    text_metrics=adaptive_metrics,
                    extra_metrics=adaptive_extra,
                )
            )

        if args.run_cakl:
            print(f"[Prompt {prompt_id}/{len(prompts)}] CA-KL")
            cakl_processor = CAKLWatermarkLogitsProcessor(
                vocab=vocab_ids,
                gamma=args.gamma,
                delta=args.cakl_delta_max,
                kl_epsilon=args.kl_epsilon,
                delta_max=args.cakl_delta_max,
                candidate_top_p=args.candidate_top_p,
                use_candidate_greenlist=False,
                use_confidence_gate=False,
                entropy_threshold=args.confidence_entropy_threshold,
                top1_threshold=args.confidence_top1_threshold,
                seeding_scheme=args.seeding_scheme,
                select_green_tokens=args.select_green_tokens,
            )
            cakl_text, cakl_token_ids, cakl_prompt_ids = generate_completion(
                prompt,
                args,
                model,
                tokenizer,
                device,
                is_decoder_only,
                logits_processor=cakl_processor,
                return_prompt_ids=True,
            )
            cakl_detection = detect_text(detector, cakl_token_ids)
            cakl_weighted_detection = detect_text_model_assisted(cakl_weighted_detector, cakl_prompt_ids, cakl_token_ids)
            cakl_metrics = compute_text_metrics(cakl_text)
            cakl_summary = cakl_processor.get_generation_summary()
            results.append(
                build_row(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    method="CA-KL",
                    delta="kl_adaptive",
                    delta_min="",
                    delta_max=args.cakl_delta_max,
                    generated_text=cakl_text,
                    token_ids=cakl_token_ids,
                    detection_result=cakl_detection,
                    text_metrics=cakl_metrics,
                    extra_metrics=cakl_summary,
                )
            )
            cakl_weighted_summary = dict(cakl_summary)
            cakl_weighted_summary.update(cakl_weighted_detection)
            results.append(
                build_row(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    method="CA-KL + Weighted Detector",
                    delta="kl_adaptive",
                    delta_min="",
                    delta_max=args.cakl_delta_max,
                    generated_text=cakl_text,
                    token_ids=cakl_token_ids,
                    detection_result=cakl_detection,
                    text_metrics=cakl_metrics,
                    extra_metrics=cakl_weighted_summary,
                )
            )

            print(f"[Prompt {prompt_id}/{len(prompts)}] CA-KL + Candidate Greenlist")
            candidate_processor = CAKLWatermarkLogitsProcessor(
                vocab=vocab_ids,
                gamma=args.gamma,
                delta=args.cakl_delta_max,
                kl_epsilon=args.kl_epsilon,
                delta_max=args.cakl_delta_max,
                candidate_top_p=args.candidate_top_p,
                use_candidate_greenlist=True,
                use_confidence_gate=False,
                entropy_threshold=args.confidence_entropy_threshold,
                top1_threshold=args.confidence_top1_threshold,
                seeding_scheme=args.seeding_scheme,
                select_green_tokens=args.select_green_tokens,
            )
            candidate_text, candidate_token_ids, candidate_prompt_ids = generate_completion(
                prompt,
                args,
                model,
                tokenizer,
                device,
                is_decoder_only,
                logits_processor=candidate_processor,
                return_prompt_ids=True,
            )
            candidate_detection = detect_text(detector, candidate_token_ids)
            candidate_weighted_detection = detect_text_model_assisted(cakl_cg_weighted_detector, candidate_prompt_ids, candidate_token_ids)
            candidate_metrics = compute_text_metrics(candidate_text)
            candidate_summary = candidate_processor.get_generation_summary()
            candidate_summary.update(candidate_weighted_detection)
            results.append(
                build_row(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    method="CA-KL + Candidate Greenlist",
                    delta="kl_adaptive",
                    delta_min="",
                    delta_max=args.cakl_delta_max,
                    generated_text=candidate_text,
                    token_ids=candidate_token_ids,
                    detection_result=candidate_detection,
                    text_metrics=candidate_metrics,
                    extra_metrics=candidate_summary,
                )
            )

            print(f"[Prompt {prompt_id}/{len(prompts)}] CA-KL + Candidate Greenlist + Weighted/WinMax")
            full_processor = CAKLWatermarkLogitsProcessor(
                vocab=vocab_ids,
                gamma=args.gamma,
                delta=args.cakl_delta_max,
                kl_epsilon=args.kl_epsilon,
                delta_max=args.cakl_delta_max,
                candidate_top_p=args.candidate_top_p,
                use_candidate_greenlist=True,
                use_confidence_gate=True,
                entropy_threshold=args.confidence_entropy_threshold,
                top1_threshold=args.confidence_top1_threshold,
                seeding_scheme=args.seeding_scheme,
                select_green_tokens=args.select_green_tokens,
            )
            full_text, full_token_ids, full_prompt_ids = generate_completion(
                prompt,
                args,
                model,
                tokenizer,
                device,
                is_decoder_only,
                logits_processor=full_processor,
                return_prompt_ids=True,
            )
            full_detection = detect_text(detector, full_token_ids)
            full_weighted_detection = detect_text_model_assisted(cakl_cg_weighted_detector, full_prompt_ids, full_token_ids)
            full_metrics = compute_text_metrics(full_text)
            full_summary = full_processor.get_generation_summary()
            full_summary.update(full_weighted_detection)
            results.append(
                build_row(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    method="CA-KL + Candidate Greenlist + Weighted/WinMax",
                    delta="kl_adaptive",
                    delta_min="",
                    delta_max=args.cakl_delta_max,
                    generated_text=full_text,
                    token_ids=full_token_ids,
                    detection_result=full_detection,
                    text_metrics=full_metrics,
                    extra_metrics=full_summary,
                )
            )

    write_results(results, args.output_csv)
    print(f"Saved {len(results)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
