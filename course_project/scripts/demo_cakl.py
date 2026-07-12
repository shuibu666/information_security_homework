# coding=utf-8
"""Gradio presentation demo for the CA-KL-CG watermarking method."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

import gradio as gr
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.processors.cakl_watermark_processor import CAKLWatermarkLogitsProcessor
from course_project.scripts.run_experiments import (
    build_detector,
    build_model_assisted_detector,
    compute_text_metrics,
    detect_text,
    detect_text_model_assisted,
    format_prediction,
    generate_completion,
    load_model_and_tokenizer,
    str2bool,
)
from watermark_processor import WatermarkLogitsProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="CA-KL-CG watermark demo for coursework presentation.")
    parser.add_argument("--model_name_or_path", type=str, default="./models/opt-1.3b")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--load_fp16", type=str2bool, default=False)
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7861)
    parser.add_argument("--demo_public", type=str2bool, default=False)
    parser.add_argument("--prompt_max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--target_new_tokens", type=int, default=None)
    parser.add_argument("--length_tolerance", type=int, default=0)
    parser.add_argument("--generation_seed", type=int, default=123)
    parser.add_argument("--use_sampling", type=str2bool, default=True)
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--n_beams", type=int, default=1)
    parser.add_argument("--seeding_scheme", type=str, default="simple_1")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--fixed_delta", type=float, default=2.0)
    parser.add_argument("--kl_epsilon", type=float, default=0.50)
    parser.add_argument("--cakl_delta_max", type=float, default=3.0)
    parser.add_argument("--candidate_top_p", type=float, default=0.95)
    parser.add_argument("--confidence_entropy_threshold", type=float, default=0.10)
    parser.add_argument("--confidence_top1_threshold", type=float, default=0.95)
    parser.add_argument("--window_sizes", type=str, default="20,40,80,max")
    parser.add_argument("--detection_z_threshold", type=float, default=4.0)
    parser.add_argument("--normalizers", type=str, default="")
    parser.add_argument("--ignore_repeated_bigrams", type=str2bool, default=False)
    parser.add_argument("--select_green_tokens", type=str2bool, default=True)
    return parser.parse_args()


def _fmt(value, digits=4, percent=False):
    if value is None or value == "":
        return "-"
    if percent:
        return f"{float(value):.2%}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _build_processor(args, vocab_ids, candidate_greenlist=False, confidence_gate=False):
    return CAKLWatermarkLogitsProcessor(
        vocab=vocab_ids,
        gamma=args.gamma,
        delta=args.cakl_delta_max,
        kl_epsilon=args.kl_epsilon,
        delta_max=args.cakl_delta_max,
        candidate_top_p=args.candidate_top_p,
        use_candidate_greenlist=candidate_greenlist,
        use_confidence_gate=confidence_gate,
        entropy_threshold=args.confidence_entropy_threshold,
        top1_threshold=args.confidence_top1_threshold,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
    )


def _metrics_table(standard_result, text_metrics, generation_summary=None, assisted_result=None):
    rows = [
        ["Standard prediction", format_prediction(standard_result.get("prediction", False))],
        ["Standard z-score", _fmt(standard_result.get("z_score"))],
        ["Green fraction", _fmt(standard_result.get("green_fraction"), percent=True)],
        ["Tokens counted", _fmt(standard_result.get("num_tokens_scored"))],
        ["Distinct-1", _fmt(text_metrics.get("distinct_1"), percent=True)],
        ["Repetition rate", _fmt(text_metrics.get("repetition_rate"), percent=True)],
    ]
    if generation_summary:
        rows.extend(
            [
                ["Average KL", _fmt(generation_summary.get("avg_kl"))],
                ["Average delta", _fmt(generation_summary.get("avg_delta"))],
                ["Gate pass rate", _fmt(generation_summary.get("gate_pass_rate"), percent=True)],
            ]
        )
    if assisted_result:
        rows.extend(
            [
                ["Weighted z-score", _fmt(assisted_result.get("weighted_z_score"))],
                ["WinMax z-score", _fmt(assisted_result.get("winmax_weighted_z_score"))],
                ["Weighted prediction", format_prediction(assisted_result.get("weighted_prediction", False))],
            ]
        )
    return rows


def _comparison_table(rows):
    records = []
    for row in rows:
        standard = row["standard"]
        summary = row.get("summary", {})
        assisted = row.get("assisted", {})
        records.append(
            {
                "Method": row["name"],
                "Standard z": round(float(standard.get("z_score", 0.0)), 4),
                "Green fraction": round(float(standard.get("green_fraction", 0.0)), 4),
                "Weighted z": _fmt(assisted.get("weighted_z_score")),
                "WinMax z": _fmt(assisted.get("winmax_weighted_z_score")),
                "Avg KL": _fmt(summary.get("avg_kl")),
                "Avg delta": _fmt(summary.get("avg_delta")),
                "Gate pass": _fmt(summary.get("gate_pass_rate"), percent=True),
            }
        )
    return pd.DataFrame(records)


def _highlights(rows):
    cakl = rows[2]
    full = rows[3]
    return "\n".join(
        [
            "### 单条生成结果说明",
            f"- CA-KL 的 standard z-score：`{cakl['standard'].get('z_score', 0.0):.3f}`。",
            f"- CA-KL-CG 的 standard z-score：`{full['standard'].get('z_score', 0.0):.3f}`。",
            f"- CA-KL-CG 的平均 delta：`{full['summary'].get('avg_delta', 0.0):.3f}`，gate 通过率：`{full['summary'].get('gate_pass_rate', 0.0):.1%}`。",
            "- 该页面用于展示单条生成与检测流程；报告中的正式结论以 499 条 prompt 的汇总和独立校准为准。",
        ]
    )


def run_demo(prompt, args, model, tokenizer, device, is_decoder_only):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("请输入一段英文 prompt 后再生成。")

    vocab_ids = list(tokenizer.get_vocab().values())
    standard_detector = build_detector(args, tokenizer, device, vocab_ids)
    cakl_detector = build_model_assisted_detector(
        args, model, tokenizer, device, vocab_ids, use_candidate_greenlist=False, use_confidence_gate=False
    )
    full_detector = build_model_assisted_detector(
        args, model, tokenizer, device, vocab_ids, use_candidate_greenlist=True, use_confidence_gate=True
    )

    plain_text, plain_ids, _ = generate_completion(
        prompt, args, model, tokenizer, device, is_decoder_only, return_prompt_ids=True
    )
    plain_standard = detect_text(standard_detector, plain_ids)
    plain_metrics = compute_text_metrics(plain_text)

    fixed_processor = WatermarkLogitsProcessor(
        vocab=vocab_ids,
        gamma=args.gamma,
        delta=args.fixed_delta,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
    )
    fixed_text, fixed_ids, _ = generate_completion(
        prompt, args, model, tokenizer, device, is_decoder_only, logits_processor=fixed_processor, return_prompt_ids=True
    )
    fixed_standard = detect_text(standard_detector, fixed_ids)
    fixed_metrics = compute_text_metrics(fixed_text)

    cakl_processor = _build_processor(args, vocab_ids)
    cakl_text, cakl_ids, cakl_prompt_ids = generate_completion(
        prompt, args, model, tokenizer, device, is_decoder_only, logits_processor=cakl_processor, return_prompt_ids=True
    )
    cakl_standard = detect_text(standard_detector, cakl_ids)
    cakl_metrics = compute_text_metrics(cakl_text)
    cakl_summary = cakl_processor.get_generation_summary()
    cakl_assisted = detect_text_model_assisted(cakl_detector, cakl_prompt_ids, cakl_ids)

    full_processor = _build_processor(args, vocab_ids, candidate_greenlist=True, confidence_gate=True)
    full_text, full_ids, full_prompt_ids = generate_completion(
        prompt, args, model, tokenizer, device, is_decoder_only, logits_processor=full_processor, return_prompt_ids=True
    )
    full_standard = detect_text(standard_detector, full_ids)
    full_metrics = compute_text_metrics(full_text)
    full_summary = full_processor.get_generation_summary()
    full_assisted = detect_text_model_assisted(full_detector, full_prompt_ids, full_ids)

    rows = [
        {"name": "No Watermark", "standard": plain_standard, "text_metrics": plain_metrics},
        {"name": f"Fixed Delta {args.fixed_delta:.1f}", "standard": fixed_standard, "text_metrics": fixed_metrics},
        {"name": "CA-KL", "standard": cakl_standard, "text_metrics": cakl_metrics, "summary": cakl_summary, "assisted": cakl_assisted},
        {"name": "CA-KL-CG", "standard": full_standard, "text_metrics": full_metrics, "summary": full_summary, "assisted": full_assisted},
    ]

    return (
        plain_text,
        _metrics_table(plain_standard, plain_metrics),
        fixed_text,
        _metrics_table(fixed_standard, fixed_metrics),
        cakl_text,
        _metrics_table(cakl_standard, cakl_metrics, cakl_summary, cakl_assisted),
        full_text,
        _metrics_table(full_standard, full_metrics, full_summary, full_assisted),
        _comparison_table(rows),
        _highlights(rows),
        args,
    )


def run_gradio(args, model, tokenizer, device, is_decoder_only):
    demo_fn = partial(run_demo, model=model, tokenizer=tokenizer, device=device, is_decoder_only=is_decoder_only)
    default_prompt = "Explain why provenance signals are useful for identifying AI-generated text."

    with gr.Blocks(title="CA-KL-CG Text Watermark Demo") as demo:
        session_args = gr.State(value=args)
        gr.HTML(
            """
            <div class="hero">
              <h1>CA-KL-CG 文本水印演示</h1>
              <p>同一 prompt 下比较无水印、固定偏置、KL 约束水印和完整 CA-KL-CG，并展示标准与匹配检测结果。</p>
            </div>
            """
        )
        gr.Markdown(f"当前基础模型：`{args.model_name_or_path}`")

        with gr.Tab("生成与检测"):
            with gr.Row():
                prompt = gr.Textbox(label="英文 Prompt", value=default_prompt, lines=5, scale=5)
                generate_button = gr.Button("生成并比较", variant="primary", scale=1)

            with gr.Row():
                with gr.Column(elem_classes="method-card"):
                    plain_output = gr.Textbox(label="无水印输出", lines=9, interactive=False)
                    plain_metrics = gr.Dataframe(headers=["Metric", "Value"], column_count=2, interactive=False)
                with gr.Column(elem_classes="method-card"):
                    fixed_output = gr.Textbox(label="Fixed Delta 输出", lines=9, interactive=False)
                    fixed_metrics = gr.Dataframe(headers=["Metric", "Value"], column_count=2, interactive=False)
            with gr.Row():
                with gr.Column(elem_classes="method-card"):
                    cakl_output = gr.Textbox(label="CA-KL 输出", lines=9, interactive=False)
                    cakl_metrics = gr.Dataframe(headers=["Metric", "Value"], column_count=2, interactive=False)
                with gr.Column(elem_classes="method-card"):
                    full_output = gr.Textbox(label="CA-KL-CG 输出（Candidate + Gate）", lines=9, interactive=False)
                    full_metrics = gr.Dataframe(headers=["Metric", "Value"], column_count=2, interactive=False)

            comparison = gr.Dataframe(label="即时检测对比", interactive=False, wrap=True)
            highlights = gr.Markdown("点击“生成并比较”后展示单条文本的检测与扰动统计量。")

        with gr.Accordion("参数设置", open=False):
            with gr.Row():
                gamma = gr.Slider(label="Greenlist ratio gamma", minimum=0.1, maximum=0.5, step=0.05, value=args.gamma)
                fixed_delta = gr.Slider(label="Fixed Delta", minimum=0.0, maximum=4.0, step=0.1, value=args.fixed_delta)
                kl_epsilon = gr.Slider(label="KL budget epsilon", minimum=0.05, maximum=0.8, step=0.05, value=args.kl_epsilon)
                cakl_delta_max = gr.Slider(label="CA-KL delta max", minimum=0.5, maximum=5.0, step=0.1, value=args.cakl_delta_max)
            with gr.Row():
                candidate_top_p = gr.Slider(label="Candidate top-p", minimum=0.5, maximum=1.0, step=0.05, value=args.candidate_top_p)
                entropy_threshold = gr.Slider(label="Gate entropy lower bound", minimum=0.0, maximum=0.6, step=0.05, value=args.confidence_entropy_threshold)
                top1_threshold = gr.Slider(label="Gate top-1 upper bound", minimum=0.5, maximum=1.0, step=0.05, value=args.confidence_top1_threshold)
                max_new_tokens = gr.Slider(label="Maximum new tokens", minimum=20, maximum=160, step=10, value=args.max_new_tokens)
            with gr.Row():
                sampling_temp = gr.Slider(label="Sampling temperature", minimum=0.1, maximum=1.2, step=0.1, value=args.sampling_temp)
                generation_seed = gr.Number(label="Generation seed", value=args.generation_seed, precision=0)
                z_threshold = gr.Slider(label="Detection z threshold", minimum=1.0, maximum=8.0, step=0.5, value=args.detection_z_threshold)

        def update_attr(session_args, attr, value):
            setattr(session_args, attr, int(value) if attr in {"generation_seed", "max_new_tokens"} else float(value))
            return session_args

        for component, attr in [
            (gamma, "gamma"),
            (fixed_delta, "fixed_delta"),
            (kl_epsilon, "kl_epsilon"),
            (cakl_delta_max, "cakl_delta_max"),
            (candidate_top_p, "candidate_top_p"),
            (entropy_threshold, "confidence_entropy_threshold"),
            (top1_threshold, "confidence_top1_threshold"),
            (max_new_tokens, "max_new_tokens"),
            (sampling_temp, "sampling_temp"),
            (generation_seed, "generation_seed"),
            (z_threshold, "detection_z_threshold"),
        ]:
            component.change(partial(update_attr, attr=attr), inputs=[session_args, component], outputs=[session_args])

        generate_button.click(
            fn=demo_fn,
            inputs=[prompt, session_args],
            outputs=[
                plain_output,
                plain_metrics,
                fixed_output,
                fixed_metrics,
                cakl_output,
                cakl_metrics,
                full_output,
                full_metrics,
                comparison,
                highlights,
                session_args,
            ],
        )

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.demo_public,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
        css="""
        .hero {padding: 18px 22px; border-radius: 16px; background: linear-gradient(120deg, #e0f2fe, #f0fdfa);}
        .hero h1 {margin: 0 0 8px 0; color: #0f3d56;}
        .hero p {margin: 0; color: #164e63;}
        .method-card {border: 1px solid #cbd5e1; border-radius: 12px; padding: 8px; background: #ffffff;}
        """,
    )


def main(args):
    model, tokenizer, device, is_decoder_only = load_model_and_tokenizer(args)
    run_gradio(args, model, tokenizer, device, is_decoder_only)


if __name__ == "__main__":
    main(parse_args())
