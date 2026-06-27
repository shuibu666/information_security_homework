# coding=utf-8
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

from course_project.scripts.run_experiments import (
    build_detector,
    compute_text_metrics,
    detect_text,
    format_prediction,
    generate_completion,
    load_model_and_tokenizer,
    read_prompts,
    str2bool,
)
from course_project.processors.adaptive_watermark_processor import AdaptiveDeltaWatermarkLogitsProcessor
from watermark_processor import WatermarkLogitsProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive delta watermark demo for coursework presentation.")
    parser.add_argument("--run_gradio", type=str2bool, default=True)
    parser.add_argument("--demo_public", type=str2bool, default=False)
    parser.add_argument("--model_name_or_path", type=str, default="./models/opt-125m")
    parser.add_argument("--prompts_path", type=str, default="course_project/data/prompts.txt")
    parser.add_argument("--prompt_max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--generation_seed", type=int, default=123)
    parser.add_argument("--use_sampling", type=str2bool, default=True)
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--n_beams", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--load_fp16", type=str2bool, default=False)
    parser.add_argument("--seed_separately", type=str2bool, default=True)
    parser.add_argument("--seeding_scheme", type=str, default="simple_1")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--adaptive_delta_min", type=float, default=0.5)
    parser.add_argument("--adaptive_delta_max", type=float, default=3.0)
    parser.add_argument("--detection_z_threshold", type=float, default=4.0)
    parser.add_argument("--normalizers", type=str, default="")
    parser.add_argument("--ignore_repeated_bigrams", type=str2bool, default=False)
    parser.add_argument("--select_green_tokens", type=str2bool, default=True)
    parser.add_argument("--skip_model_load", type=str2bool, default=False)
    return parser.parse_args()


def build_metrics_table(detection_result: dict, text_metrics: dict, extra_rows: list[list[str]] | None = None):
    def fmt(value, as_percent=False):
        if value is None:
            return "-"
        if as_percent:
            return f"{value:.2%}"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    rows = [
        ["Prediction", format_prediction(detection_result.get("prediction", False))],
        ["z-score", fmt(detection_result.get("z_score"))],
        ["p value", fmt(detection_result.get("p_value"))],
        ["Green Fraction", fmt(detection_result.get("green_fraction"), as_percent=True)],
        ["Tokens Counted", fmt(detection_result.get("num_tokens_scored"))],
        ["Green Tokens", fmt(detection_result.get("num_green_tokens"))],
        ["Confidence", fmt(detection_result.get("confidence"), as_percent=True)],
        ["Word Count", fmt(text_metrics.get("word_count"))],
        ["Distinct-1", fmt(text_metrics.get("distinct_1"), as_percent=True)],
        ["Distinct-2", fmt(text_metrics.get("distinct_2"), as_percent=True)],
        ["Repetition Rate", fmt(text_metrics.get("repetition_rate"), as_percent=True)],
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return rows


def build_comparison_dataframe(method_rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in method_rows:
        records.append(
            {
                "Method": row["method"],
                "Prediction": format_prediction(row["detection"].get("prediction", False)),
                "z-score": round(row["detection"].get("z_score", 0.0), 4),
                "Green Fraction": round(row["detection"].get("green_fraction", 0.0), 4),
                "Distinct-1": round(row["text_metrics"].get("distinct_1", 0.0), 4),
                "Distinct-2": round(row["text_metrics"].get("distinct_2", 0.0), 4),
                "Repetition Rate": round(row["text_metrics"].get("repetition_rate", 0.0), 4),
            }
        )
    return pd.DataFrame(records)


def build_story_markdown(method_rows: list[dict], adaptive_summary: dict) -> str:
    no_wm = method_rows[0]["detection"]
    fixed = method_rows[1]["detection"]
    adaptive = method_rows[2]["detection"]
    avg_step_delta = adaptive_summary.get("avg_step_delta")
    observed_min = adaptive_summary.get("observed_delta_min")
    observed_max = adaptive_summary.get("observed_delta_max")
    lines = [
        "### Comparison Highlights",
        f"- Fixed delta baseline z-score: `{fixed.get('z_score', 0.0):.3f}`",
        f"- Adaptive delta z-score: `{adaptive.get('z_score', 0.0):.3f}`",
        f"- No watermark z-score: `{no_wm.get('z_score', 0.0):.3f}`",
        f"- Adaptive average step delta: `{avg_step_delta:.4g}`" if avg_step_delta is not None else "- Adaptive average step delta: `-`",
        (
            f"- Adaptive observed range: `{observed_min:.4g} ~ {observed_max:.4g}`"
            if (observed_min is not None and observed_max is not None)
            else "- Adaptive observed range: `-`"
        ),
    ]
    return "\n".join(lines)


def build_detector_only_table(detector, text: str):
    if not text or len(text.strip()) <= 1:
        return [["Error", "Text is too short to analyze."]]
    detection_result = detector.detect(text=text)
    text_metrics = compute_text_metrics(text)
    return build_metrics_table(detection_result, text_metrics)


def run_demo(prompt, args, model=None, tokenizer=None, device=None, is_decoder_only=None):
    vocab_ids = list(tokenizer.get_vocab().values())
    detector = build_detector(args, tokenizer, device, vocab_ids)

    plain_text, plain_token_ids = generate_completion(prompt, args, model, tokenizer, device, is_decoder_only)
    plain_detection = detect_text(detector, plain_token_ids)
    plain_metrics = compute_text_metrics(plain_text)

    fixed_processor = WatermarkLogitsProcessor(
        vocab=vocab_ids,
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
    )
    fixed_text, fixed_token_ids = generate_completion(
        prompt,
        args,
        model,
        tokenizer,
        device,
        is_decoder_only,
        logits_processor=fixed_processor,
    )
    fixed_detection = detect_text(detector, fixed_token_ids)
    fixed_metrics = compute_text_metrics(fixed_text)

    adaptive_processor = AdaptiveDeltaWatermarkLogitsProcessor(
        vocab=vocab_ids,
        gamma=args.gamma,
        delta=args.adaptive_delta_max,
        delta_min=args.adaptive_delta_min,
        delta_max=args.adaptive_delta_max,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
    )
    adaptive_text, adaptive_token_ids = generate_completion(
        prompt,
        args,
        model,
        tokenizer,
        device,
        is_decoder_only,
        logits_processor=adaptive_processor,
    )
    adaptive_detection = detect_text(detector, adaptive_token_ids)
    adaptive_metrics = compute_text_metrics(adaptive_text)
    adaptive_summary = adaptive_processor.get_delta_summary()

    method_rows = [
        {"method": "No Watermark", "detection": plain_detection, "text_metrics": plain_metrics},
        {"method": f"Fixed Delta {args.delta}", "detection": fixed_detection, "text_metrics": fixed_metrics},
        {"method": "Adaptive Delta", "detection": adaptive_detection, "text_metrics": adaptive_metrics},
    ]

    adaptive_extra_rows = [
        ["Adaptive Delta Min", f"{args.adaptive_delta_min:.3g}"],
        ["Adaptive Delta Max", f"{args.adaptive_delta_max:.3g}"],
        ["Avg Step Delta", f"{adaptive_summary['avg_step_delta']:.4g}" if adaptive_summary["avg_step_delta"] is not None else "-"],
        ["Observed Delta Min", f"{adaptive_summary['observed_delta_min']:.4g}" if adaptive_summary["observed_delta_min"] is not None else "-"],
        ["Observed Delta Max", f"{adaptive_summary['observed_delta_max']:.4g}" if adaptive_summary["observed_delta_max"] is not None else "-"],
    ]

    return (
        plain_text,
        build_metrics_table(plain_detection, plain_metrics),
        fixed_text,
        build_metrics_table(
            fixed_detection,
            fixed_metrics,
            extra_rows=[["Fixed Delta", f"{args.delta:.3g}"]],
        ),
        adaptive_text,
        build_metrics_table(adaptive_detection, adaptive_metrics, adaptive_extra_rows),
        build_comparison_dataframe(method_rows),
        build_story_markdown(method_rows, adaptive_summary),
        args,
    )


def update_decoding(session_args, value):
    session_args.use_sampling = value == "multinomial"
    return session_args


def run_gradio(args, model=None, tokenizer=None, device=None, is_decoder_only=None):
    demo_fn = partial(run_demo, model=model, tokenizer=tokenizer, device=device, is_decoder_only=is_decoder_only)
    prompts = read_prompts(args.prompts_path, None)
    default_prompt = prompts[0] if prompts else "Explain how watermarking can help identify AI-generated text."

    with gr.Blocks(
        title="Adaptive Delta Watermark Demo",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
        css="""
        .hero {padding: 14px 18px; border-radius: 16px; background: linear-gradient(135deg, #eff6ff, #ecfeff);}
        .hero h1 {margin-bottom: 6px;}
        .note-card {padding: 10px 14px; border-radius: 14px; background: #f8fafc;}
        """,
    ) as demo:
        session_args = gr.State(value=args)

        gr.HTML(
            """
            <div class="hero">
              <h1>Adaptive Delta Watermark Demo</h1>
              <p>Course demo built on top of the official lm-watermarking interface. This version compares
              <b>No Watermark</b>, <b>Fixed Delta</b>, and <b>Adaptive Delta</b> side by side for video presentation.</p>
            </div>
            """
        )
        gr.Markdown(f"Model: `{args.model_name_or_path}`")

        with gr.Tab("Generate and Compare"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt = gr.Textbox(label="Prompt", lines=8, max_lines=10, value=default_prompt)
                with gr.Column(scale=1):
                    prompt_picker = gr.Dropdown(label="Prompt Presets", choices=prompts, value=default_prompt)
                    generate_btn = gr.Button("Generate Comparison", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    plain_output = gr.Textbox(label="Output Without Watermark", lines=14, max_lines=18, interactive=False)
                    plain_metrics = gr.Dataframe(headers=["Metric", "Value"], row_count=11, col_count=2, interactive=False)
                with gr.Column(scale=1):
                    fixed_output = gr.Textbox(label="Output With Fixed Delta", lines=14, max_lines=18, interactive=False)
                    fixed_metrics = gr.Dataframe(headers=["Metric", "Value"], row_count=12, col_count=2, interactive=False)
                with gr.Column(scale=1):
                    adaptive_output = gr.Textbox(label="Output With Adaptive Delta", lines=14, max_lines=18, interactive=False)
                    adaptive_metrics = gr.Dataframe(headers=["Metric", "Value"], row_count=16, col_count=2, interactive=False)
            with gr.Row():
                comparison_table = gr.Dataframe(
                    label="Method Comparison Summary",
                    interactive=False,
                    wrap=True,
                )
            with gr.Row():
                comparison_story = gr.Markdown(
                    """
                    ### Comparison Highlights
                    Generate once to see how the adaptive watermark compares with the fixed-delta baseline.
                    """
                )

        with gr.Tab("Detector Only"):
            with gr.Row():
                detector_input = gr.Textbox(label="Text to Analyze", lines=12, max_lines=16)
                detector_output = gr.Dataframe(headers=["Metric", "Value"], row_count=11, col_count=2, interactive=False)
            detect_btn = gr.Button("Detect Text")

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    decoding = gr.Radio(label="Decoding Method", choices=["multinomial", "greedy"], value=("multinomial" if args.use_sampling else "greedy"))
                    sampling_temp = gr.Slider(label="Sampling Temperature", minimum=0.1, maximum=1.2, step=0.1, value=args.sampling_temp, visible=args.use_sampling)
                    generation_seed = gr.Number(label="Generation Seed", value=args.generation_seed)
                    n_beams = gr.Dropdown(label="Number of Beams", choices=list(range(1, 11)), value=args.n_beams, visible=(not args.use_sampling))
                    max_new_tokens = gr.Slider(label="Max Generated Tokens", minimum=10, maximum=300, step=10, value=args.max_new_tokens)
                with gr.Column(scale=1):
                    gamma = gr.Slider(label="gamma", minimum=0.1, maximum=0.9, step=0.05, value=args.gamma)
                    delta = gr.Slider(label="Fixed Delta", minimum=0.0, maximum=6.0, step=0.1, value=args.delta)
                    adaptive_delta_min = gr.Slider(label="Adaptive Delta Min", minimum=0.0, maximum=6.0, step=0.1, value=args.adaptive_delta_min)
                    adaptive_delta_max = gr.Slider(label="Adaptive Delta Max", minimum=0.0, maximum=6.0, step=0.1, value=args.adaptive_delta_max)
                    detection_z_threshold = gr.Slider(label="z-score Threshold", minimum=0.0, maximum=10.0, step=0.1, value=args.detection_z_threshold)
                with gr.Column(scale=1):
                    normalizers = gr.CheckboxGroup(label="Normalizations", choices=["unicode", "homoglyphs", "truecase"], value=args.normalizers.split(",") if isinstance(args.normalizers, str) else args.normalizers)
                    ignore_repeated_bigrams = gr.Checkbox(label="Ignore Bigram Repeats", value=args.ignore_repeated_bigrams)
                    select_green_tokens = gr.Checkbox(label="Select Green Tokens from Partition", value=args.select_green_tokens)
                    current_parameters = gr.Textbox(label="Current Parameters", value=str(args), lines=8)

        with gr.Accordion("Demo Notes", open=False):
            gr.Markdown(
                """
                - The left output is the plain model response without watermarking.
                - The middle output is the official fixed-delta watermark baseline.
                - The right output is the course improvement: entropy-based adaptive delta watermarking.
                - `Distinct-1`, `Distinct-2`, and `Repetition Rate` are lightweight text quality indicators added for coursework analysis.
                """
            )

        prompt_picker.change(lambda selected: selected, inputs=[prompt_picker], outputs=[prompt])

        def refresh_parameter_box(session_args):
            return str(session_args)

        def detect_only_fn(input_text, session_args):
            detector = build_detector(session_args, tokenizer, device, list(tokenizer.get_vocab().values()))
            return build_detector_only_table(detector, input_text), session_args

        def update_sampling_temp(session_args, value): session_args.sampling_temp = float(value); return session_args
        def update_generation_seed(session_args, value): session_args.generation_seed = int(value); return session_args
        def update_n_beams(session_args, value): session_args.n_beams = int(value); return session_args
        def update_max_new_tokens(session_args, value): session_args.max_new_tokens = int(value); return session_args
        def update_gamma(session_args, value): session_args.gamma = float(value); return session_args
        def update_delta(session_args, value): session_args.delta = float(value); return session_args
        def update_adaptive_delta_min(session_args, value): session_args.adaptive_delta_min = float(value); return session_args
        def update_adaptive_delta_max(session_args, value): session_args.adaptive_delta_max = float(value); return session_args
        def update_detection_threshold(session_args, value): session_args.detection_z_threshold = float(value); return session_args
        def update_normalizers(session_args, value): session_args.normalizers = value; return session_args
        def update_ignore_repeats(session_args, value): session_args.ignore_repeated_bigrams = value; return session_args
        def update_select_green_tokens(session_args, value): session_args.select_green_tokens = value; return session_args
        def toggle_sampling_vis(value):
            if value == "multinomial":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        decoding.change(toggle_sampling_vis, inputs=[decoding], outputs=[sampling_temp, n_beams])
        decoding.change(update_decoding, inputs=[session_args, decoding], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        sampling_temp.change(update_sampling_temp, inputs=[session_args, sampling_temp], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        generation_seed.change(update_generation_seed, inputs=[session_args, generation_seed], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        n_beams.change(update_n_beams, inputs=[session_args, n_beams], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        max_new_tokens.change(update_max_new_tokens, inputs=[session_args, max_new_tokens], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        gamma.change(update_gamma, inputs=[session_args, gamma], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        delta.change(update_delta, inputs=[session_args, delta], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        adaptive_delta_min.change(update_adaptive_delta_min, inputs=[session_args, adaptive_delta_min], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        adaptive_delta_max.change(update_adaptive_delta_max, inputs=[session_args, adaptive_delta_max], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        detection_z_threshold.change(update_detection_threshold, inputs=[session_args, detection_z_threshold], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        normalizers.change(update_normalizers, inputs=[session_args, normalizers], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        ignore_repeated_bigrams.change(update_ignore_repeats, inputs=[session_args, ignore_repeated_bigrams], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])
        select_green_tokens.change(update_select_green_tokens, inputs=[session_args, select_green_tokens], outputs=[session_args]).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])

        generate_btn.click(
            fn=demo_fn,
            inputs=[prompt, session_args],
            outputs=[
                plain_output,
                plain_metrics,
                fixed_output,
                fixed_metrics,
                adaptive_output,
                adaptive_metrics,
                comparison_table,
                comparison_story,
                session_args,
            ],
        ).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])

        detect_btn.click(
            fn=detect_only_fn,
            inputs=[detector_input, session_args],
            outputs=[detector_output, session_args],
        ).then(refresh_parameter_box, inputs=[session_args], outputs=[current_parameters])

    demo.queue(default_concurrency_limit=3)
    if args.demo_public:
        demo.launch(share=True)
    else:
        demo.launch()


def main(args):
    if isinstance(args.normalizers, str) and args.normalizers:
        args.normalizers = args.normalizers.split(",")
    elif isinstance(args.normalizers, str):
        args.normalizers = []

    if args.skip_model_load:
        raise ValueError("This demo requires a loaded model. Please run with --skip_model_load False.")

    model, tokenizer, device, is_decoder_only = load_model_and_tokenizer(args)
    if args.run_gradio:
        run_gradio(args, model=model, tokenizer=tokenizer, device=device, is_decoder_only=is_decoder_only)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
