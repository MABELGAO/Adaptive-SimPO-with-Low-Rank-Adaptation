from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CHART_WIDTH = 1280
CHART_HEIGHT = 720
PANEL_MARGIN = 64
PLOT_MARGIN_LEFT = 84
PLOT_MARGIN_RIGHT = 32
PLOT_MARGIN_TOP = 56
PLOT_MARGIN_BOTTOM = 64
GRID_COLOR = "#D8DEE6"
AXIS_COLOR = "#2D3748"
TEXT_COLOR = "#1F2937"
BG_COLOR = "#FBFCFE"


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(["DejaVuSans-Bold.ttf", "Arial Bold.ttf"])
    candidates.extend(["DejaVuSans.ttf", "Arial.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = _load_font(28, bold=True)
FONT_LABEL = _load_font(18, bold=True)
FONT_AXIS = _load_font(16)
FONT_LEGEND = _load_font(16)
FONT_NOTE = _load_font(14)


def _nice_step(low: float, high: float, ticks: int = 5) -> float:
    span = max(high - low, 1e-9)
    raw = span / max(ticks, 1)
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10 ** exponent)
    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3:
        nice_fraction = 2
    elif fraction < 7:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def _ticks(low: float, high: float, ticks: int = 5) -> list[float]:
    if math.isclose(low, high):
        return [low]
    step = _nice_step(low, high, ticks=ticks)
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    out = []
    value = start
    while value <= end + step * 0.5:
        out.append(value)
        value += step
    return out


def _format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000 or (abs_value > 0 and abs_value < 0.01):
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _draw_legend(draw: ImageDraw.ImageDraw, items: Sequence[dict], x: int, y: int) -> None:
    line_len = 24
    gap = 18
    current_y = y
    for item in items:
        draw.line((x, current_y + 8, x + line_len, current_y + 8), fill=item["color"], width=4)
        draw.text((x + line_len + 10, current_y), item["label"], fill=TEXT_COLOR, font=FONT_LEGEND)
        current_y += gap


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[dict],
    *,
    note: str | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill="white", outline="#E5EAF1", width=2)
    draw.text((left + 24, top + 18), title, fill=TEXT_COLOR, font=FONT_TITLE)

    plot_left = left + PLOT_MARGIN_LEFT
    plot_top = top + PLOT_MARGIN_TOP
    plot_right = right - PLOT_MARGIN_RIGHT
    plot_bottom = bottom - PLOT_MARGIN_BOTTOM

    x_values = [x for item in series for x, _ in item["points"]]
    y_values = [y for item in series for _, y in item["points"]]
    if not x_values or not y_values:
        draw.text((plot_left, plot_top), "No data", fill=TEXT_COLOR, font=FONT_LABEL)
        return

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if math.isclose(x_min, x_max):
        x_max += 1
    if math.isclose(y_min, y_max):
        padding = max(abs(y_min) * 0.1, 1.0)
        y_min -= padding
        y_max += padding
    else:
        padding = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= padding
        y_max += padding

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)
        py = plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        return px, py

    for value in _ticks(y_min, y_max):
        if value < y_min - 1e-9 or value > y_max + 1e-9:
            continue
        _, py = project((x_min, value))
        draw.line((plot_left, py, plot_right, py), fill=GRID_COLOR, width=1)
        label = _format_number(value)
        label_w, label_h = _text_size(draw, label, FONT_AXIS)
        draw.text((plot_left - 12 - label_w, py - label_h / 2), label, fill=TEXT_COLOR, font=FONT_AXIS)

    for value in _ticks(x_min, x_max):
        if value < x_min - 1e-9 or value > x_max + 1e-9:
            continue
        px, _ = project((value, y_min))
        draw.line((px, plot_top, px, plot_bottom), fill=GRID_COLOR, width=1)
        label = f"{int(round(value))}"
        label_w, _ = _text_size(draw, label, FONT_AXIS)
        draw.text((px - label_w / 2, plot_bottom + 10), label, fill=TEXT_COLOR, font=FONT_AXIS)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS_COLOR, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS_COLOR, width=2)

    for item in series:
        points = [project(point) for point in item["points"]]
        if len(points) >= 2:
            draw.line(points, fill=item["color"], width=item.get("width", 4), joint="curve")
        for px, py in points:
            radius = item.get("radius", 3)
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=item["color"], outline=item["color"])

    x_label_w, _ = _text_size(draw, x_label, FONT_LABEL)
    draw.text(((plot_left + plot_right - x_label_w) / 2, bottom - 40), x_label, fill=TEXT_COLOR, font=FONT_LABEL)
    draw.text((left + 24, plot_top - 8), y_label, fill=TEXT_COLOR, font=FONT_LABEL)
    _draw_legend(draw, series, plot_right - 220, top + 20)

    if note:
        draw.text((left + 24, bottom - 24), note, fill="#4B5563", font=FONT_NOTE)


def _make_panel_image(title: str, y_label: str, series: Sequence[dict], note: str | None = None) -> Image.Image:
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)
    _draw_panel(
        draw,
        (PANEL_MARGIN, PANEL_MARGIN, CHART_WIDTH - PANEL_MARGIN, CHART_HEIGHT - PANEL_MARGIN),
        title=title,
        x_label="Training Step",
        y_label=y_label,
        series=series,
        note=note,
    )
    return image


def _make_two_panel_image(
    title: str,
    top_title: str,
    top_y_label: str,
    top_series: Sequence[dict],
    bottom_title: str,
    bottom_y_label: str,
    bottom_series: Sequence[dict],
    *,
    note: str | None = None,
) -> Image.Image:
    image = Image.new("RGB", (CHART_WIDTH, 1180), BG_COLOR)
    draw = ImageDraw.Draw(image)
    draw.text((PANEL_MARGIN, 20), title, fill=TEXT_COLOR, font=FONT_TITLE)
    _draw_panel(
        draw,
        (PANEL_MARGIN, 64, CHART_WIDTH - PANEL_MARGIN, 560),
        title=top_title,
        x_label="Training Step",
        y_label=top_y_label,
        series=top_series,
    )
    _draw_panel(
        draw,
        (PANEL_MARGIN, 620, CHART_WIDTH - PANEL_MARGIN, 1116),
        title=bottom_title,
        x_label="Training Step",
        y_label=bottom_y_label,
        series=bottom_series,
        note=note,
    )
    return image


def _load_event_scalars(event_file: Path) -> dict[str, list[tuple[float, float]]]:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalars: dict[str, list[tuple[float, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [(event.step, event.value) for event in accumulator.Scalars(tag)]
    return scalars


def _load_log_entries(log_file: Path) -> list[dict]:
    entries: list[dict] = []
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            payload = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _extract_resource_series(log_entries: Sequence[dict], train_steps: Sequence[float]) -> dict[str, list[tuple[float, float]]]:
    train_logs = [entry for entry in log_entries if "loss" in entry and "train_loss" not in entry]
    series = {
        "perf/time_per_step_s": [],
        "perf/peak_gpu_memory_gb": [],
    }
    limit = min(len(train_logs), len(train_steps))
    for idx in range(limit):
        step = train_steps[idx]
        entry = train_logs[idx]
        if "perf/time_per_step_s" in entry:
            series["perf/time_per_step_s"].append((step, float(entry["perf/time_per_step_s"])))
        if "perf/peak_gpu_memory_gb" in entry:
            series["perf/peak_gpu_memory_gb"].append((step, float(entry["perf/peak_gpu_memory_gb"])))
    return series


def _write_summary(output_dir: Path, event_scalars: dict[str, list[tuple[float, float]]], log_entries: Sequence[dict]) -> None:
    summary = {}
    for tag in (
        "train/loss",
        "eval/loss",
        "train/rewards/margins",
        "eval/rewards/margins",
        "train/rewards/accuracies",
        "eval/rewards/accuracies",
        "train/objective/kl",
        "eval/objective/kl",
        "train/objective/kl_seq",
        "eval/objective/kl_seq",
        "train/objective/gamma",
        "eval/objective/gamma",
    ):
        values = event_scalars.get(tag, [])
        if values:
            summary[tag] = {"step": values[-1][0], "value": values[-1][1]}

    for entry in reversed(log_entries):
        if "train_runtime" in entry:
            summary["train_runtime_s"] = entry["train_runtime"]
            summary["train_loss_final"] = entry["train_loss"]
            summary["train_steps_per_second"] = entry["train_steps_per_second"]
            summary["peak_gpu_memory_gb"] = entry.get("perf/peak_gpu_memory_gb")
            break

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def export_plots(event_file: Path, log_file: Path | None, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_scalars = _load_event_scalars(event_file)
    log_entries = _load_log_entries(log_file) if log_file else []

    loss_image = _make_panel_image(
        "Loss Curve",
        "Loss",
        [
            {"label": "Train Loss", "color": "#1D4ED8", "points": event_scalars["train/loss"]},
            {"label": "Eval Loss", "color": "#DC2626", "points": event_scalars["eval/loss"], "radius": 5},
        ],
        note="Eval loss is logged every 100 steps; train loss is logged every 5 steps.",
    )
    loss_image.save(output_dir / "loss_curve.png")

    reward_image = _make_two_panel_image(
        "Reward Dynamics",
        "Reward Margin",
        "Margin",
        [
            {"label": "Train Margin", "color": "#7C3AED", "points": event_scalars["train/rewards/margins"]},
            {"label": "Eval Margin", "color": "#EA580C", "points": event_scalars["eval/rewards/margins"], "radius": 5},
        ],
        "Reward Accuracy",
        "Accuracy",
        [
            {"label": "Train Accuracy", "color": "#059669", "points": event_scalars["train/rewards/accuracies"]},
            {"label": "Eval Accuracy", "color": "#B91C1C", "points": event_scalars["eval/rewards/accuracies"], "radius": 5},
        ],
        note="Adaptive Margin SimPO tracks both preference separation and how often chosen responses score above rejected ones.",
    )
    reward_image.save(output_dir / "reward_metrics.png")

    response_image = _make_two_panel_image(
        "Response Length Behavior",
        "Chosen vs Rejected Length",
        "Tokens",
        [
            {"label": "Chosen", "color": "#2563EB", "points": event_scalars["train/response_lengths/chosen"]},
            {"label": "Rejected", "color": "#DB2777", "points": event_scalars["train/response_lengths/rejected"]},
        ],
        "Length Difference",
        "Chosen - Rejected",
        [
            {"label": "Length Diff", "color": "#0F766E", "points": event_scalars["train/response_lengths/diff"]},
        ],
        note="These curves help inspect whether the policy is developing a response-length bias during preference optimization.",
    )
    response_image.save(output_dir / "response_length_metrics.png")

    gamma_image = _make_two_panel_image(
        "Adaptive Margin Behavior",
        "Adaptive Gamma",
        "Gamma",
        [
            {"label": "Train Gamma", "color": "#4F46E5", "points": event_scalars["train/objective/gamma"]},
            {"label": "Eval Gamma", "color": "#C2410C", "points": event_scalars["eval/objective/gamma"], "radius": 5},
        ],
        "Normalized Score Gap",
        "Score Gap",
        [
            {"label": "Train Score Gap", "color": "#0891B2", "points": event_scalars["train/objective/score_gap"]},
            {"label": "Eval Score Gap", "color": "#BE123C", "points": event_scalars["eval/objective/score_gap"], "radius": 5},
        ],
        note="The dynamic SimPO margin is driven by the normalized chosen/rejected rating gap from the preference dataset.",
    )
    gamma_image.save(output_dir / "adaptive_margin_metrics.png")

    if "train/objective/kl" in event_scalars and "eval/objective/kl" in event_scalars:
        kl_image = _make_two_panel_image(
            "Reference KL Metrics",
            "Token-Normalized KL",
            "KL",
            [
                {"label": "Train KL", "color": "#4338CA", "points": event_scalars["train/objective/kl"]},
                {"label": "Eval KL", "color": "#D97706", "points": event_scalars["eval/objective/kl"], "radius": 5},
            ],
            "Sequence-Level KL",
            "KL",
            [
                {"label": "Train KL (Seq)", "color": "#0F766E", "points": event_scalars.get("train/objective/kl_seq", [])},
                {"label": "Eval KL (Seq)", "color": "#B91C1C", "points": event_scalars.get("eval/objective/kl_seq", []), "radius": 5},
            ],
            note="These curves are available when the dataset provides precomputed reference log-probs.",
        )
        kl_image.save(output_dir / "kl_metrics.png")

    train_steps = [step for step, _ in event_scalars["train/loss"]]
    resource_series = _extract_resource_series(log_entries, train_steps)
    if resource_series["perf/time_per_step_s"] or resource_series["perf/peak_gpu_memory_gb"]:
        resource_image = _make_two_panel_image(
            "Runtime and Memory",
            "Time Per Step",
            "Seconds",
            [
                {"label": "Step Time", "color": "#9333EA", "points": resource_series["perf/time_per_step_s"]},
            ],
            "Peak GPU Memory",
            "GB",
            [
                {"label": "Peak GPU Memory", "color": "#B45309", "points": resource_series["perf/peak_gpu_memory_gb"]},
            ],
            note="Logged from the custom training callbacks during the single-GPU run on an RTX 5090.",
        )
        resource_image.save(output_dir / "runtime_memory_metrics.png")

    _write_summary(output_dir, event_scalars, log_entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export important TensorBoard metrics to PNG plots.")
    parser.add_argument("--event-file", type=Path, required=True, help="Path to the TensorBoard event file.")
    parser.add_argument("--log-file", type=Path, help="Path to the raw training log file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated PNG files.")
    args = parser.parse_args()

    export_plots(args.event_file, args.log_file, args.output_dir)


if __name__ == "__main__":
    main()
