# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties


OUT = Path(__file__).resolve().parent


def pick_font():
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No usable font found")


FONT_PATH = pick_font()
FONT = ImageFont.truetype(str(FONT_PATH), 24)
FONT_SMALL = ImageFont.truetype(str(FONT_PATH), 18)
FONT_BOLD = ImageFont.truetype(str(FONT_PATH), 28)


def draw_centered(draw, rect, text, font=FONT, fill="#111111", line_gap=8):
    x, y, w, h = rect
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + max(0, len(lines) - 1) * line_gap
    cy = y + (h - total_h) / 2
    for line, tw, th in zip(lines, widths, heights):
        draw.text((x + (w - tw) / 2, cy), line, fill=fill, font=font)
        cy += th + line_gap


def rounded_box(draw, x, y, w, h, text, fill, outline="#333333", width=3, font=FONT):
    draw.rounded_rectangle(
        [x, y, x + w, y + h], radius=18, fill=fill, outline=outline, width=width
    )
    draw_centered(draw, (x, y, w, h), text, font=font)


def arrow(draw, x1, y1, x2, y2, color="#333333"):
    draw.line([x1, y1, x2, y2], fill=color, width=4)
    angle = np.arctan2(y2 - y1, x2 - x1)
    length = 18
    for delta in [2.6, -2.6]:
        a = angle + delta
        draw.line(
            [x2, y2, x2 - length * np.cos(a), y2 - length * np.sin(a)],
            fill=color,
            width=4,
        )


def framework_diagram():
    w, h = 1800, 950
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    title = "YOLO-FD 两阶段训练与推理校准算法框架"
    title_bbox = draw.textbbox((0, 0), title, font=FONT_BOLD)
    draw.text(((w - (title_bbox[2] - title_bbox[0])) / 2, 35), title, fill="#111111", font=FONT_BOLD)

    rounded_box(draw, 70, 150, 250, 120, "鱼病图像数据\n检测框 + 病灶 mask", "#E8F1FF")
    rounded_box(draw, 410, 135, 360, 150, "YOLO-FD 原始网络\nBackbone + Neck\n检测分支 + 分割分支", "#F1F1F1")
    rounded_box(draw, 875, 120, 330, 180, "第一阶段训练\nImageNet 预训练\nAdamW + PCGrad\n不确定性多任务损失", "#FFF2CC", outline="#B8860B")
    rounded_box(draw, 1320, 120, 330, 180, "第二阶段微调\n关闭主要增强\nlr=5e-5\nbox=10, dfl=2", "#FFE6D5", outline="#CC5A00")
    rounded_box(draw, 410, 485, 360, 150, "推理阈值校准\nconf=0.05\niou=0.7", "#E6F4EA", outline="#2E7D32")
    rounded_box(draw, 875, 460, 330, 200, "输出结果\n病鱼/正常鱼类别\n鱼体检测框\n感染区域分割 mask", "#EDE7F6", outline="#6A1B9A")
    rounded_box(draw, 1320, 470, 330, 170, "评价指标\nmAP50 / mAP50-95\nmIOU / Precision / Recall", "#E3F2FD", outline="#1565C0")

    arrow(draw, 320, 210, 410, 210)
    arrow(draw, 770, 210, 875, 210)
    arrow(draw, 1205, 210, 1320, 210)
    arrow(draw, 1485, 300, 1485, 470)
    arrow(draw, 1320, 555, 1205, 555)
    arrow(draw, 875, 555, 770, 555)
    arrow(draw, 590, 485, 590, 285)

    rounded_box(
        draw,
        95,
        745,
        1560,
        90,
        "本文没有改变 YOLO-FD 网络结构；改进发生在训练配方、第二阶段无增强微调和推理后处理阈值校准。",
        "#FFF9C4",
        outline="#B8860B",
        font=FONT,
    )
    img.save(OUT / "algorithm_framework.png", quality=95)


def formula_images():
    formulas = {
        "formula_detection_loss.png": r"$L_{det}=\lambda_{box}L_{CIoU}+\lambda_{cls}L_{BCE}+\lambda_{dfl}L_{DFL}$",
        "formula_mtl_loss.png": r"$L_{MTL}=e^{-s_{det}}L_{det}+s_{det}+e^{-s_{seg}}L_{seg}+s_{seg}$",
        "formula_pcgrad.png": r"$\mathrm{if}\ g_i^Tg_j<0:\quad g_i\leftarrow g_i-\frac{g_i^Tg_j}{\|g_j\|^2}g_j$",
        "formula_iou.png": r"$IoU=\frac{|B_{pred}\cap B_{gt}|}{|B_{pred}\cup B_{gt}|}$",
    }
    for name, formula in formulas.items():
        fig = plt.figure(figsize=(8.8, 1.15), dpi=220)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.52, formula, ha="center", va="center", fontsize=24)
        fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.22, facecolor="white")
        plt.close(fig)


def metric_bar_chart():
    plt.rcParams["axes.unicode_minus"] = False
    fp = FontProperties(fname=str(FONT_PATH))
    metrics = ["mAP50", "mAP50-95", "mIOU"]
    paper = [0.942, 0.787, 0.794]
    val = [0.947, 0.796, 0.7993]
    test = [0.958, 0.803, 0.8084]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=180)
    ax.bar(x - width, paper, width, label="论文 SOTA", color="#9E9E9E")
    ax.bar(x, val, width, label="完整验证集", color="#1976D2")
    ax.bar(x + width, test, width, label="测试子集", color="#F57C00")
    ax.set_ylim(0.74, 0.98)
    ax.set_ylabel("指标值", fontproperties=fp)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("最终模型指标对比", fontproperties=fp)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for vals, offset in [(paper, -width), (val, 0), (test, width)]:
        for i, value in enumerate(vals):
            ax.text(i + offset, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(loc="lower right", prop=fp)
    fig.tight_layout()
    fig.savefig(OUT / "metric_comparison_bar.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    framework_diagram()
    formula_images()
    metric_bar_chart()
    for path in sorted(OUT.glob("*.png")):
        print(path.name, path.stat().st_size)
