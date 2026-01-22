import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import fitz  # PyMuPDF
import pandas as pd
import camelot

# unstructured 先保留 import（但我们不再依赖它产出图/表元素）
from unstructured.partition.pdf import partition_pdf

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


# ------------------------
# Config
# ------------------------
DOCS_DIR = Path("docs")
OUT_JSONL = Path("data/parsed_docs.jsonl")
IMG_DIR = Path("data/extracted_images")

VLM_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # 16GB 显存：2B 更稳

# 图片过滤阈值：过滤 logo/小icon
MIN_IMG_W = 180
MIN_IMG_H = 180

# 每个 PDF 最多做多少次“整页兜底渲染”（避免把 VLM 跑爆）
MAX_FALLBACK_FULLPAGE_PER_PDF = 12

# 触发整页兜底的关键词（页面文本命中则说明可能有 Figure/Table）
FALLBACK_PAGE_KEYWORDS = [
    "figure", "fig.", "fig ", "table", "algorithm",
    "图", "表", "算法",
]


# ------------------------
# Utils
# ------------------------
def file_sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def df_to_markdown(df: pd.DataFrame, max_rows=40, max_cols=12) -> str:
    df2 = df.copy()
    if df2.shape[0] > max_rows:
        df2 = df2.head(max_rows)
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    return df2.to_markdown(index=False)


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------
# VLM
# ------------------------
def load_vlm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(VLM_NAME, trust_remote_code=True, use_fast=False)
    model = AutoModelForVision2Seq.from_pretrained(
        VLM_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model.to("cpu")
    return processor, model


def vlm_caption(processor, model, image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")

    instruction = """
你是论文图表/表格解析助手，需要把图片内容转成“可检索的文本块”，用于本地知识库问答（RAG）。
要求：
1) 识别这张图片属于：表格截图 / 柱状图 / 折线图 / 示意图 / 其他。
2) 若是表格：尽量输出 Markdown 表格（列名清晰），并给出一句“表格结论”总结。
3) 若是图表：写出横轴/纵轴含义、单位（若有）、主要趋势、关键对比、可能的结论（3~6条要点）。
4) 允许中英文混合，优先保留图中原始术语（例如 Zipf, Throughput, Mop/s 等）。
5) 不要胡编数据；看不清的地方明确写“无法辨认”。

输出格式（严格遵守）：
[TYPE] ...
[CAPTION] ...
[DETAILS]
- ...
""".strip()

    # ✅ 关键：Qwen2-VL 需要 content=[{"type":"image"}, {"type":"text","text":...}]
    # 这样 chat_template 才会生成真实的 image tokens（否则 tokens=0）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError(
            "This processor has no apply_chat_template(). "
            "Please upgrade transformers or use a Qwen2-VL compatible processor."
        )

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 注意：text 是 prompt（已含 image token），images 传实际图
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    # 放到模型所在 device
    if hasattr(model, "device"):
        dev = model.device
        inputs = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


# ------------------------
# Text extraction (keep your original)
# ------------------------
def extract_text_pymupdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text("text").strip()
        if text:
            yield {
                "type": "text",
                "source": str(pdf_path),
                "page": page_idx,
                "content": text,
            }


# ------------------------
# Table extraction (Camelot)
# ------------------------
def extract_tables_camelot(pdf_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_md = set()

    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
        except Exception:
            continue

        for t in tables:
            try:
                df = t.df
                if df is None or df.shape[0] < 2 or df.shape[1] < 2:
                    continue

                non_empty = (df.astype(str).map(lambda x: str(x).strip() != "").to_numpy()).sum()
                if non_empty < (df.shape[0] * df.shape[1]) * 0.25:
                    continue

                md = df_to_markdown(df)
                if md in seen_md:
                    continue
                seen_md.add(md)

                page0 = None
                try:
                    page0 = int(t.page) - 1
                except Exception:
                    page0 = None

                records.append({
                    "type": "table",
                    "source": str(pdf_path),
                    "page": page0,
                    "content": f"[TABLE][camelot:{flavor}]\n{md}",
                    "meta": {"flavor": flavor},
                })
            except Exception:
                continue

    return records


# ------------------------
# Render / Crop helpers (PyMuPDF)
# ------------------------
def save_crop_from_pdf(
    pdf_path: Path,
    page_idx: int,
    bbox_xyxy: Tuple[float, float, float, float],
    out_path: Path,
    dpi: int = 220,
) -> Optional[Path]:
    doc = fitz.open(pdf_path)
    page = doc[page_idx]

    x1, y1, x2, y2 = bbox_xyxy
    rect = fitz.Rect(x1, y1, x2, y2)

    # 轻微扩边，防止文字被裁掉
    rect = rect + (-2, -2, 2, 2)
    rect = rect & page.rect  # clamp

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return out_path


def render_full_page(
    pdf_path: Path,
    page_idx: int,
    out_path: Path,
    dpi: int = 180,
) -> Path:
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return out_path


# ------------------------
# NEW: extract images via PyMuPDF blocks (no OCR, has page+bbox)
# ------------------------
def extract_figures_pymupdf_blocks(
    pdf_path: Path,
    out_dir: Path,
    min_bbox_w: float = 120.0,
    min_bbox_h: float = 120.0,
):
    """
    直接从 page.get_text("dict") 抓图片块（block.type == 1）
    - 自带 bbox
    - 自带 page_idx
    - 不依赖 OCR/tesseract
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        d = page.get_text("dict")
        blocks = d.get("blocks", [])

        for bi, b in enumerate(blocks):
            if b.get("type") != 1:
                continue

            bbox = b.get("bbox", None)  # [x0,y0,x1,y1]
            if not bbox or len(bbox) != 4:
                continue

            x0, y0, x1, y1 = bbox
            w = x1 - x0
            h = y1 - y0

            if w < min_bbox_w or h < min_bbox_h:
                continue

            img_path = out_dir / f"p{page_idx:03d}_imgblk_{bi:04d}.png"
            save_crop_from_pdf(pdf_path, page_idx, (x0, y0, x1, y1), img_path, dpi=220)
            yield page_idx, img_path


def page_text_has_keywords(page_text: str) -> bool:
    t = (page_text or "").lower()
    return any(k in t for k in FALLBACK_PAGE_KEYWORDS)


# ------------------------
# NEW: multimodal parsing without unstructured dependency
# ------------------------
def parse_pdf_multimodal_pymupdf(pdf_path: Path, processor, model) -> List[Dict[str, Any]]:
    """
    产出 figure blocks：
    1) 先用 PyMuPDF 图片块抽取（page+bbox）
    2) 若某些页无图片块但命中 Figure/Table 关键词，则整页渲染兜底（限额）
    """
    records: List[Dict[str, Any]] = []
    pdf_img_dir = IMG_DIR / pdf_path.stem
    pdf_img_dir.mkdir(parents=True, exist_ok=True)

    seen_img_hash = set()

    # --- 1) Image blocks ---
    imgblk_pages_hit = set()
    for page0, img_path in extract_figures_pymupdf_blocks(pdf_path, pdf_img_dir):
        # 像素过滤
        try:
            im = Image.open(img_path)
            if im.size[0] < MIN_IMG_W or im.size[1] < MIN_IMG_H:
                continue
        except Exception:
            continue

        ih = file_sha1(img_path)
        if ih in seen_img_hash:
            continue
        seen_img_hash.add(ih)
        imgblk_pages_hit.add(page0)

        cap = vlm_caption(processor, model, img_path)
        records.append({
            "type": "figure",
            "source": str(pdf_path),
            "page": page0,
            "content": f"[FIGURE]\n{cap}\n[IMAGE_PATH] {str(img_path)}",
        })

    # --- 2) Fallback: render full page for vector-figure pages ---
    # 条件：该页没有图片块 + 该页文本包含 Figure/Table 关键词
    fallback_used = 0
    doc = fitz.open(pdf_path)

    for page_idx in range(len(doc)):
        if fallback_used >= MAX_FALLBACK_FULLPAGE_PER_PDF:
            break
        if page_idx in imgblk_pages_hit:
            continue

        page_text = doc[page_idx].get_text("text") or ""
        if not page_text_has_keywords(page_text):
            continue

        full_img_path = pdf_img_dir / f"p{page_idx:03d}_fullpage.png"
        render_full_page(pdf_path, page_idx, full_img_path, dpi=180)

        # 像素过滤（一般不会太小，但防一下）
        try:
            im = Image.open(full_img_path)
            if im.size[0] < MIN_IMG_W or im.size[1] < MIN_IMG_H:
                continue
        except Exception:
            continue

        ih = file_sha1(full_img_path)
        if ih in seen_img_hash:
            continue
        seen_img_hash.add(ih)

        cap = vlm_caption(processor, model, full_img_path)
        records.append({
            "type": "figure",
            "source": str(pdf_path),
            "page": page_idx,
            "content": f"[FIGURE][fullpage_fallback]\n{cap}\n[IMAGE_PATH] {str(full_img_path)}",
        })
        fallback_used += 1

    return records


def main():
    if not DOCS_DIR.exists():
        raise RuntimeError("docs/ not found. Put PDFs into ./docs")

    processor, model = load_vlm()
    records: List[Dict[str, Any]] = []

    for pdf_path in DOCS_DIR.glob("*.pdf"):
        print(f"\n=== Parsing: {pdf_path.name} ===")

        # 1) 正文文本（逐页）
        records.extend(list(extract_text_pymupdf(pdf_path)))

        # 2) 矢量表格（Camelot）
        table_records = extract_tables_camelot(pdf_path)
        records.extend(table_records)
        print(f"  - Camelot tables: {len(table_records)}")

        # 3) 图/表截图/图表：PyMuPDF 图片块 + 整页兜底（无需 OCR）
        mm_records = parse_pdf_multimodal_pymupdf(pdf_path, processor, model)
        records.extend(mm_records)
        print(f"  - PyMuPDF mm blocks: {len(mm_records)}")

    write_jsonl(records, OUT_JSONL)
    print(f"\nSaved: {OUT_JSONL} | records={len(records)}")


if __name__ == "__main__":
    main()
