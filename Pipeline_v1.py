# MakeLosslessManual_with_LLM.py
# Step 1: Make a lossless merged manual: [COVER] + [ZOHO (as-is)] + [DIVIDER] + [CONFLUENCE (as-is)]
# Step 2: (Optional, now skipped) LLM-cleaned version placeholder.

import os
import difflib
import itertools
from datetime import datetime
from typing import List, Tuple

from dotenv import load_dotenv
from pypdf import PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import fitz  # PyMuPDF (text+image extraction)
from PIL import Image

# -----------------------
# Config
# -----------------------
load_dotenv()
CONFLUENCE_PATH = os.getenv("CONFLUENCE_PATH")
ZOHO_PATH       = os.getenv("ZOHO_PATH")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR")
TITLE_THRESHOLD = float(os.getenv("TITLE_THRESHOLD", "0.50"))

MARGIN = 18 * mm

# Image visibility thresholds
MIN_IMG_WIDTH_PX  = 380
MIN_IMG_HEIGHT_PX = 220

# -----------------------
# Helpers
# -----------------------
def ensure_dir(path: str):
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"Folder not found: {path}")

def list_pdf_files(folder: str) -> List[str]:
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])

def get_title(filename: str) -> str:
    name, _ = os.path.splitext(filename)
    name = name.lower()
    for token in ["prod", "duplicate", "final", "product description", "version",
                  "(", ")", "_", "-", "–", "—"]:
        name = name.replace(token, " ")
    return " ".join(name.split()).strip()

def best_unique_pairs(conf_files, zoho_files, threshold: float) -> List[Tuple[str, str, float]]:
    conf_titles = {c: get_title(c) for c in conf_files}
    zoho_titles = {z: get_title(z) for z in zoho_files}

    scored = []
    for c, z in itertools.product(conf_files, zoho_files):
        s = difflib.SequenceMatcher(None, conf_titles[c], zoho_titles[z]).ratio()
        if s >= threshold:
            scored.append((c, z, round(s, 2)))
    scored.sort(key=lambda x: x[2], reverse=True)

    used_c, used_z, result = set(), set(), []
    for c, z, s in scored:
        if c in used_c or z in used_z:
            continue
        result.append((c, z, s))
        used_c.add(c); used_z.add(z)
    return result

# -----------------------
# PDF Generation
# -----------------------
def render_simple_page_pdf(title: str, subtitle: str, body_lines: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, spaceAfter=10)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, spaceAfter=8)
    base = ParagraphStyle('Base', parent=styles['Normal'], fontName='Helvetica', fontSize=11,
                          leading=16, alignment=TA_JUSTIFY, spaceAfter=6)

    story = []
    if title:
        story.append(Paragraph(title, h1))
    if subtitle:
        story.append(Paragraph(subtitle, h2))
        story.append(Spacer(1, 6))
    for ln in body_lines:
        story.append(Paragraph(ln, base))
    doc.build(story)

def merge_pdfs_lossless(zoho_pdf: str, conf_pdf: str, cover_pdf: str, divider_pdf: str, out_pdf: str):
    writer = PdfWriter()
    if cover_pdf and os.path.exists(cover_pdf):
        writer.append(cover_pdf)
    writer.append(zoho_pdf)
    if divider_pdf and os.path.exists(divider_pdf):
        writer.append(divider_pdf)
    writer.append(conf_pdf)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with open(out_pdf, "wb") as f_out:
        writer.write(f_out)
    writer.close()

# -----------------------
# Image Extraction (for completeness)
# -----------------------
def extract_images_in_order(pdf_path: str, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc):
            for ix, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < MIN_IMG_WIDTH_PX or pix.height < MIN_IMG_HEIGHT_PX:
                    continue
                out = os.path.join(out_dir, f"p{pno+1:03d}_img{ix+1:02d}.png")
                pix.save(out)
                saved.append(out)
    return saved

# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(CONFLUENCE_PATH)
    ensure_dir(ZOHO_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔍 Scanning folders...")
    conf_files = list_pdf_files(CONFLUENCE_PATH)
    zoho_files = list_pdf_files(ZOHO_PATH)
    print(f"Confluence PDFs ({len(conf_files)}): {conf_files}")
    print(f"Zoho KB PDFs   ({len(zoho_files)}): {zoho_files}")

    print(f"\n🧮 Matching (1:1) by title with threshold ≥ {TITLE_THRESHOLD} ...")
    pairs = best_unique_pairs(conf_files, zoho_files, TITLE_THRESHOLD)
    if not pairs:
        print("No pairs met the threshold. Exiting.")
        return

    print("\n📘 Matched pairs:")
    for c, z, s in pairs:
        print(f" - {c}  ↔  {z}   (similarity={s})")

    print("\n🛠️ Building lossless manuals (LLM step skipped)...")
    for conf_f, kb_f, score in pairs:
        conf_path = os.path.join(CONFLUENCE_PATH, conf_f)
        kb_path   = os.path.join(ZOHO_PATH, kb_f)

        # Step 1: LOSSLESS MERGE
        cover_path = os.path.join(OUTPUT_DIR, "_tmp_cover.pdf")
        render_simple_page_pdf(
            title=os.path.splitext(kb_f)[0],
            subtitle="Updated Manual",
            body_lines=[
                f"Compiled on {datetime.now().strftime('%Y-%m-%d')}",
                "This manual merges the original Knowledge Base article (as-is) with the matching Confluence document."
            ],
            out_path=cover_path
        )

        divider_path = os.path.join(OUTPUT_DIR, "_tmp_divider.pdf")
        render_simple_page_pdf(
            title="Additions from Confluence",
            subtitle=os.path.splitext(conf_f)[0],
            body_lines=["The following pages contain the Confluence document exactly as in the original."],
            out_path=divider_path
        )

        lossless_name = os.path.splitext(kb_f)[0] + " (LOSSLESS UPDATED MANUAL).pdf"
        lossless_path = os.path.join(OUTPUT_DIR, lossless_name)
        merge_pdfs_lossless(kb_path, conf_path, cover_path, divider_path, lossless_path)
        print(f"✅ Lossless: {lossless_path}")

        # Step 2: Placeholder clean manual (no API)
        placeholder_path = os.path.join(OUTPUT_DIR, os.path.splitext(kb_f)[0] + " (CLEAN MANUAL PLACEHOLDER).pdf")
        render_simple_page_pdf(
            title=os.path.splitext(kb_f)[0],
            subtitle="Manual Refinement Skipped",
            body_lines=[
                "The LLM-based cleaning step was skipped in this version.",
                "Only the original merged manual (lossless) has been created."
            ],
            out_path=placeholder_path
        )
        print(f"✅ Placeholder clean manual: {placeholder_path}")

    print("\n🎉 Done. Check the output folder for results.")

if __name__ == "__main__":
    main()
