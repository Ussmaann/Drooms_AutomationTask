    # -*- coding: utf-8 -*-
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  
import os, re, io, sys, base64, textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import fitz  # PyMuPDF
from rapidfuzz import fuzz, process as rfprocess
from PIL import Image
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Flowable, KeepTogether
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from xml.sax.saxutils import escape as html_escape

# Load environment
if load_dotenv:
    if Path(".env.apicalls").exists():
        load_dotenv(".env.apicalls", override=True, verbose=False)
    elif Path(".env").exists():
        load_dotenv(override=True, verbose=False)

confluence_directory = os.getenv("confluence_directory")
zoho_directory       = os.getenv("zoho_directory")
output_directory           = os.getenv("output_directory")
openaiapi       = os.getenv("openaiapi")

if not (confluence_directory and zoho_directory and output_directory):
    print("[error] Missing env vars. Need confluence_directory, zoho_directory, output_directory")
    sys.exit(1)

c_dir  = Path(confluence_directory)
z_dir  = Path(zoho_directory)
OUTDIR = Path(output_directory)
OUTDIR.mkdir(parents=True, exist_ok=True)

# Helpers & data structures
def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = name.strip().rstrip(" .")
    return name or "untitled"
def _wrap_text(txt: str) -> str:
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# keyword based classification for document
def classify_article_type(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["troubleshoot", "error", "failed to", "how to resolve", "workaround", "issue occurs"]):
        return "troubleshooting"
    if any(k in t for k in ["bug fix", "resolved issue", "hotfix", "patch", "defect"]):
        return "bugfix"
    if any(k in t for k in ["new feature", "introduc", "launch", "available now", "added a new", "added support"]):
        return "new_feature"
    if any(k in t for k in ["updated", "improved", "enhanced", "changed behavior", "ui update", "revamp", "v1.1"]):
        return "feature_update"
    return "process_update"

#to measure Similarity 
def _cosine(u: list, v: list) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    du = sum(x*x for x in u) ** 0.5
    dv = sum(y*y for y in v) ** 0.5
    if du == 0 or dv == 0:
        return 0.0
    return sum(x*y for x, y in zip(u, v)) / (du * dv)

def read_pdf_preview(pdf_path: Path, max_pages: int = 3, max_chars: int = 2000) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    chunks = []
    for i in range(min(len(doc), max_pages)):
        try:
            t = doc[i].get_text()
        except Exception:
            t = ""
        if t:
            chunks.append(t)
        if sum(len(c) for c in chunks) >= max_chars:
            break
    return _wrap_text(" ".join(chunks))[:max_chars]

def embed_text(client: OpenAI, text: str) -> Optional[list]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=text)
        return emb.data[0].embedding
    except Exception:
        return None

#Data structures for images,paragraphs and sections
@dataclass
class ImgBlob:
    data: bytes
    ext: str
    caption: Optional[str] = None
    is_new: bool = False

@dataclass
class ParagraphBlob:
    text: str
    is_new: bool = False

@dataclass
class Section:
    title: str
    paragraphs: List[ParagraphBlob] = field(default_factory=list)
    images: List[ImgBlob] = field(default_factory=list)

@dataclass
class DocStructure:
    title: str
    sections: List[Section]


sim_threshold = 70  #section title match threshold 

# Paragraph detection thresholds
para_similarity = 85
num_re = re.compile(r'\b(\d+(?:\.\d+)*)\b', re.I)

def _numbers(text): 
    return set(num_re.findall(text or ""))

def negflag(text):
    t = (text or "").lower()
    return any(k in t for k in [" not ", "no longer", " cannot", " can't", " won't", "deprecated", "removed"])

def findConflict(a: str, b: str) -> bool:
    # Only consider near-duplicates for conflict evaluation
    if fuzz.token_set_ratio(a, b) < para_similarity:
        return False
    # numbers or versions differ (e.g., v2.3 vs v2.4, counts, dates)
    if _numbers(a) != _numbers(b):
        return True
    # flip heuristics
    if negflag(a) != negflag(b):
        return True
    t = f"{a} || {b}".lower()
    return any(k in t for k in ["instead of", "replaced by", "changed to", "now supports", "no longer", "deprecated"])

def mergeparagraph(zoho_paras: List[ParagraphBlob], conf_paras: List[ParagraphBlob]) -> List[ParagraphBlob]:
    merged = list(zoho_paras)  # keep Zoho by default
    for cp in (p for p in conf_paras if (p.text or "").strip()):
        # find nearest Zoho paragraph
        best_zi, best_score = None, -1
        for i, zp in enumerate(zoho_paras):
            s = fuzz.token_set_ratio(cp.text, zp.text)
            if s > best_score:
                best_zi, best_score = i, s
        if best_zi is not None and best_score >= para_similarity:
            if findConflict(zoho_paras[best_zi].text, cp.text):
                merged[best_zi] = ParagraphBlob(text=cp.text, is_new=True)  # Confluence wins on conflict
            else:
                merged.append(ParagraphBlob(text=cp.text, is_new=True))      # complementary detail
        else:
            merged.append(ParagraphBlob(text=cp.text, is_new=True))          # new info
    return merged

def _safe_image_sig(img: ImgBlob) -> Tuple[int, int]:
    return (len(img.data or b""), hash(img.data or b""))

#ReportLab: to built pdf
LIGHT_GREEN = colors.HexColor("#2ECC71")
CAPTION_GRAY = colors.HexColor("#666666")
STYLE_H1 = ParagraphStyle(
    name="H1", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=8, alignment=TA_LEFT
)
STYLE_H2 = ParagraphStyle(
    name="H2", fontName="Helvetica-Bold", fontSize=14, leading=18, spaceAfter=6, alignment=TA_LEFT
)
STYLE_BODY = ParagraphStyle(
    name="Body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=6, alignment=TA_LEFT
)
def paraa(text: str, style: ParagraphStyle, color=None) -> Paragraph:
    safe = html_escape(text or "")
    if color is not None:
        style = ParagraphStyle(**{**STYLE_BODY.__dict__, "name": style.name + "_c", "textColor": color})
    return Paragraph(safe, style)

def Scaletowidth(img_reader, max_w):
    iw, ih = img_reader.getSize()
    if iw <= 0 or ih <= 0:
        return max_w, max_w * 0.75
    scale = min(1.0, max_w / iw)
    return iw * scale, ih * scale

class BorderedImage(Flowable):
    def __init__(self, img_reader, width, height, border_color=None, caption=None, caption_color=CAPTION_GRAY):
        super().__init__()
        self.img_reader = img_reader
        self.w = width
        self.h = height
        self.border_color = border_color
        self.caption = caption
        self.caption_color = caption_color
        self.caption_leading = 10 if caption else 0
        self._height = self.h + (self.caption_leading + 2 if caption else 0)

    def wrap(self, availWidth, availHeight):
        return (self.w, self._height)

    def draw(self):
        c = self.canv
        c.drawImage(self.img_reader, 0, self._height - self.h, width=self.w, height=self.h, preserveAspectRatio=False, mask='auto')
        if self.border_color:
            c.setStrokeColor(self.border_color)
            c.setLineWidth(1)
            c.rect(0, self._height - self.h, self.w, self.h, stroke=1, fill=0)
        if self.caption:
            c.setFont("Helvetica", 8)
            c.setFillColor(self.caption_color)
            c.drawString(0, 0, self.caption)

# Extract Pdf Structure
def extract_pdf_structure(pdf_path: Path) -> DocStructure:
    doc = fitz.open(pdf_path)
    doc_title = pdf_path.stem
    sections: List[Section] = []
    current: Optional[Section] = None
    def page_median_font_size(textdict) -> float:
        sizes = []
        for block in textdict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    s = span.get("size", 0)
                    if s: sizes.append(float(s))
        if not sizes: return 10.0
        sizes.sort()
        return sizes[len(sizes)//2]

    for page_index, page in enumerate(doc):
        try:
            textdict = page.get_text("dict")
        except Exception:
            textdict = {}

        median_size = page_median_font_size(textdict)

        def is_heading(max_span_size, text_len):
            return (max_span_size >= median_size + 2.0) and (text_len <= 140)

        for block in textdict.get("blocks", []):
            if "lines" not in block: 
                continue
            block_text_parts, max_span_size = [], 0.0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    if t.strip():
                        block_text_parts.append(t)
                        max_span_size = max(max_span_size, float(span.get("size", 0)))
            if not block_text_parts:
                continue
            text = _wrap_text(" ".join(block_text_parts))

            if page_index == 0:
                # establish first true H1
                if current is None and is_heading(max_span_size, len(text)):
                    current = Section(title=text)
                    sections.append(current)
                    continue
                # if we already have H1, decide if this starts a new section
                if current is not None and is_heading(max_span_size, len(text)) and text != sections[0].title:
                    current = Section(title=text)
                    sections.append(current)
                else:
                    if current is None:
                        current = Section(title="Introduction")
                        sections.append(current)
                    current.paragraphs.append(ParagraphBlob(text=text))
            else:
                # subsequent pages: simpler heading rule
                if is_heading(max_span_size, len(text)):
                    current = Section(title=text)
                    sections.append(current)
                else:
                    if current is None:
                        current = Section(title="Introduction")
                        sections.append(current)
                    current.paragraphs.append(ParagraphBlob(text=text))

        # images
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
            except Exception:
                base = doc.extract_image(xref) or {}
                img_bytes = base.get("image", b"")
            if img_bytes:
                if not sections:
                    sections.append(Section(title="Introduction"))
                sections[-1].images.append(ImgBlob(data=img_bytes, ext="png"))

    if not sections:
        sections = [Section(title=doc_title)]
    return DocStructure(title=doc_title, sections=sections)

#Image Merging

def MergeImages(zoho_imgs: List[ImgBlob], conf_imgs: List[ImgBlob]) -> List[ImgBlob]:
    merged = [ImgBlob(data=im.data, ext=im.ext, caption=im.caption, is_new=False) for im in zoho_imgs]
    seen = {_safe_image_sig(im) for im in zoho_imgs}
    for cim in conf_imgs:
        sig = _safe_image_sig(cim)
        if sig not in seen:
            merged.append(ImgBlob(data=cim.data, ext=cim.ext, caption=cim.caption, is_new=True))
            seen.add(sig)
    return merged

def mergeinline(zoho: DocStructure, conf: DocStructure) -> DocStructure:
    # copy Zoho structure
    merged_sections: List[Section] = [
        Section(
            title=zs.title,
            paragraphs=[ParagraphBlob(text=p.text, is_new=False) for p in zs.paragraphs],
            images=[ImgBlob(data=im.data, ext=im.ext, caption=im.caption, is_new=False) for im in zs.images],
        )
        for zs in zoho.sections
    ]

    # fuzzy section alignment
    zoho_titles = [s.title for s in zoho.sections]

    # For each Confluence section
    for cs in conf.sections:
        new_paras = [ParagraphBlob(text=p.text, is_new=True) for p in cs.paragraphs if (p.text or "").strip()]
        new_imgs  = [ImgBlob(data=im.data, ext=im.ext, caption=im.caption, is_new=True) for im in cs.images]

        idx = None
        if zoho_titles:
            match = rfprocess.extractOne(cs.title, zoho_titles, scorer=fuzz.token_sort_ratio)
            if match and match[1] >= sim_threshold:
                idx = match[2]

        if idx is not None:
            #  conflict adjustment
            merged_sections[idx].paragraphs = mergeparagraph(merged_sections[idx].paragraphs, new_paras)
            merged_sections[idx].images     = MergeImages(merged_sections[idx].images, new_imgs)
        else:
            # add as new section 
            merged_sections.append(Section(title=cs.title, paragraphs=new_paras, images=new_imgs))

    # 1st section: lock Zoho title, union intro with Confluence
    if merged_sections and zoho.sections:
        ms0 = merged_sections[0]
        zs0 = zoho.sections[0]
        cs0 = conf.sections[0] if conf.sections else Section(title="")

        # Lock title to Zoho H1
        if zs0.title:
            ms0.title = zs0.title

        # Recompute intro using raw Zoho/Confluence first-section paragraphs to be safe
        ms0.paragraphs = mergeparagraph(
            [ParagraphBlob(text=p.text, is_new=False) for p in zs0.paragraphs],
            [ParagraphBlob(text=p.text, is_new=True)  for p in cs0.paragraphs]
        )

        # Merge images with dedupe (keep Zoho images, add Confluence new ones)
        ms0.images = MergeImages(zs0.images, cs0.images)

    return DocStructure(title=f"{zoho.title} (updated)", sections=merged_sections)

#Rendering

def buildpdf(doc_struct: DocStructure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm
    )
    story = []
    story.append(paraa(doc_struct.title, STYLE_H1))
    story.append(Spacer(1, 8))

    max_img_w = A4[0] - (doc.leftMargin + doc.rightMargin)

    for sec in doc_struct.sections:
        story.append(paraa(sec.title, STYLE_H2))
        story.append(Spacer(1, 4))

        for p in sec.paragraphs:
            color = LIGHT_GREEN if p.is_new else None
            story.append(paraa(p.text.strip(), STYLE_BODY, color=color))

        for im in sec.images:
            try:
                img_reader = ImageReader(io.BytesIO(im.data))
            except Exception:
                continue
            w, h = Scaletowidth(img_reader, max_img_w)
            border = LIGHT_GREEN if im.is_new else None
            caption_text = "(new)" if im.is_new else (im.caption or None)
            story.append(Spacer(1, 6))
            story.append(KeepTogether(BorderedImage(
                img_reader, width=w, height=h, border_color=border,
                caption=caption_text, caption_color=LIGHT_GREEN if im.is_new else CAPTION_GRAY
            )))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 10))

    doc.build(story)

#Recreation with LLM
def extractimages(pdf_path: Path, max_images: int = 30):
    imgs = []
    doc = fitz.open(pdf_path)
    counter = 1
    for pno, page in enumerate(doc):
        for info in page.get_images(full=True):
            if len(imgs) >= max_images:
                break
            xref = info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                data = pix.tobytes("png")
                mime = "image/png"
            except Exception:
                base = doc.extract_image(xref) or {}
                data = base.get("image", b"")
                ext = (base.get("ext", "png") or "png").lower()
                mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
            if not data:
                continue
            try:
                im = Image.open(io.BytesIO(data))
                w, h = im.size
            except Exception:
                w, h = (0, 0)
            imgs.append({
                "id": f"IMG{counter:03d}",
                "page": pno + 1,
                "data": data,
                "mime": mime,
                "width": w,
                "height": h,
            })
            counter += 1
        if len(imgs) >= max_images:
            break
    return imgs

def normalizeplaceholders(md_text: str) -> str:
    s = md_text
    s = re.sub(r'!\[\s*\]\(\s*ID:\s*(IMG\d{3})\s*\)', r'![screenshot](ID:\1)', s, flags=re.IGNORECASE)
    s = re.sub(r'!\[[^\]]*\]\(\s*ID:\s*(IMG\d{3})\s*\)', r'![screenshot](ID:\1)', s, flags=re.IGNORECASE)
    s = re.sub(r'^\s*\(?\s*ID:\s*(IMG\d{3})\s*\)?\s*$', r'![screenshot](ID:\1)', s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r'^\s*\{\{\s*(IMG\d{3})\s*\}\}\s*$', r'![screenshot](ID:\1)', s, flags=re.IGNORECASE | re.MULTILINE)

    def _split_inline_placeholders(line: str) -> list[str]:
        parts = []
        pos = 0
        pattern = re.compile(r'!\[screenshot\]\(ID:(IMG\d{3})\)', flags=re.IGNORECASE)
        for m in pattern.finditer(line):
            pre = line[pos:m.start()].rstrip()
            ph  = m.group(0)
            pos = m.end()
            if pre:
                parts.append(pre)
            parts.append(ph)
        tail = line[pos:].strip()
        if tail:
            parts.append(tail)
        return parts if parts else [line]

    lines = s.splitlines()
    out_lines = []
    for line in lines:
        if re.search(r'!\[screenshot\]\(ID:\s*IMG\d{3}\)', line, flags=re.IGNORECASE):
            out_lines.extend(_split_inline_placeholders(line))
        else:
            out_lines.append(line)
    s = "\n".join(out_lines)

    s = re.sub(r'!\[screenshot\]\(\s*ID:\s*(IMG\d{3})\s*\)', r'![screenshot](ID:\1)', s, flags=re.IGNORECASE)
    return s

def mdtopdf(md_text: str, images_index: dict, out_path: Path):
    lines = md_text.splitlines()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm
    )
    story = []
    max_img_w = A4[0] - (doc.leftMargin + doc.rightMargin)

    def add(text, style):
        story.append(paraa(text.strip(), style))
        story.append(Spacer(1, 4))

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
        if line.startswith("### "):
            add(line[4:], STYLE_H2)  # H3 styled as H2 for simplicity
        elif line.startswith("## "):
            add(line[3:], STYLE_H2)
        elif line.startswith("# "):
            add(line[2:], STYLE_H1)
        elif re.match(r'!\[[^\]]*\]\(ID:IMG\d{3}\)', line):
            m = re.search(r'\(ID:(IMG\d{3})\)', line)
            if m:
                img_id = m.group(1)
                entry = images_index.get(img_id)
                if entry:
                    try:
                        img_reader = ImageReader(io.BytesIO(entry["data"]))
                        w, h = Scaletowidth(img_reader, max_img_w)
                        story.append(KeepTogether(BorderedImage(
                            img_reader, width=w, height=h, border_color=None,
                            caption=f"{img_id} (page {entry['page']})", caption_color=CAPTION_GRAY
                        )))
                        story.append(Spacer(1, 8))
                    except Exception:
                        pass
        else:
            add(line, STYLE_BODY)

    doc.build(story)

def recreatewithLLM(
    merged_pdf: Path,
    title_for_output: str,
    outdir: Path,
    *,
    doc_type_hint: str,
    locked_h1: Optional[str] = None,
    locked_intro: Optional[str] = None
):
    if not openaiapi:
        print("Warning: openaiapi missing. Skipping recreate step.")
        return None

    try:
        client = OpenAI(api_key=openaiapi)
    except Exception as e:
        print(f"Error: Failed to init OpenAI client: {e}")
        return None

    print("Preparing images for OpenAI …")
    images = extractimages(merged_pdf, max_images=30)
    images_index = {e["id"]: e for e in images}
    print(f"screenshots: {len(images)}")

#We read more pages and allow more chars than the default preview.)
    source_text = read_pdf_preview(merged_pdf, max_pages=50, max_chars=30000)
    if not source_text.strip():
        print("warning: Empty source_text from merged PDF. The model may miss details.")

    # Locked content note (prevents model from rewriting H1/intro)
    lock_blocks = []
    if locked_h1:
        lock_blocks.append(f"H1_TITLE (DO NOT MODIFY):\n{locked_h1}")
    if locked_intro:
        lock_blocks.append(
            "INTRO_BLOCK (keep verbatim; you MAY append Confluence-only updates; "
            "if any conflict with Zoho exists, prefer the Confluence fact and suffix '(updated)'):\n"
            f"{locked_intro}"
        )
    locked_note = "\n\n".join(lock_blocks)

    guideline = textwrap.dedent(f"""
    You are a technical writer. Recreate the attached manual so it is easy for end users to follow.

    Requirements:
    - Use clear, logical order. Organize content with proper headings.
    - Turn procedures into step-by-step instructions followed by the correct screenshot (if available) .
    - The instructions must explain what the user sees in the screenshot. 
    - Do not miss any information from source file, even if there is no associated screenshot.
    - Do not use the images/screenshots that are fully black OR are not readable/visible.
    

    INVARIANTS (do NOT violate):
    - Do NOT omit any information from the source text provided.
    - Rephrase for readability only; preserve every fact, number, bullet, and option exactly.
    - Every feature, bullet point, and subsection in the source must appear once in the output (you may reorder, but not drop).
    - Use EXACT H1 from H1_TITLE (do not rewrite).
    - Keep Introduction paragraphs verbatim as the first section; you may append Confluence-only updates to it.
    - If the Confluence text contradicts Zoho, prefer Confluence and mark the sentence with "(updated)".

    CRITICAL - IMAGE PLACEHOLDERS (exact format, on their own line):
        ![screenshot](ID:IMG###)

    Rules:
    - Use ONLY the IMG IDs I provide (no URLs or filenames).
    - Put each placeholder on its own line, immediately AFTER the paragraph/step it illustrates.
    - If a step needs multiple screenshots, put each placeholder on its own following line.
    - The overall document type is: {doc_type_hint}.

    CHECKLIST (must pass):
    - Skip any heading that has no content between consuctive previous and the next heading.
    - H1 exactly matches H1_TITLE.
    - Introduction appears verbatim first; appended updates are clearly integrated.
    - No omissions: every bullet, subsection, and detail from SOURCE TEXT appears in the output (reordered is OK).
    - Numbers, versions, and options match SOURCE TEXT exactly.
    - Each numbered step is immediately followed by its screenshot placeholder line.
    - Use only provided IMG### IDs; no external links or invented features.
    - Provided tips/extra guidelines must come at the very end (if any).
    """).strip()


    content_parts = []

    # 3a) NEW: feed the full source text first so the model must keep everything
    content_parts.append({
        "type": "input_text",
        "text": "SOURCE TEXT (you must preserve ALL information, you may only rephrase and reorganize):\n" + source_text
    })

    # 3b) Locked content (H1 + Intro) so it cannot be rewritten/dropped
    lock_blocks = []
    if locked_h1:
        lock_blocks.append(f"H1_TITLE (DO NOT MODIFY):\n{locked_h1}")
    if locked_intro:
        lock_blocks.append(
            "INTRO_BLOCK (keep verbatim; you MAY append Confluence-only updates; "
            "if any conflict with Zoho exists, prefer the Confluence fact and suffix '(updated)'):\n"
            f"{locked_intro}"
        )
    if lock_blocks:
        content_parts.append({"type": "input_text", "text": "LOCKED CONTENT:\n" + "\n\n".join(lock_blocks)})

    # 3c) The strengthened guideline/checklist
    content_parts.append({"type": "input_text", "text": guideline})

    # 3d) Announce the available screenshots and attach them
    content_parts.append({"type": "input_text", "text": "Available screenshots (use these IDs):"})
    content_parts.append({"type": "input_text", "text": ", ".join(e["id"] for e in images) if images else "(none)"})

    for e in images:
        b64 = base64.b64encode(e["data"]).decode("ascii")
        data_uri = f"data:{e['mime']};base64,{b64}"
        content_parts.append({"type": "input_image", "image_url": data_uri})
        content_parts.append({"type": "input_text", "text": f"Image ID for the previous image: {e['id']} (from page {e['page']})"})

    try:
        print("Calling OpenAI API …")
        resp = client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": content_parts}],
            temperature=0.2,
        )
        print(" OpenAI API call succeeded.")
        out_text = getattr(resp, "output_text", None)
        if out_text:
            print("   Preview Sample")
            print(out_text[:400])
            print("   End preview")
    except Exception as e:
        print(f"Error: OpenAI response failed: {e}")
        return None

    md_text = getattr(resp, "output_text", "") or ""
    if not md_text.strip():
        print("[Model returned empty content; aborting PDF build.")
        return None

    md_text = normalizeplaceholders(md_text)

    placeholders = re.findall(r'!\[screenshot\]\(ID:(IMG\d{3})\)', md_text)
    if not placeholders and images_index:
        print("[warn] No placeholders in output; appending a Screenshots section with all images.")
        md_text += "\n\n## Screenshots\n"
        for img_id in images_index.keys():
            md_text += f"\n![screenshot](ID:{img_id})"

    md_path = outdir / (sanitize_filename(f"{title_for_output} (recreated)") + ".md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Saved model Markdown: {md_path}")
    except Exception as e:
        print(f"[warn] Could not save Markdown file: {e}")

    out_pdf_path = outdir / (sanitize_filename(f"{title_for_output} (recreated)") + ".pdf")
    print(f"Building recreated PDF: {out_pdf_path}")
    mdtopdf(md_text, images_index, out_pdf_path)

    if out_pdf_path.exists():
        print(f"OpenAI recreation completed: {out_pdf_path}")
    else:
        print("Recreated file was not generated.")
        return None

    return out_pdf_path

# Main flow
def main():
    print("Using:")
    print("  Confluence dir:", c_dir)
    print("  Zoho dir      :", z_dir)
    print("  Output dir    :", OUTDIR)
    print()

    c_files = sorted(c_dir.glob("*.pdf"))
    z_files = sorted(z_dir.glob("*.pdf"))
    if not c_files or not z_files:
        print("Error: Put your PDFs in the respective folders.")
        return

    # choose Confluence file 
    conf_file = c_files[0]

    # Document classification for file type which we may can use  
    conf_doc_type = classify_article_type(read_pdf_preview(conf_file, max_pages=1, max_chars=3000))
    print(f"\n Confluence document type: {conf_doc_type}")

    # Prepare OpenAI client only if needed later
    client = None
    if openaiapi:
        try:
            client = OpenAI(api_key=openaiapi)
        except Exception:
            client = None

    # Fuzzy title matching
    z_titles = [p.stem for p in z_files]
    match = rfprocess.extractOne(conf_file.stem, z_titles, scorer=fuzz.token_sort_ratio)

    zoho_file = None
    if match and match[1] >= 75:
        zoho_file = z_dir / f"{match[0]}.pdf"
        print(f'\n Fuzzy match for Zoho "{zoho_file.name}" and Confluence: "{conf_file.name}" [score={match[1]:.1f}]')
    else:
        # Fall back to semantic match using 
        if client is None:
            print("[warn] No OpenAI client for semantic match. Please ensure titles are similar or set openaiapi.")
            zoho_file = z_dir / f"{z_files[0].name}"
            print(f"[fallback] Using first Zoho file: {zoho_file.name}")
        else:
            # semantic fallback via Open AI embedding embeddings
            def best_zoho_by_semantics(conf_pdf: Path, zoho_dir: Path, client: OpenAI, *, min_chars: int = 120) -> Tuple[Optional[Path], float]:
                conf_txt = read_pdf_preview(conf_pdf)
                if len(conf_txt) < min_chars:
                    return None, 0.0
                conf_vec = embed_text(client, conf_txt)
                if not conf_vec:
                    return None, 0.0
                best_path, best_score = None, -1.0
                for zpdf in sorted(zoho_dir.glob("*.pdf")):
                    z_txt = read_pdf_preview(zpdf)
                    if len(z_txt) < 50:
                        continue
                    z_vec = embed_text(client, z_txt)
                    if not z_vec:
                        continue
                    score = _cosine(conf_vec, z_vec)
                    if score > best_score:
                        best_score, best_path = score, zpdf
                return best_path, best_score

            zoho_file, sem_score = best_zoho_by_semantics(conf_file, z_dir, client)
            if not zoho_file or sem_score < 0.65:
                print("warning Semantic match is too low")
                zoho_file = z_dir / f"{z_files[0].name}"
            else:
                print(f'\nSemantic match with Zoho: "{zoho_file.name}" and Confluence: "{conf_file.name}" [cos={sem_score:.3f}]')

    # Extract structures from pdf files
    zoho_struct = extract_pdf_structure(zoho_file)
    conf_struct = extract_pdf_structure(conf_file)

    # Merge the content from zoho and confluence files
    merged = mergeinline(zoho_struct, conf_struct)

    # Save updated merged PDF file
    out_name = sanitize_filename(f"{zoho_file.stem} (updated)") + ".pdf"
    merged_pdf_path = OUTDIR / out_name
    print(f"\nBuilding merged PDF  {merged_pdf_path}")
    buildpdf(merged, merged_pdf_path)
    print(f" Done. Saved: {merged_pdf_path}")

    # Recreate pdf with OpenAI
    locked_h1   = zoho_struct.sections[0].title if zoho_struct.sections else zoho_file.stem
    locked_intro = "\n\n".join(p.text for p in (merged.sections[0].paragraphs if merged.sections else []) if (p.text or "").strip())

    recreatewithLLM(
        merged_pdf_path,
        title_for_output=zoho_file.stem,
        outdir=OUTDIR,
        doc_type_hint=conf_doc_type,
        locked_h1=locked_h1,
        locked_intro=locked_intro
    )

if __name__ == "__main__":
    main()
