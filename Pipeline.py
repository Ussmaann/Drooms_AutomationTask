# Pipeline.py
# Step 1: Extract text from the Confluence PDF (hard-coded path)
# Step 2: Parse key fields and generate a KB draft (Markdown with YAML front matter)
# Step 3: Fetch ALL KB articles via Zoho Desk API and match the title (print decision + all titles checked)
# Step 4: Email the created draft to manager (attachment) and optionally wait for the reply to print it
# NOTE: Supports a manual query-title override via PDF_TITLE_OVERRIDE

from pathlib import Path
import re
import json
import sys
import os  # env vars

# ======================
# Hard-coded paths (your working ones)
# ======================
PDF_PATH = Path(r"C:\Users\usman\Downloads\Droom Task\Drooms_AutomationTask\AI Enginer Case study Data\PROD-BATCH_REDACTION (Redaction 1.1)-120925-222321.pdf")
BASE_OUT = Path(r"C:\Users\usman\Downloads\Droom Task\Drooms_AutomationTask")
OUT_TEXT_DIR = BASE_OUT / "extracted_text"
OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT = OUT_TEXT_DIR / "Batch_Redaction_1_1.txt"

DRAFTS_DIR = BASE_OUT / "drafts"
DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# Optional manual override for the title you want to match against the KB
# ======================
# Example:
# PDF_TITLE_OVERRIDE = "Redacting Selected Document Areas"
PDF_TITLE_OVERRIDE = None

# KB base (used only for pretty URLs if the API doesn't return a web URL)
KB_BASE = "https://drooms.zohodesk.eu"

# ======================
# Zoho Desk API (env overrides strongly recommended)
# ======================
ZOHO_API_BASE = os.getenv("ZOHO_API_BASE", "https://desk.zoho.eu/api/v1")
ZOHO_OAUTH_TOKEN = os.getenv("ZOHO_OAUTH_TOKEN", "1000.f6933a557434e61ce59291dc9c0b5312.e4c1c71cf98648ab061c07e6c3803c7d")
ZOHO_ORG_ID = os.getenv("ZOHO_ORG_ID", "20109871887")

# ======================
# Helpers
# ======================
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF.
    Tries PyPDF2 first; if that fails/empty, falls back to pdfminer.six.
    """
    text = ""
    try:
        import PyPDF2  # pip install PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        if text.strip():
            return text
    except Exception as e:
        print("[info] PyPDF2 failed, trying pdfminer…", e, file=sys.stderr)

    try:
        from pdfminer_high_level import extract_text as pdfminer_extract_text  # pip install pdfminer.six
    except Exception:
        from pdfminer.high_level import extract_text as pdfminer_extract_text

    try:
        return pdfminer_extract_text(str(pdf_path))
    except Exception as e:
        return f"<<Failed to extract PDF text: {e}>>"


def extract_title(text: str, fallback: str = "Untitled Article") -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            s = re.sub(r"^[#*\-\d.\s]+", "", s).strip()
            if 5 <= len(s) <= 140:
                return s
    return fallback


def extract_release_version(text: str) -> str:
    m = re.search(r"\b(?:v|version|release)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+){0,3})\b", text, flags=re.I)
    return m.group(1) if m else ""


def extract_release_date(text: str) -> str:
    patterns = [
        r"\b(20\d{2}-\d{2}-\d{2})\b",
        r"\b(\d{1,2}/\d{1,2}/20\d{2})\b",
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+20\d{2}\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(1)
    return ""


def extract_bullets(text: str):
    bullets = re.findall(r"(?m)^\s*(?:[-*]|[\d]+\.)\s+(.+)$", text)
    return [b.strip() for b in bullets]


def classify_article_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["troubleshoot", "error", "fix failed", "how to resolve", "workaround"]):
        return "troubleshooting"
    if any(k in t for k in ["bug fix", "resolved issue", "hotfix", "patch"]):
        return "bugfix"
    if any(k in t for k in ["new feature", "introduc", "launch", "available now", "added a new"]):
        return "new_feature"
    if any(k in t for k in ["updated", "improved", "enhanced", "changed behavior", "ui update", "1.1"]):
        return "feature_update"
    return "process_update"


def extract_summary(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:3])[:600]


def guess_product_area(text: str) -> str:
    areas = ["redaction", "virtual data room", "q&a", "permissions", "login", "api", "search", "viewer", "document", "mobile"]
    t = text.lower()
    for a in areas:
        if a in t:
            return a
    return ""


def extract_prereqs(text: str):
    candidates = []
    for line in text.splitlines():
        if any(k in line.lower() for k in ["prereq", "require", "you need", "before you", "permissions required"]):
            s = re.sub(r"^[#*\-\d.\s]+", "", line).strip()
            candidates.append(s)
    return candidates[:6]


def extract_steps(text: str):
    steps = re.findall(r"(?m)^\s*(?:\d+[\).\s])\s*(.+)$", text)
    if steps:
        return [s.strip() for s in steps][:12]
    verbs = ("select", "click", "open", "go to", "choose", "type", "press", "save", "define")
    bullets = extract_bullets(text)
    stepish = [b for b in bullets if any(v in b.lower() for v in verbs)]
    return stepish[:12]


def extract_breaking_changes(text: str):
    candidates = []
    for para in re.split(r"\n\s*\n", text):
        if "breaking" in para.lower() or "deprecat" in para.lower():
            found = extract_bullets(para)
            candidates.extend(found if found else [para.strip()])
    return [c[:220] for c in candidates[:5]]


def to_frontmatter(meta: dict) -> str:
    fm = {
        "title": meta["title"],
        "article_type": meta["article_type"],
        "audience": meta.get("audience", "both"),
        "product_area": meta.get("product_area", ""),
        "tags": meta.get("tags", []),
        "release_version": meta.get("release_version", ""),
        "release_date": meta.get("release_date", ""),
        "breaking_changes": meta.get("breaking_changes", []),
        "sources": meta.get("sources", []),
        "review": {
            "owner": "",
            "status": "draft",
            "checklist": [
                "Validate technical accuracy",
                "Add/verify screenshots",
                "Confirm release version/date",
                "Legal/compliance check (if needed)",
            ],
        },
    }

    def yaml_dump(obj, indent=0):
        lines = []
        for k, v in obj.items():
            if isinstance(v, list):
                lines.append(" " * indent + f"{k}:")
                for item in v:
                    lines.append(" " * (indent + 2) + f"- {json.dumps(item, ensure_ascii=False)}")
            elif isinstance(v, dict):
                lines.append(" " * indent + f"{k}:")
                for k2, v2 in v.items():
                    lines.append(" " * (indent + 2) + f"{k2}: {json.dumps(v2, ensure_ascii=False)}")
            else:
                lines.append(" " * indent + f"{k}: {json.dumps(v, ensure_ascii=False)}")
        return "\n".join(lines)

    return f"---\n{yaml_dump(fm)}\n---\n"


def render_markdown(meta: dict) -> str:
    parts = [to_frontmatter(meta)]
    parts.append(f"## Summary\n{meta['summary']}\n")
    if meta.get("prerequisites"):
        parts.append("## Prerequisites")
        parts += [f"- {p}" for p in meta["prerequisites"]]
        parts.append("")
    if meta.get("steps"):
        parts.append("## Steps")
        for i, s in enumerate(meta["steps"], 1):
            parts.append(f"{i}. {s}")
        parts.append("")
    if meta.get("breaking_changes"):
        parts.append("## Breaking changes / Deprecations")
        parts += [f"- {b}" for b in meta["breaking_changes"]]
        parts.append("")
    parts.append("## Notes\n- Auto-generated draft. Please review for accuracy.\n")
    return "\n".join(parts)


# ======================
# Step 3 helpers (Zoho Desk API search) — uses your proven GET /articles with orgId header
# ======================
def list_all_kb_articles_via_api() -> list:
    """
    Fetch KB articles via Zoho Desk API using the SAME working pattern you tested:
      - GET {ZOHO_API_BASE}/articles
      - Headers: Authorization + orgId
    Returns a list of dicts with at least: id, title, url (if present), and raw.
    """
    try:
        import requests
    except Exception as e:
        print(f"[error] requests not available: {e}")
        return []

    url = f"{ZOHO_API_BASE}/articles"
    headers = {
        "Authorization": f"Zoho-oauthtoken {ZOHO_OAUTH_TOKEN}",
        "orgId": ZOHO_ORG_ID,   # matches your working call
        "Accept": "application/json",
        "User-Agent": "KB-Matcher/1.3 (+local-script)",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        print(f"[ZohoDesk] GET {url} -> {resp.status_code}")
    except Exception as e:
        print(f"[error] Zoho Desk API request failed: {e}")
        return []

    if not resp.ok:
        print("[error] Zoho Desk API returned non-OK status.")
        try:
            print("[error] Body:", resp.text[:1000])
        except Exception:
            pass
        return []

    try:
        data = resp.json()
    except Exception as e:
        print("[error] Failed to parse JSON:", e)
        print("[error] Raw body (first 1k):", (resp.text or "")[:1000])
        return []

    # Normalize items — your working response has a top-level "data" list
    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("data") or data.get("articles") or data.get("items") or data.get("resources")
        # If "resources" is a dict with "data"
        if isinstance(items, dict) and "data" in items:
            items = items.get("data")

    if not isinstance(items, list):
        print("[warn] No articles found in response.")
        return []

    articles = []
    for it in items:
        title = (it.get("title") or it.get("name") or "").strip()
        if not title:
            continue
        url = it.get("webUrl") or it.get("webURL") or it.get("portalUrl") or it.get("url") or ""
        if not url and it.get("id"):
            url = f"{KB_BASE}/portal/en/kb/articles/{it['id']}"
        articles.append({
            "id": it.get("id"),
            "title": title,
            "url": url,
            "raw": it,
        })

    # Deduplicate by id/title
    seen_ids = set()
    seen_titles = set()
    deduped = []
    for a in articles:
        key = a.get("id") or a.get("title", "").lower()
        if key and key not in seen_ids and a.get("title", "").strip():
            deduped.append(a)
            seen_ids.add(key)
            seen_titles.add(a["title"].lower())
        elif a.get("title", "").lower() not in seen_titles and a.get("title", "").strip():
            deduped.append(a)
            seen_titles.add(a["title"].lower())

    return deduped


def _best_title_match(query_title: str, candidates: list) -> tuple:
    """Return (best_candidate_dict_or_None, best_score_float)."""
    if not candidates:
        return None, -1.0

    from difflib import SequenceMatcher

    def normalize(t):
        t = t.lower()
        t = re.sub(r"\b(version|update|release)\b", "", t)
        t = re.sub(r"\b\d+(\.\d+)*\b", "", t)
        t = re.sub(r"[^a-z0-9]+", " ", t).strip()
        return t

    def sim(a, b):
        return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

    best, best_score = None, -1.0
    for c in candidates:
        title = c.get("title", "")
        if not title:
            continue
        s = sim(query_title, title)
        if s > best_score:
            best_score = s
            best = c
    return best, best_score


# ======================
# Email helpers (Gmail SMTP + IMAP)
# ======================
import smtplib, imaplib, time, mimetypes, email
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.header import decode_header, make_header

def send_email_with_attachment_smtp(
    subject: str,
    body: str,
    to_addrs: list,
    cc_addrs: list,
    attachment_path: Path,
    smtp_user: str,
    smtp_pass: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = ", ".join(to_addrs)
    if cc_addrs:
        msg["Cc"] = ", ".join(cc_addrs)

    msg.set_content(body)

    if attachment_path and attachment_path.exists():
        ctype, _ = mimetypes.guess_type(str(attachment_path))
        if ctype is None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(attachment_path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=attachment_path.name)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
        s.ehlo()
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)

def _get_email_text(payload) -> str:
    """Return a best-effort plain text from an email.message.Message."""
    if payload.is_multipart():
        for part in payload.walk():
            ctype = part.get_content_type()
            disp = part.get("Content-Disposition", "")
            if ctype == "text/plain" and "attachment" not in (disp or "").lower():
                try:
                    return part.get_content().strip()
                except Exception:
                    try:
                        return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore").strip()
                    except Exception:
                        continue
        part = payload.get_payload(0)
        try:
            return part.get_content().strip()
        except Exception:
            return ""
    else:
        try:
            return payload.get_content().strip()
        except Exception:
            return payload.get_payload(decode=True).decode(payload.get_content_charset() or "utf-8", errors="ignore").strip()

def wait_for_reply_imap(
    original_subject: str,
    expect_from: str,
    imap_user: str,
    imap_pass: str,
    imap_host: str = "imap.gmail.com",
    mailbox: str = "INBOX",
    timeout_minutes: int = 15,
    poll_interval_seconds: int = 15,
) -> dict:
    """
    Poll the inbox for a reply that matches:
      - From: expect_from
      - Subject contains original_subject (or 'Re: original_subject')
    Returns a dict with keys: found(bool), subject, from, date, body
    """
    deadline = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
    subject_tokens = [original_subject.lower(), f"re: {original_subject}".lower()]

    while datetime.now(timezone.utc) < deadline:
        try:
            conn = imaplib.IMAP4_SSL(imap_host, 993)
            conn.login(imap_user, imap_pass)
            conn.select(mailbox)

            since = (datetime.utcnow() - timedelta(days=1)).strftime("%d-%b-%Y")
            typ, data = conn.search(None, f'(SINCE {since})')
            if typ != "OK":
                conn.logout()
                time.sleep(poll_interval_seconds)
                continue

            ids = data[0].split()
            for msg_id in reversed(ids):
                typ, msgdata = conn.fetch(msg_id, "(RFC822)")
                if typ != "OK" or not msgdata or not msgdata[0]:
                    continue
                raw = msgdata[0][1]
                payload = email.message_from_bytes(raw)
                from_hdr = str(make_header(decode_header(payload.get("From", ""))))
                subj_hdr = str(make_header(decode_header(payload.get("Subject", ""))))
                date_hdr = payload.get("Date", "")

                subj_l = subj_hdr.lower()
                if expect_from.lower() in from_hdr.lower() and any(tok in subj_l for tok in subject_tokens):
                    body_text = _get_email_text(payload)
                    conn.logout()
                    return {
                        "found": True,
                        "from": from_hdr,
                        "subject": subj_hdr,
                        "date": date_hdr,
                        "body": (body_text or "").strip()[:5000],
                    }

            conn.logout()
        except Exception:
            pass
        time.sleep(poll_interval_seconds)

    return {"found": False}


# ======================
# API-based title match (replaces old scraping/slug fallback)
# ======================
def find_kb_title_match_via_api(pdf_title: str) -> dict:
    """
    Fetch all articles from the Zoho Desk API, then fuzzy-match titles.
    Decision:
      - If best score >= THRESHOLD -> UPDATE existing
      - Else -> CREATE new
    """
    decision = {
        "action": "new",
        "query_title": pdf_title,
        "matched_title": None,
        "matched_url": None,
        "matched_id": None,
        "score": 0.0,
        "candidates": [],   # list of {"title": "...", "url": "...", "id": "..."}
        "api_base": ZOHO_API_BASE,
    }

    articles = list_all_kb_articles_via_api()
    candidates = []
    for a in articles:
        if not a.get("title"):
            continue
        candidates.append({
            "title": a.get("title", "").strip(),
            "url": a.get("url") or "",
            "id": a.get("id"),
        })
    decision["candidates"] = candidates

    best, best_score = _best_title_match(pdf_title, candidates)
    THRESHOLD = 0.85

    if best and best_score >= THRESHOLD:
        decision.update({
            "action": "update",
            "matched_title": best["title"],
            "matched_url": best.get("url"),
            "matched_id": best.get("id"),
            "score": round(float(best_score), 4),
        })
    elif best:
        decision.update({
            "matched_title": best["title"],
            "matched_url": best.get("url"),
            "matched_id": best.get("id"),
            "score": round(float(best_score), 4),
        })

    return decision


# ======================
# Main flow
# ======================
def main():
    if not PDF_PATH.exists():
        print(f"[error] PDF not found:\n{PDF_PATH}")
        sys.exit(1)

    print(">> Step 1: Extracting text from PDF…")
    pdf_text = extract_text_from_pdf(PDF_PATH)
    OUT_TXT.write_text(pdf_text, encoding="utf-8", errors="ignore")

    print("\n✅ Extracted text preview (first 500 chars):\n")
    print((pdf_text[:500] if isinstance(pdf_text, str) else "<<no text>>").replace("\n", " "))
    print("\nFull text saved to:", OUT_TXT)

    print("\n>> Step 2: Parsing fields and generating KB draft…")
    extracted_text = pdf_text if isinstance(pdf_text, str) else ""
    derived_title = extract_title(extracted_text, fallback=PDF_PATH.stem)

    meta = {
        "title": derived_title[:140],
        "article_type": classify_article_type(extracted_text),
        "audience": "both",
        "product_area": guess_product_area(extracted_text),
        "tags": ["auto-generated", "kb-draft"],
        "summary": extract_summary(extracted_text),
        "prerequisites": extract_prereqs(extracted_text),
        "steps": extract_steps(extracted_text),
        "release_version": extract_release_version(extracted_text),
        "release_date": extract_release_date(extracted_text),
        "breaking_changes": extract_breaking_changes(extracted_text),
        "sources": [str(PDF_PATH)],
    }

    safe_title = re.sub(r"[^a-zA-Z0-9\-_.]+", "_", meta["title"]).strip("_") or "KB_Draft"
    draft_md_path = DRAFTS_DIR / f"{safe_title}.md"
    draft_md_path.write_text(render_markdown(meta), encoding="utf-8")

    print("\n==== Parsed Fields ====")
    print("Title:           ", meta["title"])
    print("Article Type:    ", meta["article_type"])
    print("Product Area:    ", meta["product_area"])
    print("Release Version: ", meta["release_version"] or "(not found)")
    print("Release Date:    ", meta["release_date"] or "(not found)")
    print("Steps Detected:  ", len(meta["steps"]))
    print("Prereqs Detected:", len(meta["prerequisites"]))
    print("Draft Markdown:  ", draft_md_path.resolve())

    print("\n>> Step 3: Fetching KB articles via Zoho Desk API and matching title…")
    query_title = PDF_TITLE_OVERRIDE if (isinstance(PDF_TITLE_OVERRIDE, str) and PDF_TITLE_OVERRIDE.strip()) else meta["title"]
    decision = find_kb_title_match_via_api(query_title)

    print("\nQuery Title (searched):", decision.get("query_title") or query_title)

    # Print all KB titles fetched
    all_titles = [c["title"] for c in decision.get("candidates", [])]
    if all_titles:
        print("\nAll KB Titles Found (checked for match):")
        for idx, t in enumerate(all_titles, 1):
            print(f"  {idx}. {t}")
    else:
        print("\nNo KB articles returned by the API or titles unavailable.")

    matched_title = decision.get("matched_title")
    matched_url = decision.get("matched_url")
    matched_id = decision.get("matched_id")
    score = decision.get("score", 0.0)

    if decision.get("action") == "update":
        print(f"\nDecision: UPDATE existing KB")
        print(f"Matched Title: {matched_title}")
        print(f"Article ID:    {matched_id or '(unknown)'}")
        print(f"URL:           {matched_url or '(none)'}")
        print(f"Score:         {score:.2f}")
    else:
        print("\nDecision: CREATE NEW KB ARTICLE (no strong title match found).")
        if matched_title:
            print(f"Closest Match: {matched_title}  [Score={score:.2f}]")
            print(f"Article ID:    {matched_id or '(unknown)'}")
            print(f"URL:           {matched_url or '(none)'}")
        else:
            print("Closest Match: (none)")

    decision_path = BASE_OUT / "kb_match_decision.json"
    with open(decision_path, "w", encoding="utf-8") as jf:
        json.dump(decision, jf, indent=2, ensure_ascii=False)

    print("\nDecision JSON saved to:", decision_path.resolve())

    # ======================
    # Step 4: Email the draft for review and (optionally) wait for reply
    # ======================
    print("\n>> Step 4: Emailing the draft to manager for review…")

    # Addresses: defaults based on your request; can override via env if needed
    me_addr = os.getenv("SENDER_EMAIL", "usman.verbinden@gmail.com")
    manager_addr = os.getenv("MANAGER_EMAIL", "usman.alot761@gmail.com")

    # SMTP/IMAP creds (use a Gmail App Password, not your regular password)
    smtp_user = os.getenv("SMTP_USER", me_addr)
    smtp_pass = os.getenv("SMTP_PASS")  # REQUIRED
    imap_user = os.getenv("IMAP_USER", me_addr)
    imap_pass = os.getenv("IMAP_PASS")  # REQUIRED

    if not smtp_pass or not imap_pass:
        print("[error] Missing SMTP_PASS/IMAP_PASS env vars. Skipping email send + wait for reply.")
        print("\n✅ Done.")
        return

    subject = f"[KB Review] {meta['title']}"
    body = (
        "Hi,\n\n"
        "Please review the attached KB draft generated from the Confluence PDF.\n\n"
        f"Decision: {('UPDATE existing' if decision.get('action')=='update' else 'CREATE new')} KB\n"
        f"Matched Title: {decision.get('matched_title') or '(none)'}\n"
        f"Matched URL: {decision.get('matched_url') or '(none)'}\n"
        f"Matched ID: {decision.get('matched_id') or '(none)'}\n"
        f"Score: {decision.get('score', 0.0)}\n\n"
        "Thanks!\n"
    )

    try:
        send_email_with_attachment_smtp(
            subject=subject,
            body=body,
            to_addrs=[manager_addr],
            cc_addrs=[me_addr],
            attachment_path=draft_md_path,
            smtp_user=smtp_user,
            smtp_pass=smtp_pass,
        )
        print("✅ Email sent to manager:", manager_addr)
    except Exception as e:
        print("[error] Failed to send email:", e)
        print("\n✅ Done.")
        return

    # Optional: wait for a reply (minutes) via env var; default 0 (don't wait)
    wait_minutes = int(os.getenv("WAIT_FOR_REPLY_MINUTES", "0"))
    if wait_minutes > 0:
        print(f">> Waiting up to {wait_minutes} minute(s) for a reply from {manager_addr}…")
        reply = wait_for_reply_imap(
            original_subject=subject,
            expect_from=manager_addr,
            imap_user=imap_user,
            imap_pass=imap_pass,
            timeout_minutes=wait_minutes,
            poll_interval_seconds=15,
        )
        if reply.get("found"):
            print("\n✅ Manager reply received:")
            print("From:   ", reply["from"])
            print("Date:   ", reply["date"])
            print("Subject:", reply["subject"])
            print("\n--- Reply Body (first 5k chars) ---\n")
            print(reply["body"])
            print("\n--- End Reply ---\n")
        else:
            print("\n(no reply received within the wait window)")
    else:
        print("(not waiting for reply; set WAIT_FOR_REPLY_MINUTES>0 to enable)")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
