import os
import csv
from datetime import datetime, timezone
import requests

API_URL = "https://desk.zoho.eu/api/v1/articles"

# Read secrets from env (set these in PowerShell before running)
#   $env:ZOHO_TOKEN = "Zoho-oauthtoken xxx"
#   $env:ZOHO_ORGID = "20109871887"
TOKEN = os.getenv("ZOHO_TOKEN")
ORG_ID = os.getenv("ZOHO_ORGID")

if not TOKEN or not ORG_ID:
    raise SystemExit("Set env vars ZOHO_TOKEN and ZOHO_ORGID before running.")

headers = {
    "Authorization": TOKEN,  # e.g. "Zoho-oauthtoken xxx"
    "orgId": ORG_ID,
    "Accept": "application/json",
}

def iso_to_local(iso):
    if not iso:
        return ""
    try:
        # Example: "2025-10-17T21:01:03.000Z"
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso

def fetch_articles(url=API_URL, headers=headers, params=None):
    r = requests.get(url, headers=headers, params=params, timeout=30)
    print("Status code:", r.status_code)
    try:
        j = r.json()
    except Exception:
        print("Non-JSON response:")
        print(r.text[:1000])
        raise
    return r.status_code, j

status, payload = fetch_articles()

if status != 200:
    print("Request failed.")
    print(payload)
    raise SystemExit(1)

items = payload.get("data", [])
if not items:
    print("No articles found.")
    raise SystemExit(0)

# Print a compact table
print(f"\nFound {len(items)} article(s):")
print("-" * 120)
print(f"{'Title':40}  {'ID':18}  {'Status':10}  {'Category':12}  {'Created':16}  {'Portal URL'}")
print("-" * 120)
for a in items:
    title = (a.get("title") or "")[:40]
    aid = a.get("id", "")
    status = a.get("status", "")
    cat = (a.get("category", {}) or {}).get("name", "")
    created = iso_to_local(a.get("createdTime"))
    portal = a.get("portalUrl", "") or a.get("webUrl", "")
    print(f"{title:40}  {aid:18}  {status:10}  {cat:12}  {created:16}  {portal}")

# Optional: save to CSV
SAVE_CSV = True
if SAVE_CSV:
    out = "zoho_articles.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","title","status","category","createdTime","modifiedTime","portalUrl","webUrl","summary","locale"])
        for a in items:
            w.writerow([
                a.get("id",""),
                a.get("title",""),
                a.get("status",""),
                (a.get("category") or {}).get("name",""),
                a.get("createdTime",""),
                a.get("modifiedTime",""),
                a.get("portalUrl",""),
                a.get("webUrl",""),
                a.get("summary",""),
                a.get("locale",""),
            ])
    print(f"\nSaved {len(items)} rows to {out}")
