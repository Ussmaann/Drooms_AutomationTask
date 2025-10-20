import os
import sys
import json
import requests
from requests.auth import HTTPBasicAuth

def need(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return v

SITE = need("ATLASSIAN_SITE").rstrip("/")                  # e.g. https://usmanverbinden.atlassian.net
EMAIL = need("ATLASSIAN_EMAIL")                            # your Atlassian login email
API_TOKEN = need("ATLASSIAN_API_TOKEN")                    # API token from id.atlassian.com
auth = HTTPBasicAuth(EMAIL, API_TOKEN)
headers = {"Accept": "application/json"}

def get(url, **kwargs):
    r = requests.get(url, headers=headers, auth=auth, **kwargs)
    if not r.ok:
        print(f"\nHTTP {r.status_code} for GET {url}\nResponse:\n{r.text}\n", file=sys.stderr)
        r.raise_for_status()
    return r.json()

def auth_check():
    # Simple, reliable auth test (v1):
    url = f"{SITE}/wiki/rest/api/user/current"
    data = get(url)
    print("✅ Auth OK as:", data.get("displayName") or data.get("username") or "unknown user")

def list_pages(limit=25):
    # Use CQL search to get pages you can see (works even with personal space)
    url = f"{SITE}/wiki/rest/api/search"
    data = get(url, params={"cql": "type=page ORDER BY created DESC", "limit": str(limit)})
    pages = []
    for r in data.get("results", []):
        c = r.get("content") or {}
        if c.get("type") == "page":
            pages.append({"id": c.get("id"), "title": c.get("title")})
    return pages

def read_page_storage(page_id: str):
    # Read storage body + version (needed if you later update)
    url = f"{SITE}/wiki/rest/api/content/{page_id}"
    data = get(url, params={"expand": "body.storage,version"})
    title = data.get("title")
    storage = data.get("body", {}).get("storage", {}).get("value", "")
    version = data.get("version", {}).get("number")
    return title, storage, version

if __name__ == "__main__":
    print("Site:", SITE)
    print("Email:", EMAIL)
    auth_check()

    pages = list_pages(limit=25)
    if not pages:
        print("No pages found. If you’re sure one exists, it may be restricted—try opening it in the browser to confirm.")
        sys.exit(0)

    print("\nPages:")
    for p in pages:
        print(f"- {p['id']} :: {p['title']}")

    # Read the first page (since you said you have just one)
    pid = pages[0]["id"]
    print(f"\nReading page {pid} …")
    title, storage, version = read_page_storage(pid)
    print("Title:", title)
    print("Version:", version)
    print("\nStorage body (first 1000 chars):\n")
    print(storage[:1000])
