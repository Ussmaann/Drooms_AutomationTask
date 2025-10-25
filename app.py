# app.py
import os
import re
import sys
import uuid
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Confluence↔Zoho PDF Merger API")


# ----------------------------
# Helpers
# ----------------------------
def _unzip_to_dir(zip_file: UploadFile, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(zip_file.file.read())
            tmp_path = Path(tmp.name)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail=f"{zip_file.filename} is not a valid ZIP.")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _zip_dir(src_dir: Path, out_zip_path: Path) -> None:
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))


def _find_pdfs_anywhere(directory: Path) -> List[Path]:
    """Find PDFs even if the user zipped nested folders."""
    return sorted([p for p in directory.rglob("*.pdf") if p.is_file()])


def _parse_real_output_dir(stdout_text: str) -> Optional[Path]:
    """
    Your script prints a line like:
      Output dir    : C:\...\output
    We parse that to follow the *actual* directory it wrote to.
    """
    m = re.search(r"Output dir\s*:\s*(.+)", stdout_text)
    if not m:
        return None
    # Windows paths with spaces are fine; strip trailing line breaks
    p = Path(m.group(1).strip())
    return p if p.exists() else None


# ----------------------------
# Endpoint
# ----------------------------
@app.post("/merge")
def merge_docs(
    confluence_zip: UploadFile = File(..., description="ZIP containing *.pdf from Confluence"),
    zoho_zip: UploadFile = File(..., description="ZIP containing *.pdf from Zoho"),
    openaiapi: Optional[str] = Form(None, description="OpenAI API key (optional)"),
    openaibaseurl: Optional[str] = Form(None, description="OpenAI base URL (optional)"),
):
    """
    Upload two ZIPs, run your merging script, and return:
      - A single PDF if only one is produced
      - A ZIP if multiple outputs exist
    """

    # Workspace
    work = Path(tempfile.mkdtemp(prefix="mergejob_")).resolve()
    c_dir = work / "confluence"
    z_dir = work / "zoho"
    tmp_out = work / "out"  # a default output dir we pass to the child
    for d in (c_dir, z_dir, tmp_out):
        d.mkdir(parents=True, exist_ok=True)

    # Unpack uploads
    _unzip_to_dir(confluence_zip, c_dir)
    _unzip_to_dir(zoho_zip, z_dir)

    # Sanity: must contain PDFs (accept nested)
    if not _find_pdfs_anywhere(c_dir):
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, "No PDFs found inside confluence_zip.")
    if not _find_pdfs_anywhere(z_dir):
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, "No PDFs found inside zoho_zip.")

    # Script path (configurable via env MERGER_SCRIPT)
    script_name = os.getenv("MERGER_SCRIPT", "Task_Sol1.py")  # change default if your file is named differently
    script_path = (Path(__file__).parent / script_name).resolve()
    if not script_path.exists():
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"Merger script not found at: {script_path}")

    # Env for the child process (force UTF-8 to avoid emoji encoding issues)
    env = os.environ.copy()
    env["confluence_directory"] = str(c_dir)     # child script reads these
    env["zoho_directory"] = str(z_dir)
    env["output_directory"] = str(tmp_out)
    if openaiapi:
        env["openaiapi"] = openaiapi
    if openaibaseurl:
        env["openaibaseurl"] = openaibaseurl
    env["PYTHONIOENCODING"] = "utf-8"

    # Run the script as a subprocess (avoid import-time sys.exit crashes)
    try:
        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            timeout=60 * 30,  # 30 minutes, adjust as needed
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(504, "Merging timed out. Try fewer/smaller PDFs.")

    # If the child failed, expose its logs
    if result.returncode != 0:
        # print logs to server console for debugging
        print("==== child STDOUT ====")
        print(result.stdout)
        print("==== child STDERR ====")
        print(result.stderr)
        err = (result.stderr or "").strip() or "Unknown error"
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"{script_path.name} failed:\n{err}")

    # Figure out where outputs were actually written
    # 1) Prefer the directory printed by the child ("Output dir : ...")
    real_out_dir = _parse_real_output_dir(result.stdout)
    # 2) Fallback to the temp out dir we passed in
    out_dir = real_out_dir if real_out_dir else tmp_out

    # Collect PDFs from the resolved output directory
    pdfs = _find_pdfs_anywhere(out_dir)

    if not pdfs:
        # Return logs so you can see what happened
        logs = {
            "stdout": result.stdout[-4000:],  # tail for brevity
            "stderr": result.stderr[-4000:],
        }
        shutil.rmtree(work, ignore_errors=True)
        return JSONResponse({"message": "No output files produced.", "logs": logs}, status_code=200)

    # If only one PDF, return it directly
    if len(pdfs) == 1:
        pdf_path = pdfs[0]
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=pdf_path.name,
        )

    # Otherwise, zip them all and return a ZIP
    out_zip = work / f"outputs_{uuid.uuid4().hex[:8]}.zip"
    _zip_dir(out_dir, out_zip)
    return FileResponse(
        path=str(out_zip),
        media_type="application/zip",
        filename=out_zip.name,
    )
