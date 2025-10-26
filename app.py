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

app = FastAPI(title="Confluence and Zoho PDF Merger API")


# ----------------------------
# Helpers
# ----------------------------
def unzipdir(zip_file: UploadFile, target_dir: Path) -> None:
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


def findpdfs(directory: Path) -> List[Path]:
    #Find PDFs in case of zipped nested folders.
    return sorted([p for p in directory.rglob("*.pdf") if p.is_file()])


def parserealoutputdir(stdout_text: str) -> Optional[Path]:
    m = re.search(r"Output dir\s*:\s*(.+)", stdout_text)
    if not m:
        return None
    
    p = Path(m.group(1).strip())
    return p if p.exists() else None

# Endpoint
@app.post("/merge")
def merge_docs(
    confluence_zip: UploadFile = File(..., description="ZIP containing *.pdf from Confluence"),
    zoho_zip: UploadFile = File(..., description="ZIP containing *.pdf from Zoho"),
    openaiapi: Optional[str] = Form(None, description="OpenAI API key)"),
    
):

    # Workspace
    work = Path(tempfile.mkdtemp(prefix="mergejob_")).resolve()
    c_dir = work / "confluence"
    z_dir = work / "zoho"
    tmp_out = work / "out"  
    for d in (c_dir, z_dir, tmp_out):
        d.mkdir(parents=True, exist_ok=True)

    # Unpack uploads
    unzipdir(confluence_zip, c_dir)
    unzipdir(zoho_zip, z_dir)

    # check pdfs exist in both
    if not findpdfs(c_dir):
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, "No PDFs found inside confluence_zip.")
    if not findpdfs(z_dir):
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, "No PDFs found inside zoho_zip.")

    # Script path (to run as subprocess)
    script_name = "Task_Sol1.py"  
    script_path = (Path(__file__).parent / script_name).resolve()
    if not script_path.exists():
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"Merger script not found at: {script_path}")

    # Env for the child process
    env = os.environ.copy()
    env["confluence_directory"] = str(c_dir)     
    env["zoho_directory"] = str(z_dir)
    env["output_directory"] = str(tmp_out)
    env["PYTHONIOENCODING"] = "utf-8"

    # Run the script as a subprocess
    try:
        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            timeout=60 * 15,  # waiting time
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(504, "Merging timed out. Try fewer/smaller PDFs.")

    # If the child failed
    if result.returncode != 0:
        # print logs 
        print("Child output")
        print(result.stdout)
        print(result.stderr)
        err = (result.stderr or "").strip() or "Unknown error"
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"{script_path.name} failed:\n{err}")


    realoutputdir = parserealoutputdir(result.stdout)
    out_dir = realoutputdir if realoutputdir else tmp_out
    pdfs = findpdfs(out_dir)

    if not pdfs:
        # maintain logs for debugging
        logs = {
            "stdout": result.stdout[-4000:],  # tail for brevity
            "stderr": result.stderr[-4000:],
        }
        shutil.rmtree(work, ignore_errors=True)
        return JSONResponse({"message": "No output files produced.", "logs": logs}, status_code=200)

    # If only one PDF
    if len(pdfs) == 1:
        pdf_path = pdfs[0]
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=pdf_path.name,
        )

    # more, zip all and return 
    out_zip = work / f"outputs_{uuid.uuid4().hex[:8]}.zip"
    _zip_dir(out_dir, out_zip)
    return FileResponse(
        path=str(out_zip),
        media_type="application/zip",
        filename=out_zip.name,
    )
