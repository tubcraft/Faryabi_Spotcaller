"""
app.py  —  Faryabi Spot Calling  v3  —  Web GUI backend
════════════════════════════════════════════════════════════════════════════════

Start:
    conda activate bigfish
    python app.py --host 0.0.0.0 --port 5000 --username user --password secret

Routes:
    GET  /               GUI
    GET  /viewer         Spot viewer
    GET  /docs           Documentation
    POST /api/run        Submit a job
    GET  /api/jobs       List jobs
    GET  /api/jobs/<id>  Job details + log
    GET  /api/stream/<id>  SSE live log
    GET  /api/logs/<id>  Download log file
    POST /api/cancel/<id>  Cancel a running job
    GET  /api/browse     Directory browser (path autocomplete)
    GET  /api/models     Model list from Piscis_models / Cellpose_models
    GET  /api/output_files  List files in an output dir (for viewer auto-load)
    GET  /api/output_file   Serve a single output file
"""

import argparse
import hashlib
import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory

app = Flask(__name__)

# ── Auth ──────────────────────────────────────────────────────────────────────

def _check_auth(username: str, password: str) -> bool:
    cfg_u = app.config.get("AUTH_USERNAME", "")
    cfg_p = app.config.get("AUTH_PASSWORD", "")
    if not cfg_u or not cfg_p:
        return True
    return (secrets.compare_digest(username.encode(), cfg_u.encode()) and
            secrets.compare_digest(
                hashlib.sha256(password.encode()).hexdigest(),
                hashlib.sha256(cfg_p.encode()).hexdigest()))


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not app.config.get("AUTH_USERNAME"):
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not _check_auth(auth.username, auth.password):
            return Response("Authentication required.", 401,
                            {"WWW-Authenticate": 'Basic realm="Faryabi Spot Calling"'})
        return f(*args, **kwargs)
    return decorated


# ── Job store ─────────────────────────────────────────────────────────────────

jobs: dict[str, dict]        = {}
log_queues: dict[str, queue.Queue] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_script() -> Path:
    for p in [Path(app.config.get("SCRIPT_PATH", "")),
              Path(__file__).parent / "run_piscis.py",
              Path(__file__).parent.parent / "run_piscis.py"]:
        if p.is_file():
            return p
    return Path("run_piscis.py")


def build_cmd(params: dict) -> list[str]:
    """Convert GUI params dict into a run_piscis.py command list."""
    det = params.get("detector", "piscis")
    cmd = [
        sys.executable, str(find_script()),
        "--input_dir",  params["input_dir"],
        "--output_dir", params["output_dir"],
        "--detector",   det,
        "--projection", params.get("projection", "max"),
        "--workers",    str(max(1, min(int(params.get("workers", 4)), 20))),
        "--percentile_low",  str(params.get("percentile_low",  1.0)),
        "--percentile_high", str(params.get("percentile_high", 99.5)),
    ]
    if det == "piscis":
        cmd += ["--model",     params.get("model", "20230905")]
        cmd += ["--threshold", str(params.get("threshold", 0.5))]
        if params.get("stack"):
            cmd.append("--stack")
    else:
        cmd += ["--threshold",        str(params.get("threshold", 5.0))]
        cmd += ["--spot_yx",          str(params.get("spot_yx", 3.0))]
        cmd += ["--spot_z",           str(params.get("spot_z",  2.0))]
        cmd += ["--min_snr",          str(params.get("min_snr", 2.0))]
        cmd += ["--max_sigma_ratio",  str(params.get("max_sigma_ratio", 2.5))]
        cmd += ["--tile_size",        str(params.get("tile_size", 128))]
        if params.get("use_3d"):
            cmd.append("--use_3d")

    if params.get("cp_model"):
        cmd += ["--cellpose_model", params["cp_model"]]
    if float(params.get("cell_diameter", 0)) > 0:
        cmd += ["--cellpose_diam", str(params["cell_diameter"])]
    if int(params.get("expand_mask", 0)) > 0:
        cmd += ["--expand_mask", str(int(params["expand_mask"]))]

    run_only = params.get("run_only", "both")
    if run_only != "both":
        cmd += ["--run_only", run_only]
    if params.get("pattern", "*.tif") != "*.tif":
        cmd += ["--pattern", params["pattern"]]
    if params.get("no_gpu"):
        cmd.append("--no_gpu")
    if params.get("save_previews", True):
        cmd.append("--save_previews")

    return cmd


def run_job(job_id: str, cmd: list[str]):
    """Execute pipeline subprocess, stream output to SSE queue and job store."""
    job = jobs[job_id]
    q   = log_queues[job_id]
    job["status"] = "running"
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        job["pid"] = proc.pid
        for line in proc.stdout:
            line = line.rstrip()
            job["log_lines"].append(line)
            if "Log file:" in line and not job.get("log_file"):
                job["log_file"] = line.split("Log file:")[-1].strip()
            try: q.put_nowait(line)
            except queue.Full: pass
        proc.wait()
        job["returncode"] = proc.returncode
        job["status"]     = "done" if proc.returncode == 0 else "error"
    except Exception as exc:
        job["log_lines"].append(f"[GUI ERROR] {exc}")
        job["status"]     = "error"
        job["returncode"] = -1
    finally:
        job["finished_at"] = time.time()
        q.put(None)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
@require_auth
def index():
    return render_template("index.html")


@app.route("/viewer")
@require_auth
def viewer():
    for p in [Path(app.config.get("VIEWER_PATH","")),
              Path(__file__).parent / "spot_viewer.html",
              Path(__file__).parent.parent / "spot_viewer.html"]:
        if p.is_file():
            return send_from_directory(str(p.parent), p.name)
    return "spot_viewer.html not found", 404


@app.route("/docs")
@require_auth
def docs():
    for p in [Path(__file__).parent / "docs.html",
              Path(__file__).parent.parent / "docs.html"]:
        if p.is_file():
            return send_from_directory(str(p.parent), p.name)
    return "docs.html not found", 404


@app.route("/api/run", methods=["POST"])
@require_auth
def api_run():
    params = request.json or {}
    for field in ("input_dir", "output_dir"):
        if not params.get(field):
            return jsonify({"error": f"Missing: {field}"}), 400
    if params.get("detector","piscis") == "piscis" and not params.get("model"):
        return jsonify({"error": "Piscis requires a model"}), 400

    job_id = str(uuid.uuid4())[:8]
    cmd    = build_cmd(params)
    jobs[job_id] = {
        "id": job_id, "status": "queued", "cmd": " ".join(cmd),
        "params": params, "created_at": time.time(), "finished_at": None,
        "output_dir": params["output_dir"], "log_lines": [],
        "log_file": None, "returncode": None, "pid": None,
    }
    log_queues[job_id] = queue.Queue(maxsize=4000)
    threading.Thread(target=run_job, args=(job_id, cmd),
                     name=f"job-{job_id}", daemon=True).start()
    return jsonify({"job_id": job_id, "cmd": " ".join(cmd)})


@app.route("/api/jobs")
@require_auth
def api_jobs():
    return jsonify([
        {"id": j["id"], "status": j["status"],
         "detector": j["params"].get("detector","piscis"),
         "input_dir": j["params"].get("input_dir"),
         "output_dir": j["output_dir"],
         "created_at": j["created_at"], "finished_at": j["finished_at"],
         "returncode": j["returncode"], "n_lines": len(j["log_lines"]),
         "has_log_file": bool(j.get("log_file"))}
        for j in sorted(jobs.values(), key=lambda x: x["created_at"], reverse=True)
    ])


@app.route("/api/jobs/<job_id>")
@require_auth
def api_job(job_id):
    j = jobs.get(job_id)
    if not j: return jsonify({"error": "not found"}), 404
    return jsonify({**{k: v for k, v in j.items() if k != "log_lines"},
                    "log_lines": j["log_lines"][-500:]})


@app.route("/api/stream/<job_id>")
@require_auth
def api_stream(job_id):
    j = jobs.get(job_id)
    if not j: return jsonify({"error": "not found"}), 404
    q = log_queues.get(job_id)

    def generate():
        for line in list(j["log_lines"]):
            yield f"data: {json.dumps(line)}\n\n"
        if j["status"] in ("done","error","cancelled") or q is None:
            yield "data: __DONE__\n\n"; return
        while True:
            try:
                line = q.get(timeout=30)
                if line is None: yield "data: __DONE__\n\n"; break
                yield f"data: {json.dumps(line)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.route("/api/logs/<job_id>")
@require_auth
def api_download_log(job_id):
    j = jobs.get(job_id)
    if not j: return jsonify({"error": "not found"}), 404
    log_file = j.get("log_file")
    if log_file and Path(log_file).is_file():
        return send_file(log_file, as_attachment=True, download_name=Path(log_file).name)
    content = "\n".join(j["log_lines"])
    return Response(content, mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename=job_{job_id}.log"})


@app.route("/api/cancel/<job_id>", methods=["POST"])
@require_auth
def api_cancel(job_id):
    j = jobs.get(job_id)
    if not j: return jsonify({"error": "not found"}), 404
    if j.get("pid") and j["status"] == "running":
        try:
            import signal; os.kill(j["pid"], signal.SIGTERM)
            j["status"] = "cancelled"
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True})


@app.route("/api/models")
@require_auth
def api_models():
    """List models from Piscis_models/ or Cellpose_models/ + HuggingFace built-ins."""
    kind       = request.args.get("kind", "piscis")
    script_dir = Path(find_script()).parent
    folder     = script_dir / ("Piscis_models" if kind == "piscis" else "Cellpose_models")

    ALLOWED  = {".pt", ".pkl", ".npz", ".h5", ".pth", ""}
    EXCLUDED = {".py",".tif",".tiff",".png",".jpg",".jpeg",
                ".csv",".log",".txt",".npy",".json",
                ".yaml",".yml",".md",".sh",".bat",".zip",".tar",".gz"}

    results = []
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if p.name.startswith("."): continue
            ext = p.suffix.lower()
            if ext in EXCLUDED: continue
            if p.is_dir() or ext in ALLOWED:
                results.append({"name": p.name, "path": str(p), "builtin": False})

    if kind == "piscis":
        builtins = [
            ("20251212",                "General — newest (Dec 2025)"),
            ("20230905",                "General — stable (Sep 2023)"),
            ("20230905_dice",           "General — DICE loss"),
            ("20230905_bce",            "General — BCE loss"),
            ("20230905_focal",          "General — focal loss"),
            ("20230709",                "General — Jul 2023"),
            ("20230616",                "General — Jun 2023"),
            ("deepblink_smfish_new",    "smFISH — updated"),
            ("deepblink_smfish",        "smFISH — original"),
            ("deepblink_particle_new",  "Particle — updated"),
            ("deepblink_particle",      "Particle — original"),
            ("deepblink_receptor_new",  "Receptor — updated"),
            ("deepblink_receptor",      "Receptor — original"),
            ("deepblink_suntag_new",    "SunTag — updated"),
            ("deepblink_suntag",        "SunTag — original"),
            ("deepblink_vesicle_new",   "Vesicle — updated"),
            ("deepblink_vesicle",       "Vesicle — original"),
            ("deepblink_microtubule_new","Microtubule — updated"),
            ("deepblink_microtubule",   "Microtubule — original"),
            ("spotiflow_general",          "Spotiflow — general"),
            ("spotiflow_synthetic_complex","Spotiflow — synthetic complex"),
        ]
        for model_id, label in builtins:
            results.append({"name": f"{model_id}  ({label})", "path": model_id, "builtin": True})

    return jsonify(results)


@app.route("/api/output_files")
@require_auth
def api_output_files():
    """
    List files in an output directory for the viewer auto-load.

    Handles the v3 run-subfolder structure automatically:
      /results/C1_left/                       ← base dir (passed by GUI)
        piscis_20251212_t0p50_20260319_.../   ← run subfolders
        spot_caller_threshold6_t6p00_.../

    If the requested path contains run subfolders (dirs matching
    piscis_* or spot_caller_*) but no image/CSV files itself,
    the response includes a "run_subfolders" list so the viewer
    can show a picker. The most recent run is also listed as
    "selected_run" for auto-load.

    If a specific run subfolder is requested directly, files are
    returned normally (no subfolder detection needed).
    """
    path = request.args.get("path", "").strip()
    if not path: return jsonify({"error": "path required"}), 400
    try:
        base = Path(path).expanduser().resolve()
        if not base.is_dir(): return jsonify({"error": f"Not a directory: {base}"}), 400

        # Detect run subfolders: dirs whose name starts with piscis_ or spot_caller_
        RUN_PREFIXES = ("piscis_", "spot_caller_")
        run_subdirs = sorted(
            [d for d in base.iterdir()
             if d.is_dir() and d.name.startswith(RUN_PREFIXES)],
            key=lambda d: d.stat().st_mtime,
            reverse=True,  # newest first
        )

        # Check if this base dir has actual pipeline outputs itself
        has_own_files = any(
            p.suffix.lower() in (".csv", ".png", ".tif", ".tiff")
            for p in base.iterdir() if p.is_file()
        )

        result = {"files": [], "run_subfolders": [], "selected_run": None}

        # If base has run subfolders and no files of its own → it's the parent dir
        if run_subdirs and not has_own_files:
            result["run_subfolders"] = [
                {"name": d.name, "path": str(d),
                 "mtime": d.stat().st_mtime}
                for d in run_subdirs
            ]
            result["selected_run"] = str(run_subdirs[0])  # most recent
            # Also return files from the most recent run for immediate display
            target = run_subdirs[0]
        else:
            # This is already a run subfolder (or a flat structure) — use directly
            target = base

        # List files in the target directory
        for p in sorted(target.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                result["files"].append({"name": p.name, "url": f"/api/output_file?path={p}"})
        previews = target / "previews"
        if previews.is_dir():
            for p in sorted(previews.iterdir()):
                if p.is_file() and not p.name.startswith("."):
                    result["files"].append({"name": p.name, "url": f"/api/output_file?path={p}"})

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/output_file")
@require_auth
def api_output_file():
    path = request.args.get("path", "").strip()
    if not path: return jsonify({"error": "path required"}), 400
    p = Path(path).expanduser().resolve()
    if not p.is_file(): return jsonify({"error": "not found"}), 404
    return send_file(str(p))


@app.route("/api/batch", methods=["POST"])
@require_auth
def api_batch():
    """
    Submit a batch of runs to execute sequentially.
    Body: {"runs": [{input_dir, output_dir, ...params}, ...]}
    Each run uses the same detector/model/threshold params but different I/O dirs.
    Returns a batch_id and list of job_ids created.
    """
    data        = request.json or {}
    runs        = data.get("runs", [])
    base_params = data.get("params", {})
    if not runs:
        return jsonify({"error": "No runs provided"}), 400

    batch_id = str(uuid.uuid4())[:8]
    job_ids  = []

    for i, run in enumerate(runs):
        p      = {**base_params, **run}   # per-run dirs override base params
        if not p.get("input_dir") or not p.get("output_dir"):
            continue
        cmd    = build_cmd(p)             # build once, store as list
        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {
            "id": job_id, "status": "queued",
            "cmd": " ".join(cmd), "_cmd_list": cmd,   # store both
            "params": p, "created_at": time.time(), "finished_at": None,
            "output_dir": p["output_dir"], "log_lines": [],
            "log_file": None, "returncode": None, "pid": None,
            "batch_id": batch_id, "batch_index": i,
            "detector": p.get("detector", "piscis"),
        }
        log_queues[job_id] = queue.Queue(maxsize=4000)
        job_ids.append(job_id)

    if not job_ids:
        return jsonify({"error": "No valid runs"}), 400

    def run_batch_sequential(jids):
        for jid in jids:
            if jobs[jid].get("status") == "cancelled":
                continue
            # Use the pre-built cmd list — never rebuild (timestamp would change)
            run_job(jid, jobs[jid]["_cmd_list"])

    threading.Thread(target=run_batch_sequential, args=(job_ids,),
                     name=f"batch-{batch_id}", daemon=True).start()

    return jsonify({"batch_id": batch_id, "job_ids": job_ids, "n_runs": len(job_ids)})


@app.route("/api/browse")
@require_auth
def api_browse():
    path = request.args.get("path", "").strip() or str(Path.home())
    try:
        p = Path(path).expanduser().resolve()
        while not p.exists() and p != p.parent: p = p.parent
        if p.is_file(): p = p.parent
        if not p.is_dir(): p = Path.home()
        entries = sorted(
            [{"name": e.name, "path": str(e), "is_dir": e.is_dir()}
             for e in p.iterdir() if not e.name.startswith(".")],
            key=lambda x: (not x["is_dir"], x["name"].lower()))
        return jsonify({"path": str(p), "entries": entries, "parent": str(p.parent)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faryabi Spot Calling v3 — GUI")
    parser.add_argument("--host",     default="0.0.0.0")
    parser.add_argument("--port",     default=5000, type=int)
    parser.add_argument("--script",   default="")
    parser.add_argument("--viewer",   default="")
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--username", default="",
                        help="HTTP Basic Auth username (omit to disable auth)")
    parser.add_argument("--password", default="",
                        help="HTTP Basic Auth password")
    args = parser.parse_args()

    app.config["SCRIPT_PATH"]   = args.script
    app.config["VIEWER_PATH"]   = args.viewer
    app.config["AUTH_USERNAME"] = args.username
    app.config["AUTH_PASSWORD"] = args.password

    print(f"\n  Faryabi Spot Calling v3  →  http://localhost:{args.port}")
    if args.username:
        print(f"  Auth enabled  →  user: {args.username}")
    else:
        print("  Auth disabled  (pass --username / --password to enable)")
    print()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
