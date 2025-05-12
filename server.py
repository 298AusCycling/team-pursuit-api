import uuid
import time
import traceback
from io import BytesIO, StringIO
from typing import Dict, Any
import json

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
import pandas as pd

from final_optimization import genetic_algorithm  # ensure correct import path

app = FastAPI(title="Team Pursuit Optimizer API")

# In-memory job store: {job_id: {'state': str, 'progress': int, ...}}
jobs: Dict[str, Dict[str, Any]] = {}

@app.post("/run_optimization")
async def run_optimization(
    background_tasks: BackgroundTasks,
    peel: int = Form(..., description="Half-lap index where the third rider peels off"),
    initial_order: str = Form(..., description="Comma-separated initial rider order, e.g. '1,2,3,4'"),
    acceleration_length: int = Form(..., description="Length of the acceleration phase in half-laps"),
    num_changes: int = Form(..., description="Number of switch changes to evaluate"),
    drag_adv: str = Form(None, description="JSON array of drafting advantages, e.g. '[1,0.58,0.52,0.53]'. Uses default if omitted."),
    file: UploadFile = File(...),
):
    """
    Starts an optimization job in the background using both file data and form parameters.
    Returns a job_id immediately.
    """
    # Generate job ID and initial state
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"state": "running", "progress": 0}

    # Read uploaded file bytes
    content = await file.read()
    filename = file.filename or ""
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''

    def optimization_task(
        data_bytes: bytes,
        fname: str,
        peel_val: int,
        init_order_str: str,
        accel_len: int,
        changes: int,
        drag_adv_str: str,
        jid: str
    ):
        start_time = time.time()

        # Parse DataFrame from file
        df = None
        parse_error = None
        # Determine parser by extension
        if ext in ("xlsx", "xls"):
            try:
                df = pd.read_excel(BytesIO(data_bytes), engine="openpyxl")
            except Exception as e:
                parse_error = f"Excel parse error: {e}"
        elif ext == "csv":
            try:
                df = pd.read_csv(BytesIO(data_bytes))
            except Exception as e:
                parse_error = f"CSV parse error: {e}"
        else:
            # Fallback: try excel, then csv
            try:
                df = pd.read_excel(BytesIO(data_bytes), engine="openpyxl")
            except Exception:
                try:
                    df = pd.read_csv(BytesIO(data_bytes))
                except Exception as e:
                    parse_error = f"General parse error: {e}"

        # Additional CSV sniffing
        if df is None:
            try:
                text = data_bytes.decode('utf-8', errors='ignore')
                df = pd.read_csv(StringIO(text), sep=None, engine='python')
            except Exception as e:
                parse_error = f"CSV sniff parse error: {e}"

        # Validate DataFrame
        if df is None or df.empty:
            err_msg = parse_error or "No data parsed from file."
            jobs[jid]["state"] = "error"
            jobs[jid]["error"] = err_msg
            return

        # Parse form parameters
        try:
            # initial_order given as comma-separated string
            order_list = [int(i) for i in init_order_str.split(',')]
        except Exception:
            jobs[jid]["state"] = "error"
            jobs[jid]["error"] = f"Invalid initial_order format: '{init_order_str}'"
            return

        # Drag advantage list parsing or default
        if drag_adv_str:
            try:
                drag_adv_list = json.loads(drag_adv_str)
            except Exception:
                jobs[jid]["state"] = "error"
                jobs[jid]["error"] = f"Invalid drag_adv JSON: '{drag_adv_str}'"
                return
        else:
            drag_adv_list = [1, 0.58, 0.52, 0.53]

        # Run the genetic algorithm
        try:
            time_of_race, switch_schedule, schedule_dict = genetic_algorithm(
                peel=peel_val,
                initial_order=order_list,
                acceleration_length=accel_len,
                num_changes=changes,
                drag_adv=drag_adv_list,
                df=df,
                rider_data=None,
                W_rem=None
            )

            # Collect top results
            sorted_items = sorted(schedule_dict.items(), key=lambda x: x[1])
            top_results = [{"schedule": {str(s): t}} for s, t in sorted_items[:5]]

            jobs[jid].update({
                "state": "done",
                "runtime_seconds": time.time() - start_time,
                "total_races_simulated": len(schedule_dict),
                "top_results": top_results,
            })
        except Exception:
            tb = traceback.format_exc()
            jobs[jid]["state"] = "error"
            jobs[jid]["error"] = tb

    # Launch background task with all parameters
    background_tasks.add_task(
        optimization_task,
        content,
        filename,
        peel,
        initial_order,
        acceleration_length,
        num_changes,
        drag_adv,
        job_id
    )

    return {"job_id": job_id}

@app.get("/run_optimization/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    state = job["state"]
    if state == "running":
        return {"state": state, "progress": job.get("progress", 0)}
    elif state == "done":
        return {
            "state": state,
            "runtime_seconds": job["runtime_seconds"],
            "total_races_simulated": job["total_races_simulated"],
            "top_results": job["top_results"],
        }
    else:
        # state == 'error'
        return {"state": state, "error": job.get("error", "Unknown error")}