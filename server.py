import uuid
import time
import traceback
from io import BytesIO
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
import pandas as pd

from final_optimization import genetic_algorithm  # ensure correct import path

app = FastAPI(title="Team Pursuit Optimizer API")

# In-memory job store: {job_id: {'state': str, 'progress': int, ...}}
jobs: Dict[str, Dict[str, Any]] = {}

@app.post("/run_optimization")
async def run_optimization(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accepts an Excel file, starts an optimization job in the background,
    and returns a job_id immediately.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"state": "running", "progress": 0}

    content = await file.read()

    def optimization_task(data_bytes: bytes, jid: str):
        start_time = time.time()
        try:
            # Attempt to read Excel with openpyxl engine explicitly
            df = pd.read_csv(BytesIO(data_bytes), engine="openpyxl")

            # Extract any necessary parameters from df or defaults
            peel = df.attrs.get("peel")
            order = df.attrs.get("initial_order")
            accel_len = df.attrs.get("acceleration_length")
            num_changes = df.attrs.get("num_changes")
            drag_adv = df.attrs.get("drag_adv", [1, 0.58, 0.52, 0.53])

            # Run optimizer
            total_time, switch_schedule, schedule_dict = genetic_algorithm(
                peel=peel,
                initial_order=order,
                acceleration_length=accel_len,
                num_changes=num_changes,
                drag_adv=drag_adv,
                df=df,
            )

            # Build top results
            sorted_items = sorted(schedule_dict.items(), key=lambda x: x[1])
            top_results = [{"schedule": {str(s): t}} for s, t in sorted_items[:5]]

            jobs[jid].update({
                "state": "done",
                "runtime_seconds": time.time() - start_time,
                "total_races_simulated": len(schedule_dict),
                "top_results": top_results,
            })
        except Exception:
            # Capture full traceback for debugging
            tb = traceback.format_exc()
            jobs[jid]["state"] = "error"
            jobs[jid]["error"] = tb

    background_tasks.add_task(optimization_task, content, job_id)
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
        # state == "error"
        return {"state": state, "error": job.get("error", "Unknown error")}
