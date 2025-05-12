import uuid
import time
from io import BytesIO
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
import pandas as pd

from final_optimization import genetic_algorithm  # ensure this path is correct

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
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    # Initialize job state
    jobs[job_id] = {"state": "running", "progress": 0}

    # Read uploaded file content into memory
    content = await file.read()

    def optimization_task(data_bytes: bytes, jid: str):
        start_time = time.time()
        try:
            # Load DataFrame from bytes
            df = pd.read_csv(BytesIO(data_bytes))

            # Call your genetic algorithm (adjust args as needed)
            # peel, initial_order, acceleration_length, num_changes, drag_adv, df, rider_data, W_rem
            # You may want to extract these from `df` or set defaults
            peel = df.attrs.get("peel")  # example: reading from sheet metadata
            order = df.attrs.get("initial_order")
            accel_len = df.attrs.get("acceleration_length")
            num_changes = df.attrs.get("num_changes")
            drag_adv = df.attrs.get("drag_adv", [1, 0.58, 0.52, 0.53])
            rider_data = None
            W_rem = None

            # Run optimizer
            total_time, switch_schedule, schedule_dict = genetic_algorithm(
                peel=peel,
                initial_order=order,
                acceleration_length=accel_len,
                num_changes=num_changes,
                drag_adv=drag_adv,
                df=df,
                rider_data=rider_data,
                W_rem=W_rem
            )

            # Build top results (e.g., best 5 schedules)
            sorted_items = sorted(schedule_dict.items(), key=lambda x: x[1])
            top_results = []
            for sched, t in sorted_items[:5]:
                top_results.append({"schedule": {str(sched): t}})

            # Update job store
            jobs[jid].update({
                "state": "done",
                "runtime_seconds": time.time() - start_time,
                "total_races_simulated": len(schedule_dict),
                "top_results": top_results,
            })
        except Exception as e:
            jobs[jid]["state"] = "error"
            jobs[jid]["error"] = str(e)

    # Launch background task
    background_tasks.add_task(optimization_task, content, job_id)

    return {"job_id": job_id}

@app.get("/run_optimization/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll this endpoint to retrieve the status and eventually the results.
    """
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
