import uuid
import time
import traceback
from io import BytesIO, StringIO
from typing import Dict, Any, List
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
    initial_order: str = Form(..., description="Comma-separated initial rider order, e.g. '1,2,3,4'"),
    acceleration_length: int = Form(..., description="Length of the acceleration phase in half-laps"),
    peel_min: int = Form(..., description="Minimum peel location (half-lap index) to sweep"),
    peel_max: int = Form(..., description="Maximum peel location (half-lap index) to sweep"),
    changes_min: int = Form(..., description="Minimum number of switch changes to sweep"),
    changes_max: int = Form(..., description="Maximum number of switch changes to sweep"),
    drag_adv: str = Form(None, description="JSON array of drafting advantages, e.g. '[1,0.58,0.52,0.53]'. Uses default if omitted."),
    file: UploadFile = File(...),
):
    """
    Starts an optimization job sweeping peel and switch-change ranges in the background.
    Returns a job_id immediately.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"state": "running", "progress": 0}

    # Read uploaded file bytes
    content = await file.read()
    filename = file.filename or ""
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''

    def optimization_task(
        data_bytes: bytes,
        fname: str,
        order_str: str,
        accel_len: int,
        p_min: int,
        p_max: int,
        c_min: int,
        c_max: int,
        drag_adv_str: str,
        jid: str
    ):
        start_time = time.time()
        try:
            # Parse DataFrame
            if ext in ("xlsx", "xls"):
                df = pd.read_excel(BytesIO(data_bytes), engine="openpyxl")
            else:
                try:
                    df = pd.read_csv(BytesIO(data_bytes))
                except Exception:
                    text = data_bytes.decode('utf-8', errors='ignore')
                    df = pd.read_csv(StringIO(text), sep=None, engine='python')

            if df.empty:
                raise ValueError("No data parsed from file.")

            # Parse inputs
            order_list = [int(i) for i in order_str.split(',')]
            if drag_adv_str:
                drag_adv_list = json.loads(drag_adv_str)
            else:
                drag_adv_list = [1, 0.58, 0.52, 0.53]

            # Build rider_data and initial W'
            rider_data = {}
            W_rem = {}
            for rider_id in order_list:
                mask = df['Name'].astype(str).str.contains(f"M{rider_id}")
                if not mask.any():
                    raise ValueError(f"No data found for rider M{rider_id}")
                row = df[mask].iloc[0]
                Wp = row.get("W'", None)
                CP = row.get("CP", None)
                Pmax = row.get("Pmax", None)
                AC = row.get("CdA", None)
                m_rider = row.get("Mass", None)
                if None in (Wp, CP, Pmax, AC, m_rider):
                    raise ValueError(f"Incomplete data for rider M{rider_id}")
                rider_data[rider_id] = {"W_prime": Wp, "CP": CP, "AC": AC, "Pmax": Pmax, "m_rider": m_rider}
                W_rem[rider_id] = Wp

            # Sweep over ranges and collect best times
            results: List[Dict[str, Any]] = []
            total_runs = 0
            for peel_val in range(p_min, p_max + 1):
                for changes in range(c_min, c_max + 1):
                    total_runs += 1
                    try:
                        t_race, switch_sched, sched_dict = genetic_algorithm(
                            peel=peel_val,
                            initial_order=order_list,
                            acceleration_length=accel_len,
                            num_changes=changes,
                            drag_adv=drag_adv_list,
                            df=df,
                            rider_data=rider_data,
                            W_rem=W_rem
                        )
                        # record the best schedule for this config
                        best_sched, best_time = min(sched_dict.items(), key=lambda x: x[1])
                        results.append({
                            "peel": peel_val,
                            "num_changes": changes,
                            "initial_order": order_list,
                            "switches": list(best_sched),
                            "time": best_time
                        })
                    except Exception as e:
                        # skip failed runs but record error
                        continue

            # Sort overall and take top 5
            sorted_results = sorted(results, key=lambda x: x['time'])[:5]

            jobs[jid].update({
                "state": "done",
                "runtime_seconds": time.time() - start_time,
                "total_races_simulated": total_runs,
                "top_results": sorted_results,
            })
        except Exception:
            tb = traceback.format_exc()
            jobs[jid]["state"] = "error"
            jobs[jid]["error"] = tb

    background_tasks.add_task(
        optimization_task,
        content,
        filename,
        initial_order,
        acceleration_length,
        peel_min,
        peel_max,
        changes_min,
        changes_max,
        drag_adv,
        job_id
    )
    return {"job_id": job_id}

@app.get("/run_optimization/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    state = job['state']
    if state == 'running':
        return {'state': state, 'progress': job.get('progress', 0)}
    if state == 'done':
        return {
            'state': state,
            'runtime_seconds': job['runtime_seconds'],
            'total_races_simulated': job['total_races_simulated'],
            'top_results': job['top_results'],
        }
    # error
    return {'state': state, 'error': job.get('error', 'Unknown error')}
