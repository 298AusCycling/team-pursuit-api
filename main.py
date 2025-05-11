from fastapi import FastAPI, BackgroundTasks
from datetime import datetime
from final_optimization import genetic_algorithm
import itertools
from googleapiclient import discovery
from google.auth import compute_engine
import time
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
jobs: dict[str, dict] = {}        # job_id ➜ {"state": "...", "progress": 0-100, "result": …}

class OptRequest(BaseModel):
    workbook: str
    rider_ids: list[int]
    drag_adv: list[float]
    rho: float
    Crr: float
    v0: float

def run_opt_job(job_id: str):
    try:
        t0 = time.time()
        jobs[job_id] = {"state": "running", "progress": 0}

        tasks = [
            (al, peel, order, chg)
            for al in [3, 4]
            for peel in range(10, 33)
            for order in itertools.permutations([0, 1, 2, 3])
            for chg in [3, 5]
        ]
        total = len(tasks)
        total_races = 0
        results = []

        with ProcessPoolExecutor() as pool:
            for i, res in enumerate(pool.map(simulate_one, tasks), 1):
                if res["success"]:
                    results.append(res["result"])        # <-- full optimiser tuple
                    total_races += res["races"]          # ← each simulate_one returns how many inner races it ran

        top5 = sorted(results, key=lambda x: x[1])[:5]
        runtime = time.time() - t0
        jobs[job_id] = {
            "state": "done",
            "progress": 100,
            "runtime_seconds": runtime,
            "total_races_simulated": total_races,
            "top_results": [
            {
                "time": t,
                "switches": sched[0],               # (4, 8, 13, 20, …)
                "initial_order": sched[2:6],        # (4, 1, 3, 2)
                "peel": sched[-1],                  # 20
            }
            for sched, t in top5
        ],
        }

        # Optional – fire and forget
        Thread(target=trigger_shutdown, daemon=True).start()

    except Exception as e:
        jobs[job_id] = {"state": "error", "error": str(e)}
def simulate_one(args):
    accel_len, peel, order, changes = args
    try:
        time_race, schedule_tuple, _ = genetic_algorithm(
            peel=peel,
            initial_order=list(order),
            acceleration_length=accel_len,
            num_changes=changes,
            num_children=10,
            num_seeds=4,
            num_rounds=5,
        )
        # genetic_algorithm simulates (10 children × 5 rounds) = 50 races
        return {
            "success": True,
            "result": (schedule_tuple, time_race),
            "races": 50,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "races": 0}
def shutdown_vm(project_id, zone, instance_name):
    credentials = compute_engine.Credentials()
    service = discovery.build('compute', 'v1', credentials=credentials)
    request = service.instances().stop(project=project_id, zone=zone, instance=instance_name)
    response = request.execute()
    print("🛑 Shutdown request sent.")
    return response

def trigger_shutdown():
    print("🕒 Waiting 15 seconds before shutdown...")
    time.sleep(15)
    shutdown_vm("team-pursuit-optimizer", "us-central1-f", "optimization-backend")

@app.post("/run_optimization")
def run_optimization(req: OptRequest, background: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "state": "queued",
        "ctx": {
            "df": pd.read_json(req.workbook, orient="split"),
            "rider_ids": req.rider_ids,
            "drag_adv": req.drag_adv,
            "rho": req.rho,
            "Crr": req.Crr,
            "v0": req.v0,
        },
    }
    background.add_task(run_opt_job, job_id)
    return {"job_id": job_id}

@app.get("/run_optimization/{job_id}")
def optimisation_status(job_id: str):
    """Return current state / progress or 404 if unknown."""
    if job_id not in jobs:
        return {"error": "job_id not found"}
    return jobs[job_id]
