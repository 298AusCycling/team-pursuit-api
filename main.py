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
import re
from typing import Tuple, Dict, Any

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
    ctx   = jobs[job_id]["ctx"]            # ← pull once
    df    = ctx["df"]
    r_ids = ctx["rider_ids"]
    try:
        t0 = time.time()
        jobs[job_id] = {"state": "running", "progress": 0}

        tasks = [
            (al, peel, order, chg, ctx)        # ← append ctx
            for al in [3, 4]
            for peel in range(10, 33)
            for order in itertools.permutations(r_ids)
            for chg in [3, 5]
        ]
        print("Prepared", len(tasks), "tasks")
        total_races = 0
        results = []
        print("Prepared", len(tasks), "tasks")
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
def simulate_one(args: Tuple[int, int, Tuple[int, ...], int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run one GA optimisation and package the result so `run_opt_job`
    can slice it cleanly later on.

    Parameters
    ----------
    args
        (accel_len, peel, order, changes, ctx) where
        * accel_len : int      – acceleration length (m)
        * peel      : int      – half-lap at which the lead rider peels
        * order     : tuple    – initial paceline order (rider IDs)
        * changes   : int      – number of rider changes allowed
        * ctx       : dict     – shared context (df, drag_adv, rider_ids, …)

    Returns
    -------
    dict
        {
            "success": True/False,
            "result" : (schedule_descr, race_time)  # only on success
            "races"  : <int>                        # GA inner sims
            "error"  : <str>                        # only on failure
        }
    """
    accel_len, peel, order, changes, ctx = args
    df        = ctx["df"]
    drag_adv  = ctx["drag_adv"]
    rider_ids = ctx["rider_ids"]

    # -----------------------------------------------------------------
    # helper: pull bio-mechanical data for each rider once
    # -----------------------------------------------------------------
    number_to_name = {
        int(m.group(1)): name
        for name in df["Name"]
        if (m := re.search(r"M(\d+)", name))
    }

    def info(rid: int) -> Dict[str, float]:
        row = df[df["Name"] == number_to_name[rid]].iloc[0]
        return {
            "W'":  float(row["W'"]) * 1000,  # convert kJ → J
            "CP":  float(row["CP"]),
            "CdA": float(row["CdA"]),
            "Pmax":float(row["Pmax"]),
            "mass":float(row["Mass"]),
        }

    rider_data = {rid: info(rid) for rid in rider_ids}
    W_rem      = [rider_data[r]["W'"] for r in rider_ids]

    # -----------------------------------------------------------------
    # main call – your GA returns (time, switch_tuple, _)
    # -----------------------------------------------------------------
    try:
        time_race, switch_tuple, _ = genetic_algorithm(
            peel              = peel,
            initial_order     = list(order),
            acceleration_length = accel_len,
            num_changes       = changes,
            drag_adv          = drag_adv,
            df                = df,
            rider_data        = rider_data,
            W_rem             = W_rem,
            num_children      = 10,
            num_seeds         = 4,
            num_rounds        = 5,
        )

        # -----------------------------------------------------------------
        # Build the *rich* tuple expected by `run_opt_job`
        #
        #   0 → switch tuple             (4, 8, 13, 20, …)
        #   1 → literal string           "initial order:"
        #   2–5 → four rider IDs         4, 1, 3, 2
        #   6 → literal string           "peel location:"
        #   7 → peel half-lap            20
        # -----------------------------------------------------------------
        schedule_descr = (
            switch_tuple,
            "initial order:", *order,
            "peel location:", peel,
        )

        return {
            "success": True,
            "result": (schedule_descr, time_race),
            "races": 50,   # one GA call covers 50 inner simulations
        }

    except Exception as e:
        print("simulate_one failed:", e)
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