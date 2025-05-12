from fastapi import FastAPI, BackgroundTasks
from datetime import datetime
from optimization import genetic_algorithm
import itertools
from googleapiclient import discovery
from google.auth import compute_engine
import time
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid

app = FastAPI()
jobs: dict[str, dict] = {}        # job_id ➜ {"state": "...", "progress": 0-100, "result": …}
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
                    results.append((res["schedule"], res["time"]))
                    total_races += 1            # or += 50 if that’s the real count
                jobs[job_id]["progress"] = int(i / total * 100)

        top5 = sorted(results, key=lambda x: x[1])[:5]
        runtime = time.time() - t0
        jobs[job_id] = {
            "state": "done",
            "progress": 100,
            "runtime_seconds": runtime,
            "total_races_simulated": total_races,
            "top_results": [
                {"schedule": {str(k): v for k, v in sched.items()},
                 "time": t}
                for sched, t in top5
            ],
        }

        # Optional – fire and forget
        Thread(target=trigger_shutdown, daemon=True).start()

    except Exception as e:
        jobs[job_id] = {"state": "error", "error": str(e)}
def simulate_one(args):
    try:
        _, best_time, best_schedule = genetic_algorithm(
            acceleration_length=args[0],
            peel=args[1],
            num_changes=args[3],
            initial_order=list(args[2]),
            num_children=10,
            num_seeds=4,
            num_rounds=5,
        )
        return {
            "success": True,
            "schedule": best_schedule,
            "time": best_time
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "peel": args[1],
            "order": args[2]
        }

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
def run_optimization(background_tasks: BackgroundTasks):   
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_opt_job, job_id)       
    return {"job_id": job_id}

@app.get("/run_optimization/{job_id}")
def optimisation_status(job_id: str):
    """Return current state / progress or 404 if unknown."""
    if job_id not in jobs:
        return {"error": "job_id not found"}
    return jobs[job_id]