from fastapi import FastAPI
from datetime import datetime
from optimization import genetic_algorithm
import itertools
from googleapiclient import discovery
from google.auth import compute_engine
import time
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, as_completed

app = FastAPI()

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
def run_optimization():
    start_time = datetime.now()
    print(f"🚀 Optimization triggered at {start_time}")

    try:
        tasks = []
        for acceleration_length in [3, 4]:
            for peel in range(10, 33):
                for initial_order in itertools.permutations([0, 1, 2, 3]):
                    for num_changes in [3, 5]:
                        tasks.append((acceleration_length, peel, initial_order, num_changes))

        all_results = []
        total_races = 0

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(simulate_one, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    all_results.append((result["schedule"], result["time"]))
                    total_races += 10 * 5
                else:
                    print(f"⚠️ Failed for order {result['order']}, peel {result['peel']}, error: {result['error']}")

        top_results = sorted(all_results, key=lambda x: x[1])[:5]
        runtime = (datetime.now() - start_time).total_seconds()
        print(f"✅ Optimization completed with {total_races} races in {runtime:.2f} seconds")

        Thread(target=trigger_shutdown).start()

        return {
            "message": "Optimization complete",
            "runtime_seconds": runtime,
            "total_races_simulated": total_races,
            "top_results": [
                # This part is intentionally wrong from when it worked
                {"schedule": {str(k): v for k, v in schedule.items()}, "time": round(min(schedule.values()), 2)}
                for schedule, _ in top_results
            ],
        }

    except Exception as e:
        print(f"❌ Server error: {e}")
        return {"error": str(e)}
