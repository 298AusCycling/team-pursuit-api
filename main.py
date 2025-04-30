from fastapi import FastAPI
from datetime import datetime
from optimization import genetic_algorithm
import itertools

app = FastAPI()

@app.post("/run_optimization")
def run_optimization():
    start_time = datetime.now()
    print(f"🚀 Optimization triggered at {start_time}")

    list_of_initial_orders = list(itertools.permutations([0, 1, 2, 3]))
    num_changes_list = [3, 5]

    all_results = []
    total_races = 0

    for acceleration_length in [3, 4]:
        for peel in range(10, 33):  # 10 to 32 inclusive
            for initial_order in list_of_initial_orders:
                for num_changes in num_changes_list:
                    # Run the optimizer
                    try:
                        _, best_time, best_schedule = genetic_algorithm(
                            acceleration_length=acceleration_length,
                            peel_index=peel,
                            num_changes=num_changes,
                            initial_order=list(initial_order),
                            num_children=10,
                            num_seeds=4,
                            num_rounds=5,
                        )
                        total_races += 10 * 5  # children × rounds
                        all_results.append((best_schedule, best_time))
                    except Exception as e:
                        print(f"⚠️ Optimization failed for order {initial_order}, peel {peel}, error: {e}")

    # Sort results and select top 5
    top_results = sorted(all_results, key=lambda x: x[1])[:5]
    runtime = (datetime.now() - start_time).total_seconds()

    print(f"✅ Optimization completed with {total_races} races in {runtime:.2f} seconds")

    return {
        "message": "Optimization complete",
        "runtime_seconds": runtime,
        "total_races_simulated": total_races,
        "top_results": [(schedule, round(time, 2)) for schedule, time in top_results],
    }
