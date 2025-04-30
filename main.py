from fastapi import FastAPI
from optimization import genetic_algorithm
import itertools
import time

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Optimization API is running."}

@app.post("/run_optimization")
def run_full_optimization():
    all_orders = list(itertools.permutations([1, 2, 3, 4]))
    list_of_initial_orders = [list(p) for p in all_orders]

    huge_dict = {}
    start = time.time()
    
    for acceleration_length in range(3, 5):
        for peel in range(10, 33):
            for initial_order in list_of_initial_orders: 
                for num_changes in (7, 16):
                    output = genetic_algorithm(
                        peel=peel,
                        initial_order=initial_order,
                        acceleration_length=acceleration_length,
                        num_changes=num_changes,
                        num_children=10,
                        num_seeds=4,
                        num_rounds=5
                    )[2]  # result dictionary
                    huge_dict.update(output)

    sorted_final_dict = dict(sorted(huge_dict.items(), key=lambda item: item[1]))
    top_5 = list(sorted_final_dict.items())[:5]
    end = time.time()

    return {
        "status": "completed",
        "top_results": top_5,
        "total_races_simulated": len(huge_dict),
        "runtime_seconds": round(end - start, 2)
    }
