from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from iteration import simulate_race

import pandas as pd
import json

app = FastAPI()

class SimulateRequest(BaseModel):
    switch_schedule: List[int]
    chosen_athletes: List[int]
    start_order: List[int]
    drafting_percents: List[float]
    peel_location: int
    df_athletes: List[Dict[str, Any]]  # list of dicts (DataFrame)
    power_duration_df: List[Dict[str, Any]]  # same

@app.post("/simulate")
def run_simulation(data: SimulateRequest):
    df_athletes = pd.DataFrame(data.df_athletes)
    power_duration_df = pd.DataFrame(data.power_duration_df)

    final_order, final_time, final_distance, final_half_lap_count, W_rem = simulate_race(
        switch_schedule=data.switch_schedule,
        chosen_athletes=data.chosen_athletes,
        start_order=data.start_order,
        drafting_percents=data.drafting_percents,
        peel_location=data.peel_location,
        power_duration_df=power_duration_df,
        df_athletes=df_athletes,
        total_mass=70,
        v0=0.5,
        rho=1.225
    )

    return {
        "final_order": final_order,
        "final_time": final_time,
        "final_distance": final_distance,
        "final_half_lap_count": final_half_lap_count,
        "W_rem": W_rem
    }
