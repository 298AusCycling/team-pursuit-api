from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import time
import sqlite3
import json
from datetime import datetime
from iteration import simulate_race

app = Flask(__name__)
CORS(app)

DATABASE = "simulations.db"
DRAFTING_PERCENTS = [1.0, 0.58, 0.52, 0.53]

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            chosen_athletes TEXT,
            start_order TEXT,
            switch_schedule TEXT,
            peel_location INTEGER,
            final_order TEXT,
            final_time REAL,
            final_distance REAL,
            final_half_lap_count INTEGER,
            W_rem TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_simulation_to_db(record):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO simulations (
            timestamp, chosen_athletes, start_order, switch_schedule,
            peel_location, final_order, final_time, final_distance,
            final_half_lap_count, W_rem
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.fromtimestamp(record["timestamp"]).isoformat(),
        json.dumps(record["chosen_athletes"]),
        json.dumps(record["start_order"]),
        json.dumps(record["switch_schedule"]),
        record["peel_location"],
        json.dumps(record["final_order"]),
        record["final_time"],
        record["final_distance"],
        record["final_half_lap_count"],
        json.dumps(record["W_rem"])
    ))
    conn.commit()
    conn.close()

@app.route("/simulate", methods=["POST"])
def run_simulation():
    data = request.json
    df_athletes = pd.read_json(data["df_athletes"])
    power_duration_df = pd.read_json(data["power_duration_df"])

    final_order, final_time, final_distance, final_half_lap_count, W_rem = simulate_race(
        switch_schedule=data["switch_schedule"],
        chosen_athletes=data["chosen_athletes"],
        start_order=data["start_order"],
        drafting_percents=DRAFTING_PERCENTS,
        peel_location=data["peel_location"],
        power_duration_df=power_duration_df,
        df_athletes=df_athletes
    )

    result = {
        "timestamp": time.time(),
        "chosen_athletes": data["chosen_athletes"],
        "start_order": data["start_order"],
        "switch_schedule": data["switch_schedule"],
        "peel_location": data["peel_location"],
        "final_order": final_order,
        "final_time": final_time,
        "final_distance": final_distance,
        "final_half_lap_count": final_half_lap_count,
        "W_rem": W_rem
    }

    save_simulation_to_db(result)
    return jsonify(result)

@app.route("/history", methods=["GET"])
def get_history():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM simulations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    simulations = []
    for row in rows:
        simulations.append({
            "id": row[0],
            "timestamp": row[1],
            "chosen_athletes": json.loads(row[2]),
            "start_order": json.loads(row[3]),
            "switch_schedule": json.loads(row[4]),
            "peel_location": row[5],
            "final_order": json.loads(row[6]),
            "final_time": row[7],
            "final_distance": row[8],
            "final_half_lap_count": row[9],
            "W_rem": json.loads(row[10])
        })

    return jsonify(simulations)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)
