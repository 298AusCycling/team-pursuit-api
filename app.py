# app.py

import streamlit as st
import pandas as pd
import time
import sqlite3
import json
from datetime import datetime
from iteration import simulate_race
import io

st.title("Team Pursuit Race Simulator")

# Connect to SQLite database
conn = sqlite3.connect("simulations.db", check_same_thread=False)
cursor = conn.cursor()

# Create the simulations table if it doesn't exist
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

drafting_percents = [1.0, 0.58, 0.52, 0.53]

def switch_schedule_description(switch_schedule):
    return [i+1 for i, v in enumerate(switch_schedule) if v == 1]

def save_simulation_to_db(record):
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

uploaded_file = st.file_uploader("Upload Performance Data Excel File", type=["xlsx"])

if uploaded_file:
    df_athletes = pd.read_excel(uploaded_file)
    power_duration_df = pd.read_excel(uploaded_file, sheet_name="Power Curves")

    available_athletes = (
        df_athletes["Name"]
        .str.extract(r'M(\d+)')[0]
        .dropna()
        .astype(int)
        .tolist()
    )

    chosen_athletes = st.multiselect("Select 4 Athletes", available_athletes)
    st.markdown(f"Selected Riders: {sorted(chosen_athletes)}.")

    if len(chosen_athletes) == 4:
        start_order = st.multiselect("Initial Rider Order", sorted(chosen_athletes))
        st.markdown(f"Initial Starting Order: {start_order}")

        st.subheader("Switch Schedule (32 half-laps)")
        switch_schedule = []
        peel_schedule = []

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Switch (1 = switch after this half-lap)**")
            for i in range(31):
                val = st.checkbox(f"{i+1}", key=f"switch_{i}")
                switch_schedule.append(1 if val else 0)

        with col2:
            st.markdown("**Peel (1 = peel here)**")
            for i in range(31):
                val = st.checkbox(f"{i+1}", key=f"peel_{i}")
                peel_schedule.append(1 if val else 0)

        try:
            peel_location = peel_schedule.index(1)
        except ValueError:
            peel_location = None

        if peel_location is None:
            st.warning("Please select at least one peel location.")
        else:
            if st.button("Simulate Race"):
                with st.spinner("Running simulation..."):
                    final_order, final_time, final_distance, final_half_lap_count, W_rem = simulate_race(
                        switch_schedule=switch_schedule,
                        chosen_athletes=chosen_athletes,
                        start_order=start_order,
                        drafting_percents=drafting_percents,
                        peel_location=peel_location + 1,
                        power_duration_df=power_duration_df,
                        df_athletes=df_athletes,
                        total_mass=70,
                        v0=0.5,
                        rho=1.225
                    )

                st.success("✅ Simulation Complete!")
                st.write(f"**Final Order:** {final_order}")
                st.write(f"**Total Time:** {final_time:.2f} seconds")
                st.write(f"**Total Distance:** {final_distance:.2f} m")
                st.write(f"**Half Laps Completed:** {final_half_lap_count}")
                st.write(f"**Switch at half-laps:** {switch_schedule_description(switch_schedule)}")

                st.subheader("W′ Remaining per Rider:")
                for k, v in W_rem.items():
                    st.write(f"{k}: {v:.1f} J")

                simulation_record = {
                    "timestamp": time.time(),
                    "chosen_athletes": chosen_athletes,
                    "start_order": start_order,
                    "switch_schedule": switch_schedule,
                    "peel_location": peel_location,
                    "final_order": final_order,
                    "final_time": final_time,
                    "final_distance": final_distance,
                    "final_half_lap_count": final_half_lap_count,
                    "W_rem": W_rem
                }
                save_simulation_to_db(simulation_record)
                st.info("Simulation saved to database!")

st.header("Past Simulations (Saved)")

cursor.execute("SELECT * FROM simulations ORDER BY id DESC")
rows = cursor.fetchall()

if not rows:
    st.write("No simulations saved yet.")
else:
    for row in rows:
        sim_id, timestamp, chosen_athletes, start_order, switch_schedule, peel_location, final_order, final_time, final_distance, final_half_lap_count, W_rem = row

        with st.expander(f"Simulation #{sim_id} ({timestamp})"):
            st.write(f"**Riders:** {json.loads(chosen_athletes)}")
            st.write(f"**Start Order:** {json.loads(start_order)}")
            st.write(f"**Switch Schedule:** {json.loads(switch_schedule)}")
            st.write(f"**Peel Location:** {peel_location}")
            st.write(f"**Final Order:** {json.loads(final_order)}")
            st.write(f"**Final Time:** {final_time:.2f} s")
            st.write(f"**Final Distance:** {final_distance:.2f} m")
            st.write(f"**Final Half Laps:** {final_half_lap_count}")
            st.write("**W′ Remaining:**")
            for k, v in json.loads(W_rem).items():
                st.write(f"{k}: {v:.1f} J")

            if st.button(f"🗑️ Delete Simulation #{sim_id}", key=f"delete_{sim_id}"):
                cursor.execute("DELETE FROM simulations WHERE id = ?", (sim_id,))
                conn.commit()
                st.success(f"Simulation #{sim_id} deleted.")
                st.experimental_rerun()


# Add download button at bottom
st.subheader("Download Past Simulations")

# Fetch all simulation records from DB
cursor.execute("SELECT * FROM simulations ORDER BY id DESC")
all_rows = cursor.fetchall()

if all_rows:
    # Convert to pandas DataFrame
    df_download = pd.DataFrame([{
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
    } for row in all_rows])

    # Convert to CSV in-memory
    csv_buffer = io.StringIO()
    df_download.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    # Create download button
    st.download_button(
        label="Download Simulations as CSV",
        data=csv_bytes,
        file_name="simulations.csv",
        mime="text/csv"
    )
else:
    st.info("No simulations available to download.")
