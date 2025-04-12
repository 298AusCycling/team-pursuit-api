import streamlit as st
import pandas as pd
import time
import sqlite3
import json
import requests
from datetime import datetime

st.title("Team Pursuit Race Simulator")

DRAFTING_PERCENTS = [1.0, 0.58, 0.52, 0.53]

def switch_schedule_description(switch_schedule):
    return [i+1 for i, v in enumerate(switch_schedule) if v == 1]

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

    chosen_athletes = st.multiselect("Select 4 Athletes", options=available_athletes)
    if len(chosen_athletes) == 4:
        start_order = st.multiselect("Initial Rider Order", options=sorted(chosen_athletes))

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
        elif st.button("Simulate Race"):
            st.spinner("Running simulation...")

            payload = {
                "switch_schedule": switch_schedule,
                "chosen_athletes": chosen_athletes,
                "start_order": start_order,
                "peel_location": peel_location + 1,
                "df_athletes": df_athletes.to_json(),
                "power_duration_df": power_duration_df.to_json()
            }

            response = requests.post("http://localhost:5000/simulate", json=payload)

            if response.status_code == 200:
                data = response.json()

                st.success("✅ Simulation Complete!")
                st.write(f"**Final Order:** {data['final_order']}")
                st.write(f"**Total Time:** {data['final_time']:.2f} seconds")
                st.write(f"**Total Distance:** {data['final_distance']:.2f} m")
                st.write(f"**Half Laps Completed:** {data['final_half_lap_count']}")
                st.write(f"**Switch at half-laps:** {switch_schedule_description(switch_schedule)}")

                st.subheader("W′ Remaining per Rider:")
                for k, v in data["W_rem"].items():
                    st.write(f"{k}: {v:.1f} J")
            else:
                st.error("❌ Simulation failed.")

st.header("Past Simulations")
history = requests.get("http://localhost:5000/history").json()

if not history:
    st.write("No simulations saved yet.")
else:
    for sim in history:
        with st.expander(f"Simulation #{sim['id']} ({sim['timestamp']})"):
            st.write(f"**Riders:** {sim['chosen_athletes']}")
            st.write(f"**Start Order:** {sim['start_order']}")
            st.write(f"**Switch Schedule:** {sim['switch_schedule']}")
            st.write(f"**Peel Location:** {sim['peel_location']}")
            st.write(f"**Final Order:** {sim['final_order']}")
            st.write(f"**Final Time:** {sim['final_time']:.2f} s")
            st.write(f"**Final Distance:** {sim['final_distance']:.2f} m")
            st.write(f"**Final Half Laps:** {sim['final_half_lap_count']}")
            st.write("**W′ Remaining:**")
            for k, v in sim['W_rem'].items():
                st.write(f"{k}: {v:.1f} J")
