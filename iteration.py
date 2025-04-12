# iteration.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# -------------------------------------------------------------------------
# Helpers for W', CP, and power calculations
# -------------------------------------------------------------------------

def get_rider_info(rider_number, df_athletes):
    """
    Retrieve athlete power parameters from a DataFrame based on their rider number.
    
    Args:
        rider_number (int): e.g. 1, 2, 3, etc. for 'M1', 'M2', 'M3', ...
        df_athletes (DataFrame): The DataFrame containing athlete data, with columns:
            'Name', 'W'', 'Tmax', 'CP', 'CdA', 'Pmax'
    
    Returns:
        (W_prime, t_prime, CP, AC, Pmax) for the selected rider.
    """
    athlete_name = f"M{rider_number}"
    athlete_row = df_athletes[df_athletes['Name'] == athlete_name]

    W_prime = athlete_row["W'"].iloc[0] * 1000  # Convert kJ to J
    t_prime = athlete_row['Tmax'].iloc[0]
    CP      = athlete_row['CP'].iloc[0]
    AC      = athlete_row['CdA'].iloc[0]
    Pmax    = athlete_row['Pmax'].iloc[0]

    return W_prime, t_prime, CP, AC, Pmax


def get_power_duration_curve(power_duration_df, rider_id):
    """
    Create a function that returns the power for a given time (in seconds),
    based on a column in power_duration_df.

    Args:
        power_duration_df (DataFrame): Contains 'Time (s)' and per-rider columns.
        rider_id (str): e.g. 'M1', 'M2', etc. (column name in power_duration_df)
    
    Returns:
        interp1d object that takes time -> power (W).
    """
    times = power_duration_df["Time (s)"].values
    rider_power = power_duration_df[rider_id].values

    # Build a 1D linear interpolator. Will extrapolate if time is beyond dataset.
    return interp1d(times, rider_power, kind='linear', fill_value='extrapolate')


# -------------------------------------------------------------------------
# Numerical integration for acceleration phase
# -------------------------------------------------------------------------

def cumulative_trapezoid(y, x):
    """
    Custom version of cumulative trapezoid to avoid importing all of scipy.integrate
    for cumtrapz. Returns the cumulative integral of y wrt x using trapezoidal rule.
    """
    result = np.zeros_like(y)
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2
    result[1:] = np.cumsum(avg_y * dx)
    return result


def make_v_of_t_numerical(power, Mtot, rho, AC, v0):
    """
    Numerically solve dv/dt = (power - 0.5*rho*AC*v^3)/(Mtot*v) 
    with initial velocity v0, for t in [0, 100] seconds (or until solution is found).

    Returns:
        v_func(t): interpolation of velocity vs time
        l_func(t): interpolation of distance vs time
    """
    def dvdt(t, v):
        v = v[0]
        if v <= 0:
            return [0]
        drag = 0.5 * rho * AC * v**3
        return [(power - drag) / (Mtot * v)]

    t_span = (0, 100)
    t_eval = np.linspace(*t_span, 1000)

    sol = solve_ivp(dvdt, t_span, [v0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    t_vals = sol.t
    v_vals = sol.y[0]

    # Interpolate velocity
    v_func = interp1d(t_vals, v_vals, kind='cubic', fill_value='extrapolate')

    # Integrate to get distance, then interpolate distance
    l_vals = cumulative_trapezoid(v_vals, t_vals)
    l_func = interp1d(t_vals, l_vals, kind='cubic', fill_value='extrapolate')

    return v_func, l_func


def power_to_time(power, Mtot, rho, AC, num_laps_till_switch, v0):
    """
    Compute how long it takes to cover (125 * num_laps_till_switch) meters
    given a constant power and a numeric integration of dv/dt.
    """
    l_target = 125 * num_laps_till_switch
    v_func, l_func = make_v_of_t_numerical(power, Mtot, rho, AC, v0)

    def objective(t):
        return l_func(t) - l_target

    result = root_scalar(objective, bracket=[1, 100], method='brentq')
    return result.root if result.converged else np.nan


def find_self_consistent_power(
    initial_power_guess,
    power_duration_df,
    rider_col,
    Mtot,
    rho,
    AC,
    num_laps_till_switch,
    v0,
    epsilon=1.0,
    max_iter=50
):
    """
    Iteratively find a power value such that:
        a) The physics model says we take time T to do num_laps_till_switch half-laps at that power.
        b) The physiology model (power_duration_curve) says that at time T, the rider can only produce that power.

    Returns:
        (final_power, final_time)
    """
    # Build the power-duration interpolator for the lead rider
    power_curve = get_power_duration_curve(power_duration_df, rider_col)

    power_guess = initial_power_guess

    for i in range(max_iter):
        # Step 1: find time from physics
        t_result = power_to_time(power_guess, Mtot, rho, AC, num_laps_till_switch, v0)
        if np.isnan(t_result) or t_result <= 0:
            print(f"[WARNING] Iteration {i}: invalid time returned for power={power_guess:.2f}")
            return np.nan, np.nan

        # Step 2: get the “actual” power from the physiology curve
        power_actual = power_curve(t_result)

        # Step 3: check convergence
        diff = abs(power_actual - power_guess)
        print(f"Iteration {i}: guess={power_guess:.2f}, time={t_result:.2f}, actual={power_actual:.2f}, diff={diff:.2f}")
        if diff < epsilon:
            return power_actual, t_result

        power_guess = power_actual

    print("[WARNING] Maximum iterations reached without convergence.")
    return power_guess, t_result


# -------------------------------------------------------------------------
# Handling the race in phases
# -------------------------------------------------------------------------

def rotate_list(lst):
    """Rotate the first element to the end."""
    return lst[1:] + [lst[0]]


def accel_phase(
    v0,
    total_mass,
    switch_schedule,
    chosen_athletes,
    start_order,
    drafting_percents,
    power_duration_df,
    df_athletes,
    rho=1.225
):
    """
    Acceleration Phase: from the start until the first '1' in switch_schedule.
    
    Returns:
        (new_order, total_time, total_distance, half_lap_completed, W_rem)
    """
    # Find first switch in schedule
    try:
        next_switch_idx = switch_schedule.index(1)
    except ValueError:
        raise ValueError("No switches in schedule, can't determine end of acceleration phase.")

    # # half-laps from start until that switch
    num_laps_till_switch = next_switch_idx + 1

    # Build dictionary of athlete info
    rider_data = {}
    W_rem = {}
    for rider in chosen_athletes:
        Wp, t_prime, CP, AC, Pmax = get_rider_info(rider, df_athletes)
        rider_data[rider] = {
            "W_prime": Wp,
            "t_prime": t_prime,
            "CP": CP,
            "AC": AC,
            "Pmax": Pmax
        }
        # W' initially
        W_rem[f"W{rider}_rem"] = Wp

    # Solve for power/time for lead rider
    current_order = start_order.copy()
    lead_rider = current_order[0]
    AC_lead = rider_data[lead_rider]["AC"]
    rider_col = f"M{lead_rider}"

    initial_guess = 700.0  # or some rough estimate
    final_power, final_time = find_self_consistent_power(
        initial_power_guess=initial_guess,
        power_duration_df=power_duration_df,
        rider_col=rider_col,
        Mtot=total_mass,
        rho=rho,
        AC=AC_lead,
        num_laps_till_switch=num_laps_till_switch,
        v0=v0
    )

    # Update W' usage for each rider
    distance_covered = 125 * num_laps_till_switch
    for i, rider in enumerate(current_order):
        # draft factor
        draft_factor = drafting_percents[i]
        cp = rider_data[rider]["CP"]

        # Rider's actual power from physiology curve
        rider_power_curve = get_power_duration_curve(power_duration_df, f"M{rider}")
        # Multiply by drafting factor for their actual power output
        rider_power = rider_power_curve(final_time) * draft_factor

        # W' depletion
        delta_W = (rider_power - cp) * final_time
        W_rem[f"W{rider}_rem"] -= delta_W
        if W_rem[f"W{rider}_rem"] < 0:
            print(f"⚠️ Rider {rider} exceeded W′ during acceleration!")

    half_lap_completed = num_laps_till_switch

    return current_order, final_time, distance_covered, half_lap_completed, W_rem


def get_chunks_from_schedule(switch_schedule, half_lap_completed, peel_location=None, half_lap_dist=125):
    """
    Break the race into 'chunks' of half-laps between switches (marked as 1 in switch_schedule).
    This helps in 4-rider or 3-rider phases to handle segments with constant order.
    """
    chunks = []
    current_chunk = []
    chunk_start_idx = half_lap_completed - 1  # because half_lap_completed is how many we've finished

    rest_schedule = switch_schedule[chunk_start_idx:]
    for i, val in enumerate(rest_schedule):
        current_chunk.append(val)
        if val == 1:
            # We reached a switch
            chunk_half_laps = len(current_chunk)
            switch_idx = chunk_start_idx + chunk_half_laps

            # If there's a peel_location and we've gone past it, break
            if peel_location is not None and switch_idx > peel_location:
                break

            distance = 0
            # Summation of half-laps in this chunk:
            for j in range(chunk_start_idx, chunk_start_idx + chunk_half_laps):
                # If there's a switch at j, maybe you assume a slightly different distance?
                # This example code had 125 or 127. 
                # We'll keep it simple here with 125 each time unless you prefer otherwise.
                distance += half_lap_dist

            chunks.append({
                'start_idx': chunk_start_idx,
                'num_half_laps': chunk_half_laps,
                'distance': distance,
                'switch_idx': switch_idx
            })

            current_chunk = []
            chunk_start_idx += chunk_half_laps

    # If there's a trailing chunk after final switch
    if current_chunk:
        chunk_half_laps = len(current_chunk)
        distance = chunk_half_laps * half_lap_dist
        chunks.append({
            'start_idx': chunk_start_idx,
            'num_half_laps': chunk_half_laps,
            'distance': distance,
            'switch_idx': None
        })

    return chunks


def get_chunk_velocity(power_duration_df, lead_rider_id, AC, dist, rho=1.225):
    """
    Solve for velocity (m/s) needed so that aerodynamic power = rider power (from power-duration curve).
    The time is implicit in the root solve; once we get time, v = dist/time.

    Returns:
        velocity (float) or None if not solvable
    """
    power_curve = get_power_duration_curve(power_duration_df, lead_rider_id)

    def equation_to_solve(t):
        if t <= 0:
            return np.inf
        aerodynamic_power = 0.5 * rho * AC * (dist / t)**3
        return aerodynamic_power - power_curve(t)

    try:
        result = root_scalar(equation_to_solve, bracket=[1, 1000], method='brentq')
        if result.converged:
            tau = result.root
            return dist / tau
        else:
            raise ValueError("Root-finding did not converge.")
    except Exception as e:
        print(f"⚠️ Failed to solve chunk velocity: {e}")
        return None


def SS_4rider(
    power_duration_df,
    switch_schedule,
    peel_location,
    current_order,
    chosen_athletes,
    time,
    distance,
    W_rem,
    half_lap_completed,
    drafting_percents,
    df_athletes
):
    """
    4-Rider steady-state phase: from end of acceleration to the peel point.

    Returns:
        (order_after_4r, total_time, total_distance, half_lap_count, W_rem)
    """
    # Build data for all riders if needed
    rider_data = {}
    for rider in chosen_athletes:
        Wp, t_prime, CP, AC, Pmax = get_rider_info(rider, df_athletes)
        rider_data[rider] = {
            "W_prime": Wp, "t_prime": t_prime, "CP": CP, "AC": AC, "Pmax": Pmax
        }

    total_time = time
    total_distance = distance
    order = current_order
    count = half_lap_completed

    chunks = get_chunks_from_schedule(switch_schedule, count, peel_location)
    for chunk in chunks:
        dist_chunk = chunk['distance']
        switch_idx = chunk['switch_idx']
        lead_rider = order[0]

        # Solve velocity
        AC_lead = rider_data[lead_rider]['AC']
        velo = get_chunk_velocity(power_duration_df, f"M{lead_rider}", AC_lead, dist_chunk)
        if velo is None:
            print(f"Invalid velocity for chunk ending at switch {switch_idx}")
            return order, total_time, total_distance, count, W_rem

        chunk_time = dist_chunk / velo

        # Deplete W' for each of the 4 riders in this chunk
        for i in range(4):
            rider_id = order[i]
            cp = rider_data[rider_id]["CP"]
            power_curve = get_power_duration_curve(power_duration_df, f"M{rider_id}")
            actual_power = power_curve(chunk_time) * drafting_percents[i]

            W_rem[f"W{rider_id}_rem"] -= (actual_power - cp) * chunk_time
            if W_rem[f"W{rider_id}_rem"] < 0:
                print(f"⚠️ Rider {rider_id} exhausted W' in 4-rider phase.")

        # Update totals
        total_time += chunk_time
        total_distance += dist_chunk
        count += chunk['num_half_laps']

        # Rotate order after each chunk
        order = rotate_list(order)

    # Once we’re done with 4-rider phase, the lead rider peels off
    print(f"Peel location reached at switch {switch_idx}. Peeling off {order[0]}.")
    order = order[1:]  # drop the front rider

    return order, total_time, total_distance, count, W_rem


def SS_3rider(
    power_duration_df,
    switch_schedule,
    current_order,
    chosen_athletes,
    time,
    distance,
    W_rem,
    half_lap_completed,
    drafting_percents,
    df_athletes
):
    """
    3-Rider steady-state phase: from the peel point to the end of the schedule.

    Returns:
        (final_order, total_time, total_distance, final_half_lap_count, W_rem)
    """
    rider_data = {}
    for rider in chosen_athletes:
        Wp, t_prime, CP, AC, Pmax = get_rider_info(rider, df_athletes)
        rider_data[rider] = {
            "W_prime": Wp, "t_prime": t_prime, "CP": CP, "AC": AC, "Pmax": Pmax
        }

    total_time = time
    total_distance = distance
    order = current_order
    count = half_lap_completed

    chunks = get_chunks_from_schedule(switch_schedule, count, peel_location=None)
    for chunk in chunks:
        dist_chunk = chunk['distance']
        switch_idx = chunk['switch_idx']
        lead_rider = order[0]

        velo = get_chunk_velocity(power_duration_df, f"M{lead_rider}", rider_data[lead_rider]['AC'], dist_chunk)
        if velo is None:
            print(f"Invalid velocity for chunk ending at switch {switch_idx}")
            return order, total_time, total_distance, count, W_rem

        chunk_time = dist_chunk / velo

        # Deplete W' for 3 riders
        for i in range(3):
            rider_id = order[i]
            cp = rider_data[rider_id]["CP"]
            power_curve = get_power_duration_curve(power_duration_df, f"M{rider_id}")
            actual_power = power_curve(chunk_time) * drafting_percents[i]

            W_rem[f"W{rider_id}_rem"] -= (actual_power - cp) * chunk_time
            if W_rem[f"W{rider_id}_rem"] < 0:
                print(f"⚠️ Rider {rider_id} exhausted W' in 3-rider phase.")

        total_time += chunk_time
        total_distance += dist_chunk
        count += chunk['num_half_laps']

        # Rotate after chunk
        order = rotate_list(order)

    return order, total_time, total_distance, count, W_rem


def simulate_race(
    switch_schedule,
    chosen_athletes,
    start_order,
    drafting_percents,
    peel_location,
    power_duration_df,
    df_athletes,
    total_mass=70,
    v0=0.5,
    rho=1.225
):
    """
    Orchestrates a full race simulation:
      1) Acceleration phase (until first switch)
      2) 4-Rider steady-state phase (until peel_location)
      3) 3-Rider steady-state phase (the remainder)

    Args:
        switch_schedule (list of int): 32 half-laps with 1's marking switch points.
        chosen_athletes (list of int): e.g. [1,2,3,4]
        start_order (list of int): initial ordering of those riders.
        drafting_percents (list of float): e.g. [1.0, 0.58, 0.52, 0.53]
        peel_location (int): index in switch_schedule for where the peel occurs.
        power_duration_df (DataFrame): Data for times and each rider's possible power.
        df_athletes (DataFrame): Contains columns for 'Name', 'W'', 'Tmax', 'CP', 'CdA', 'Pmax', etc.
        total_mass (float): approximate total mass for acceleration phase.
        v0 (float): initial velocity in m/s.
        rho (float): air density.

    Returns:
        final_order, final_time, final_distance, final_half_lap_count, W_rem
    """

    print("\n=== Acceleration Phase ===")
    order, total_time, total_distance, half_lap_completed, W_rem = accel_phase(
        v0=v0,
        total_mass=total_mass,
        switch_schedule=switch_schedule,
        chosen_athletes=chosen_athletes,
        start_order=start_order,
        drafting_percents=drafting_percents,
        power_duration_df=power_duration_df,
        df_athletes=df_athletes,
        rho=rho
    )

    print("\n=== 4-Rider Steady-State Phase ===")
    order, total_time, total_distance, half_lap_completed, W_rem = SS_4rider(
        power_duration_df=power_duration_df,
        switch_schedule=switch_schedule,
        peel_location=peel_location,
        current_order=order,
        chosen_athletes=chosen_athletes,
        time=total_time,
        distance=total_distance,
        W_rem=W_rem,
        half_lap_completed=half_lap_completed,
        drafting_percents=drafting_percents,
        df_athletes=df_athletes
    )

    print("\n=== 3-Rider Steady-State Phase ===")
    # The next half-lap index is (peel_location + 1),
    # but in practice we keep counting from half_lap_completed.
    next_half_lap_completed = half_lap_completed + 1

    final_order, final_time, final_distance, final_half_lap_count, W_rem = SS_3rider(
        power_duration_df=power_duration_df,
        switch_schedule=switch_schedule,
        current_order=order,
        chosen_athletes=chosen_athletes,
        time=total_time,
        distance=total_distance,
        W_rem=W_rem,
        half_lap_completed=next_half_lap_completed,
        drafting_percents=drafting_percents,
        df_athletes=df_athletes
    )

    return final_order, final_time, final_distance, final_half_lap_count, W_rem
