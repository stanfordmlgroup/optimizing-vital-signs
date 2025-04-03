import datetime

import numpy as np
import pandas as pd
import pytz

def get_local_coverage(waveform_vals, last_nurse_val, bound):
    covered = 0
    total = 0
    i = 0
    while i < len(waveform_vals):
        if (last_nurse_val - bound) <= waveform_vals[i] <= (last_nurse_val + bound):
            covered += 1
        total += 1
        i += 1
    return total, covered


def interpolate_arr(new_times, new_vals):
    new_times_inter = []
    new_vals_inter = []
    last_val = None
    last_time = None
    for t_idx in range(len(new_times)):
        t = new_times[t_idx]
        v = new_vals[t_idx]

        if last_time is not None:
            # Interpolate any missing values minute-by-minute
            # e.g. if last_time was 10:00, t is 10:03, we need
            # to add the value at 10:00 for 10:01 and 10:02
            while last_time < (t - datetime.timedelta(minutes=1)):
                last_time = last_time + datetime.timedelta(minutes=1)
                new_times_inter.append(last_time)
                new_vals_inter.append(last_val)
        new_times_inter.append(t)
        new_vals_inter.append(v)
        last_time = t
        last_val = v
    return new_times_inter, new_vals_inter


def avg_arr(b, col, avg_over=60, interpolate=False):
    if col == "MAP":
        s_times = b[f"NBPs-time"]
        d_times = b[f"NBPd-time"]
        s_vals = np.array(b[f"NBPs"])
        d_vals = np.array(b[f"NBPd"])

        times = []
        vals = []
        i = 0
        j = 0
        while i < len(s_times) and j < len(d_times):
            if s_times[i] == d_times[j]:
                vals.append((1 / 3) * (s_vals[i] + d_vals[j] * 2))
                times.append(s_times[i])
                i += 1
                j += 1
            elif s_times[i] < d_times[j]:
                i += 1
            else:
                j += 1

        # avg_over = None
    else:
        times = b[f"{col}-time"]
        vals = b[col]

    # if avg_over is None:
    #     if interpolate:
    #         times, vals = interpolate_arr(times, vals)
    #     print(f"col={col} len {len(times)}")
    #     return np.array(times), np.array(vals)

    # Go through window by window
    new_times = []
    new_vals = []
    window_times = []
    window_vals = []
    for i in range(len(times)):
        if avg_over is None:
            new_times.append(times[i])
            new_vals.append(vals[i])
        elif len(window_vals) == 0:
            window_times.append(times[i])
            window_vals.append(vals[i])
        elif (times[i] - window_times[0]).total_seconds() >= avg_over:
            new_times.append(window_times[0])
            new_vals.append(np.mean(window_vals))
            window_times = []
            window_vals = []
        window_times.append(times[i])
        window_vals.append(vals[i])

    # Add any straggling remaining values
    if len(window_times) > 0:
        new_times.append(window_times[0])
        new_vals.append(np.mean(window_vals))

    # Interpolate if necessary
    if interpolate:
        new_times, new_vals = interpolate_arr(new_times, new_vals)

    return np.array(new_times), np.array(new_vals)


def coverage_fraction_step(waveform_times, waveform_vals, nurse_times, nurse_vals, bound):
    if len(waveform_times) <= 0 or len(nurse_times) <= 0:
        return None, None, None, None

    nurse_times = np.array(nurse_times)
    nurse_vals = np.array(nurse_vals)
    covered_arr = []  # bit array of 1 = covered, 0 = not covered
    y_pred = []  # the latest nurse chart repeated

    covered_count = 0
    total_count = 0

    # Skip any waveform times that come before the first nurse time
    j = 0
    while j < len(waveform_times) and nurse_times[0] > waveform_times[j]:
        covered_arr.append(float("nan"))
        y_pred.append(float("nan"))
        total_count += 1
        j += 1

    while j < len(waveform_times):
        # Get most recent nurse measure
        curr_nurse_vals = nurse_vals[nurse_times <= waveform_times[j]]
        if len(curr_nurse_vals) == 0:
            # Waveform time is before earliest nurse time so we punish algorithm
            covered_arr.append(0)
            y_pred.append(float("nan"))
        else:
            next_nurse_val = curr_nurse_vals[-1]
            if next_nurse_val - bound <= waveform_vals[j] <= next_nurse_val + bound:
                covered_arr.append(1)
                covered_count += 1
            else:
                covered_arr.append(0)
            y_pred.append(next_nurse_val)
        total_count += 1
        j += 1
    return covered_count, total_count, np.array(covered_arr), np.array(y_pred)


def get_raw_data(csn, root_folder, cols=["MAP", "RR", "HR", "SpO2"]):
    hash_folder = str(csn)[-2:]

    cols = cols[:]
    if "MAP" in cols:
        cols.remove("MAP")
        cols.append("NBPs")
        cols.append("NBPd")

    b = {}
    for col in cols:
        df_col = pd.read_csv(f"{root_folder}/{hash_folder}/{csn}/{col}.csv")
        times = df_col["recorded_time"].tolist()
        vals = df_col[col].tolist()

        new_times = []
        new_vals = vals
        for t in times:
            if col == "NBPs" or col == "NBPd":
                try:
                    t = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S%z")
                except:
                    t = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f%z")
            else:
                try:
                    t = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f%z")
                except:
                    t = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S%z")
            t = t.replace(microsecond=0)
            t = t.replace(second=0)
            new_times.append(t)

        b[f"{col}-time"] = np.array(new_times)
        b[f"{col}"] = np.array(new_vals)
        # print(f"{col}: {len(new_vals)}")

    return b


def get_data(b, df_states, csn, col="HR", avg_over=None, interpolate=False, manual_data_source="nurse",
             manual_sampling_avg_over=None):
    pt_df = df_states[(df_states["CSN"] == csn) & (df_states["Variable"] == col)]

    nurse_times = []
    for x in pt_df["Time"].tolist():
        t = datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        t = pytz.timezone('America/Vancouver').localize(t)
        nurse_times.append(t)
    nurse_vital_signs = pt_df["Charted"].tolist()

    waveform_times, waveform_vital_signs = avg_arr(b, col, avg_over=avg_over, interpolate=interpolate)

    if manual_data_source != "nurse":
        nurse_times, nurse_vital_signs = avg_arr(b, col, avg_over=manual_sampling_avg_over)

    return nurse_times, nurse_vital_signs, waveform_times, waveform_vital_signs

