#!/usr/bin/env python

"""
Script to calculate the coverage of the nursing charts with respect to the monitoring data.

Example: python optimal_coverage.py -i /tmp/coverage_2022_02_17.charted.csv -o /tmp/coverage.optimal.summary.2.csv -l /tmp/coverage.optimal.summary.pkl -p 10 -m t

nohup python -u calculate_coverage_multi_modality_optimal.py -i /tmp/coverage_2022_02_17.charted.csv -o /tmp/coverage.optimal.summary.csv -l /tmp/coverage.optimal.summary.pkl -m t > calculate_coverage_multi_modality_optimal.out &

python -u optimal_coverage.py -i /tmp/coverage_2022_02_17.charted.head.single_measure.csv -o /tmp/coverage.optimal.summary.csv -l /tmp/coverage.optimal.summary.pkl -m t

"""

import argparse
import csv
import datetime
from concurrent import futures

import numpy as np
import pandas as pd
import pytz
import pickle
from tqdm import tqdm
from calculate_coverage_common import *

pd.set_option('display.max_columns', None)

COLS = ["HR", "RR", "SpO2", "MAP"]
BOUNDS = {
    "HR": 5,
    "RR": 3,
    "SpO2": 2,
    "MAP": 6
}


# Note: coverage defined by three fields:
# (start index, end index) => (simulated nurse measure, num waveform elements during this range, num waveform elements covered)
# (0, 10) => [110, 4, 2] # if there are values in this range
# (0, 10) => [] # if there are no values in this range

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

def generate_coverage(col_to_data, min_time, max_time):
    start_end_to_coverage = {}
    for col_idx, col in enumerate(COLS):
        bound = BOUNDS[col]
        waveform_times = col_to_data[col]["waveform_times"]
        waveform_vals = col_to_data[col]["waveform_vital_signs"]
        curr_time = min_time
        # has_started = True
        while curr_time <= max_time:
            if curr_time < min(waveform_times):
                # This modality has not started recording yet, so there's no coverage to calculate
                # has_started = False

                start_min = int((curr_time - min_time).total_seconds() / 60.0)
                curr_time_inner = curr_time
                while curr_time_inner <= max_time:
                    end_min = int((curr_time_inner - min_time).total_seconds() / 60.0)
                    if (start_min, end_min) not in start_end_to_coverage:
                        start_end_to_coverage[(start_min, end_min)] = [[float("nan") for _ in range(len(COLS))], 0, 0]
                    curr_time_inner = curr_time_inner + datetime.timedelta(seconds=60)
                curr_time = curr_time + datetime.timedelta(seconds=60)
            else:
                # Get the last waveform val before this time
                simulated_nurse_measure = waveform_vals[waveform_times <= curr_time][-1]
                start_min = int((curr_time - min_time).total_seconds() / 60.0)
                curr_time_inner = curr_time
                while curr_time_inner <= max_time:
                    end_min = int((curr_time_inner - min_time).total_seconds() / 60.0)
                    if curr_time_inner == max_time:
                        local_vals = waveform_vals[(waveform_times <= curr_time_inner) & (waveform_times >= curr_time)]
                    else:
                        local_vals = waveform_vals[(waveform_times < curr_time_inner) & (waveform_times >= curr_time)]
                    total, covered = get_local_coverage(local_vals, simulated_nurse_measure, bound)

                    if (start_min, end_min) not in start_end_to_coverage:
                        start_end_to_coverage[(start_min, end_min)] = [[float("nan") for _ in range(len(COLS))], 0, 0]
                    start_end_to_coverage[(start_min, end_min)][0][col_idx] = simulated_nurse_measure
                    start_end_to_coverage[(start_min, end_min)][1] += total
                    start_end_to_coverage[(start_min, end_min)][2] += covered

                    curr_time_inner = curr_time_inner + datetime.timedelta(seconds=60)
                curr_time = curr_time + datetime.timedelta(seconds=60)
    return start_end_to_coverage


def find_optimal_coverage(orig_num_resources_left, col_to_data, min_time, max_time, silent=False):
    visited = {}

    def get_simulated_nurse_helper(start_end_to_coverage, min_time, start_time, simulated_nurse_vals,
                                   num_resources_left):

        start_min = int((start_time - min_time).total_seconds() / 60.0)
        if (start_min, num_resources_left) in visited:
            return visited[(start_min, num_resources_left)]

        if num_resources_left == 0:
            # No more nurse resources, just give the remaining waveform vals to the last known nurse val
            covered = 0
            total = 0
            for col_idx, col in enumerate(COLS):
                bound = BOUNDS[col]
                waveform_times = col_to_data[col]["waveform_times"]
                waveform_vals = col_to_data[col]["waveform_vital_signs"]
                waveform_vals = waveform_vals[waveform_times > start_time]
                for val in waveform_vals:
                    if simulated_nurse_vals[col_idx] != float("nan"):
                        if (simulated_nurse_vals[col_idx] - bound) <= val <= (simulated_nurse_vals[col_idx] + bound):
                            covered += 1
                    total += 1

                if (start_min, num_resources_left) not in visited:
                    visited[(start_min, num_resources_left)] = [[], [], 0, 0]
                visited[(start_min, num_resources_left)][2] += total
                visited[(start_min, num_resources_left)][3] += covered
            return [], [], visited[(start_min, num_resources_left)][2], visited[(start_min, num_resources_left)][3]

        best_coverage = 0
        best_nurse_times = []
        best_nurse_vals = []
        best_total = 0
        best_covered = 0
        best_coverage_params = None
        curr_min = int((start_time - min_time).total_seconds() / 60.0)
        curr_time = start_time
        while curr_time < max_time:
            #         if start_idx == 0:
            #             print(f"Working on ({start_idx}, {i})")
            if (start_min, curr_min) not in start_end_to_coverage:
                curr_time += datetime.timedelta(seconds=60)
                curr_min += 1

            coverage_params = start_end_to_coverage[(start_min, curr_min)]
            simulated_nurse_vals = coverage_params[0]
            total_in_window = coverage_params[1]
            covered_in_window = coverage_params[2]

            # Also consider how the performance is if we just take the current simulated_nurse_val
            # and use no more further nurse resources
            max_min = int((max_time - min_time).total_seconds() / 60.0)
            coverage_params_to_end = start_end_to_coverage[(start_min, max_min)]
            total_in_window_to_end = coverage_params_to_end[1]
            covered_in_window_to_end = coverage_params_to_end[2]

            nt, nv, total_ret, covered_ret = get_simulated_nurse_helper(start_end_to_coverage, min_time, curr_time,
                                                                        simulated_nurse_vals,
                                                                        num_resources_left - 1)

            nt_so_far = [start_time]
            nv_so_far = [simulated_nurse_vals]
            nt_so_far.extend(nt)
            nv_so_far.extend(nv)

            total = total_in_window + total_ret
            covered = covered_in_window + covered_ret
            curr_coverage = covered / total
            if curr_coverage >= best_coverage:
                best_coverage = curr_coverage
                best_nurse_times = nt_so_far
                best_nurse_vals = nv_so_far
                best_total = total
                best_covered = covered
                best_coverage_params = ((start_min, curr_min), coverage_params)

            # Could taking fewer number of resources also work?
            total_to_end = total_in_window_to_end
            covered_to_end = covered_in_window_to_end
            if total_to_end > 0:
                curr_coverage_to_end = covered_to_end / total_to_end
                # Note the equals because we want to prefer fewer resources if possible
                if curr_coverage_to_end >= best_coverage:
                    best_coverage = curr_coverage_to_end
                    best_nurse_times = [start_time]
                    best_nurse_vals = [simulated_nurse_vals]
                    best_total = total_in_window_to_end
                    best_covered = covered_in_window_to_end
                    best_coverage_params = ((start_min, curr_min), coverage_params)

            curr_time += datetime.timedelta(seconds=60)
            curr_min += 1

        visited[(start_min, num_resources_left)] = [best_nurse_times, best_nurse_vals, best_total, best_covered]
        # if num_resources_left == orig_num_resources_left:
        #     print(num_resources_left, best_coverage_params)
        #     print(best_nurse_times)
        return best_nurse_times, best_nurse_vals, best_total, best_covered

    if not silent:
        print(f"Building Coverage Lookup")
    start_end_to_coverage = generate_coverage(col_to_data, min_time, max_time)
    # for k in start_end_to_coverage.keys():
    #     if k[0] <= 1:
    #         print(k, start_end_to_coverage[k])
    if not silent:
        print(f"num_resources_left={orig_num_resources_left}")
        print(f"Simulating Nurse...")
    nurse_times, nurse_vital_signs, best_total, best_covered = get_simulated_nurse_helper(start_end_to_coverage,
                                                                                          min_time=min_time,
                                                                                          start_time=min_time,
                                                                                          simulated_nurse_vals=None,
                                                                num_resources_left=orig_num_resources_left)

    coverage = best_covered / best_total
    # if not silent:
    # print("===")
    # print(f"Coverage Fraction = {coverage}")
    # print(f"Covered = {best_covered}")
    # print(f"Total = {best_total}")
    # print(f"len(nurse_times) = {len(nurse_times)}")
    # print(f"nurse_times = {len(nurse_times)}")
    # print(f"nurse_vital_signs = {len(nurse_vital_signs)}")
    # print("===")

    if silent:
        return best_covered, best_total, nurse_times, nurse_vital_signs


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


def get_coverage(input):
    i, total_rows, df, csn, interpolate = input
    print(f"Working on {i}/{total_rows} for CSN={csn}...")

    row = [csn]

    try:
        b = get_raw_data(csn)
        col_to_data = {}
        orig_covereds = []
        orig_totals = []
        orig_coverages = []
        min_time = None
        max_time = None
        num_resources = None

        for k in range(len(COLS)):
            col = COLS[k]
            bound = BOUNDS[col]
            nurse_times, nurse_vital_signs, waveform_times, waveform_vital_signs = get_data(b, df, csn, col=col,
                                                                                            avg_over=60,
                                                                                            interpolate=interpolate,
                                                                                            manual_data_source="nurse",
                                                                                            manual_sampling_avg_over=None)
            seen = set()
            to_remove = []
            for i in range(len(nurse_times)):
                if nurse_times[i] in seen:
                    to_remove.append(i)
                seen.add(nurse_times[i])

            nurse_times = np.array(nurse_times)
            nurse_times = np.delete(nurse_times, to_remove)
            num_resources = len(nurse_times)

            col_to_data[col] = {
                "waveform_times": waveform_times,
                "waveform_vital_signs": waveform_vital_signs
            }

            orig_covered, orig_total, _, _ = coverage_fraction_step(waveform_times, waveform_vital_signs, nurse_times,
                                                         nurse_vital_signs, bound=bound)
            orig_coverage = orig_covered / orig_total
            orig_coverages.append(orig_coverage)
            orig_covereds.append(orig_covered)
            orig_totals.append(orig_total)
            if min_time is None:
                min_time = min(waveform_times)
            min_time = min(min_time, min(waveform_times))
            if max_time is None:
                max_time = max(waveform_times)
            max_time = max(max_time, max(waveform_times))

        best_covered, best_total, best_nurse_times, best_nurse_vital_signs = find_optimal_coverage(num_resources,
                                                                                                   col_to_data,
                                                                                                   min_time=min_time,
                                                                                                   max_time=max_time,
                                                                                                   silent=True)
        print(best_nurse_times)
        print(best_nurse_vital_signs)
        simulated_covered = []
        simulated_total = []
        for col_idx, col in enumerate(COLS):
            bound = BOUNDS[col]
            waveform_times = col_to_data[col]["waveform_times"]
            waveform_vals = col_to_data[col]["waveform_vital_signs"]
            simulated_vals = np.array(best_nurse_vital_signs)[:, col_idx]
            covered_count, total_count, _, _ = coverage_fraction_step(waveform_times, waveform_vals, best_nurse_times,
                                                                      simulated_vals, bound=bound)
            simulated_covered.append(covered_count)
            simulated_total.append(total_count)

        row.extend([num_resources, np.sum(orig_covereds), np.sum(orig_totals), np.sum(orig_covereds) / np.sum(orig_totals), len(best_nurse_times), best_covered, best_total, best_covered / best_total])
        for col_idx, col in enumerate(COLS):
            row.append(simulated_covered[col_idx])
            row.append(simulated_total[col_idx])
        return row, best_nurse_times, best_nurse_vital_signs
    except Exception as e:
        print(e)
        return None, None, None
        # raise e


def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    input_file = args.input_file
    output_file = args.output_file
    output_list_file = args.output_list_file
    interpolate = args.interpolate_missing == "t"
    max_patients = int(args.max_patients) if args.max_patients is not None else None

    df = pd.read_csv(input_file)
    print(f"Found input_file={input_file} with shape {df.shape} and interpolate={interpolate}")

    csns = sorted(list(set(df["CSN"].tolist())))
    total_rows = len(csns)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, csn in tqdm(enumerate(csns), disable=True):
            input_args = [i, total_rows, df, csn, interpolate]
            future = executor.submit(get_coverage, input_args)
            fs.append(future)
            if max_patients is not None and i >= (max_patients - 1):
                break

    output_times = []
    with open(f"{output_file}", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        header = ["CSN"]
        header.extend([f"orig_len", f"orig_covered", f"orig_total", f"orig_coverage", f"best_len", f"best_covered", f"best_total", f"best_coverage"])
        for col_idx, col in enumerate(COLS):
            header.append(f"{col}_covered")
            header.append(f"{col}_total")
        writer.writerow(header)
        for future in futures.as_completed(fs):
            # Blocking call - wait for 1 hour for a single future to complete
            # (highly unlikely, most likely something is wrong)
            row, times_list, _ = future.result(timeout=60 * 60)
            if row is not None:
                output_times.append(times_list)
                writer.writerow(row)

    with open(f"{output_list_file}", 'wb') as f:
        output_obj = {
            "output_times": output_times
        }
        pickle.dump(output_obj, f)

    print(f"Output is written to: {output_file}")
    print(f"END TIME: {datetime.datetime.now()}")


#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the coverage')
    parser.add_argument('-i', '--input-file',
                        required=True,
                        help='The path to the input/states file')
    parser.add_argument('-o', '--output-file',
                        required=True,
                        help='The output file location')
    parser.add_argument('-l', '--output-list-file',
                        required=True,
                        help='The output file list location')
    parser.add_argument('-m', '--interpolate-missing',
                        default="f",
                        help='Interpolates any missing points')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')

    args = parser.parse_args()

    run(args)

    print("DONE")
