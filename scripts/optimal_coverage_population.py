#!/usr/bin/env python

"""
Script to calculate the coverage of the nursing charts with respect to the monitoring data using population methods

Example:

python -u /deep/u/tomjin/optimizing-vital-signs/scripts/optimal_coverage_population.py -i /deep/group/physiologic-states/v3/coverage_2022_02_17.charted.csv -o /deep/group/physiologic-states/v3/optimizing_coverage_population.optimal.csv -l /deep/group/physiologic-states/v3/optimizing_coverage_population.optimal.pkl -m t -r /deep/group/physiologic-states/v3/csv -c HR,RR,SpO2,MAP -p 10

"""

import argparse
import csv
import datetime
import traceback
from concurrent import futures

import operator
import numpy as np
import pandas as pd
import pytz
import pickle
from tqdm import tqdm
from calculate_coverage_common import *

pd.set_option('display.max_columns', None)

BOUNDS = {
    "HR": 5,
    "RR": 3,
    "SpO2": 2,
    "MAP": 6
}

TRUNCATION_TOP_K = 10
MUTATION_RATE = 0.05
MUTATION_MAX_SHIFT_MINUTES = 5


# Note: coverage defined by three fields:
# (start index, end index) => (simulated nurse measure, num waveform elements during this range, num waveform elements covered)
# (0, 10) => [110, 4, 2] # if there are values in this range
# (0, 10) => [] # if there are no values in this range


def generate_coverage(cols, col_to_data, min_time, max_time):
    start_end_to_coverage = {}
    for col_idx, col in enumerate(cols):
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
                        start_end_to_coverage[(start_min, end_min)] = [[float("nan") for _ in range(len(cols))], 0, 0]
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
                        start_end_to_coverage[(start_min, end_min)] = [[float("nan") for _ in range(len(cols))], 0, 0]
                    start_end_to_coverage[(start_min, end_min)][0][col_idx] = simulated_nurse_measure
                    start_end_to_coverage[(start_min, end_min)][1] += total
                    start_end_to_coverage[(start_min, end_min)][2] += covered

                    curr_time_inner = curr_time_inner + datetime.timedelta(seconds=60)
                curr_time = curr_time + datetime.timedelta(seconds=60)
    return start_end_to_coverage


def generate_individual(max_index, max_charted_events):
    indices = list(range(max_index))
    return np.random.choice(indices, size=max_charted_events, replace=False)


def initialize_population(max_individuals, col_to_data, cols, max_charted_events):
    max_index = 0
    for c in cols:
        waveform_times = col_to_data[cols[0]]["waveform_times"]
        if len(waveform_times) > max_index:
            max_index = len(waveform_times)

    population = []
    for i in range(max_individuals):
        population.append(generate_individual(max_index, max_charted_events))
    return population


def select(population, waveform_times, ):
    return


def crossover(parents, population_size):
    children = []
    for i in range(population_size):
        # Pick two parents at random
        p1 = parents[np.random.choice(range(len(parents)))]
        p2 = parents[np.random.choice(range(len(parents)))]
        crossover_point = np.random.randint(0, len(p1))
        child = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
        children.append(child)
    return children


def mutate(children):
    mutated_children = []
    for child in children:
        if np.random.rand() >= MUTATION_RATE:
            # Don't mutate in this case
            mutated_children.append(child)
        else:
            random_index = np.random.choice(list(range(len(child))))
            random_shift_min = np.random.randint(-MUTATION_MAX_SHIFT_MINUTES, MUTATION_MAX_SHIFT_MINUTES + 1)
            child[random_index] = max(child[random_index] + random_shift_min, 0)
            # Sort the list in case the mutation caused ordering to change
            child = sorted(child)
            mutated_children.append(child)
    return mutated_children


def get_population_coverages(population, col_to_data, cols):
    coverages = []
    for individual_indices in population:
        covered = 0
        total = 0
        for col_idx, col in enumerate(cols):
            bound = BOUNDS[col]
            waveform_times = col_to_data[col]["waveform_times"]
            waveform_vals = col_to_data[col]["waveform_vital_signs"]

            individual_indices = individual_indices[individual_indices < len(waveform_times)]
            simulated_times = waveform_times[individual_indices]
            simulated_vals = waveform_vals[individual_indices]
            covered_count, total_count, _, _ = coverage_fraction_step(waveform_times, waveform_vals, simulated_times,
                                                                      simulated_vals, bound=bound)
            if covered_count is not None and total_count is not None:
                covered += covered_count
                total += total_count

        if total == 0:
            coverages.append(0)
        else:
            coverages.append(covered / total)

    return coverages


def get_coverage(input):
    i, total_rows, df, csn, interpolate, root_folder, cols, population_size, max_iterations = input
    print(f"Working on {i}/{total_rows} for CSN={csn}...")
    fn_start = datetime.datetime.now()

    row = [csn]

    np.random.seed(i)

    try:
        b = get_raw_data(csn, root_folder, cols)
        col_to_data = {}
        orig_covereds = []
        orig_totals = []
        orig_coverages = []
        min_time = None
        max_time = None
        num_resources = None

        for k in range(len(cols)):
            col = cols[k]
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

        population = initialize_population(population_size, col_to_data, cols, num_resources)

        iter = 0
        while iter < max_iterations:
            # Calculate coverage for the current population
            #
            coverages = get_population_coverages(population, col_to_data, cols)

            # Select for the fittest individuals (with highest coverage)
            fittest_indices = np.argpartition(coverages, -TRUNCATION_TOP_K)[-TRUNCATION_TOP_K:]
            # print(f"Coverage in iter {iter} is {np.max(coverages)}")
            parents = operator.itemgetter(*fittest_indices)(population)
            children = crossover(parents, population_size)
            children = mutate(children)
            population = np.array(children)

            iter += 1

        coverages = get_population_coverages(population, col_to_data, cols)
        best_individual_index = np.argmax(coverages)
        best_coverage = coverages[best_individual_index]
        best_individual = population[best_individual_index]
        best_times = col_to_data[cols[0]]["waveform_times"][best_individual]

        simulated_covered = []
        simulated_total = []
        for col_idx, col in enumerate(cols):
            bound = BOUNDS[col]
            waveform_times = col_to_data[col]["waveform_times"]
            waveform_vals = col_to_data[col]["waveform_vital_signs"]
            simulated_times = waveform_times[best_individual[best_individual < len(waveform_times)]]
            simulated_vals = waveform_vals[best_individual[best_individual < len(waveform_times)]]
            covered_count, total_count, _, _ = coverage_fraction_step(waveform_times, waveform_vals, simulated_times,
                                                                      simulated_vals, bound=bound)
            if covered_count is not None and total_count is not None:
                simulated_covered.append(covered_count)
                simulated_total.append(total_count)
            else:
                simulated_covered.append(0)
                simulated_total.append(0)

        row.extend(
            [num_resources, np.sum(orig_covereds), np.sum(orig_totals), np.sum(orig_covereds) / np.sum(orig_totals),
             len(best_individual), sum(simulated_covered), sum(simulated_total), sum(simulated_covered) / sum(simulated_total) if sum(simulated_total) > 0 else 0])
        for col_idx, col in enumerate(cols):
            row.append(simulated_covered[col_idx])
            row.append(simulated_total[col_idx])

        fn_end = datetime.datetime.now()
        row.append((fn_end - fn_start).total_seconds())
        return row, best_times
    except Exception as e:
        print(f"CSN {csn} had an error")
        print(e)
        print(traceback.format_exc())
        return None, None
#         raise e


def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    input_file = args.input_file
    output_file = args.output_file
    output_list_file = args.output_list_file
    root_folder = args.patient_folder
    interpolate = args.interpolate_missing == "t"
    cols = args.columns.split(",")
    max_iterations = int(args.max_iterations)
    population_size = int(args.population_size)
    max_patients = int(args.max_patients) if args.max_patients is not None else None

    df = pd.read_csv(input_file)
    print(f"Found input_file={input_file} with shape {df.shape} and interpolate={interpolate}")

    csns = sorted(list(set(df["CSN"].tolist())))
    total_rows = len(csns)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, csn in tqdm(enumerate(csns), disable=True):
            input_args = [i, total_rows, df, csn, interpolate, root_folder, cols, population_size, max_iterations]
            future = executor.submit(get_coverage, input_args)
            fs.append(future)
            if max_patients is not None and i >= (max_patients - 1):
                break

    output_times = []
    with open(f"{output_file}", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        header = ["CSN"]
        header.extend(
            [f"orig_len", f"orig_covered", f"orig_total", f"orig_coverage", f"best_len", f"best_covered", f"best_total",
             f"best_coverage"])
        for col_idx, col in enumerate(cols):
            header.append(f"{col}_covered")
            header.append(f"{col}_total")
        header.append(f"runtime")
        writer.writerow(header)
        for future in futures.as_completed(fs):
            # Blocking call - wait for 1 hour for a single future to complete
            # (highly unlikely, most likely something is wrong)
            row, times_list = future.result(timeout=60 * 60)
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
                        help='''
                            The absolute path to the file containing vital signs charted by the nurses. The headers should include:
                            - CSN: The patient visit identifier
                            - Variable: The modality of the charted event (HR/RR/SpO2/MAP)
                            - Time: The ISO-8601 formatted timestamp of the charted event
                            - Charted: The recorded value of the charted event
                        ''')
    parser.add_argument('-o', '--output-file',
                        required=True,
                        help='''
                            The absolute path to the CSV file that will be generated that will indicate the calculated optimal coverage for each patient.
                        ''')
    parser.add_argument('-l', '--output-list-file',
                        required=True,
                        help='''
                            The absolute path to the Pickle file that will be generated with the following structure:
                            {
                                "output_times": [['2021-01-02T00:00:00Z', '2021-01-02T01:00:00Z']]
                            }
                            Note that output_times is a list of length n, where n represents the number of patients (the order corresponds to the order in the CSV output file). Each element in this list represents time stamps that led to the optimal coverage for the corresponding patient.
                        ''')
    parser.add_argument('-m', '--interpolate-missing',
                        default="f",
                        help='''
                            Use 't' to indicate that the script should interpolate missing points by carrying forward the last known value. Defaults to 'f' otherwise.
                        ''')
    parser.add_argument('-r', '--patient-folder',
                        default="f",
                        help='''
                            The absolute path to the root folder where the patient CSV files reside (e.g. the files containing the monitor vital signs for each patient).
                        ''')
    parser.add_argument('-c', '--columns',
                        default="HR,RR,SpO2,MAP",
                        help='''
                            Sets the modalities to retrieve as a comma separated list - note each modality is treated equally.'
                        ''')
    parser.add_argument('-s', '--population-size',
                        default=50,
                        help='''
                            Sets the number of individuals to use in the population based method.'
                        ''')
    parser.add_argument('-t', '--max-iterations',
                        default=20,
                        help='''
                            Sets the maximum number of iterations.'
                        ''')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='''
                            Sets the maximum number of patients to process, useful for debugging. Defaults to no limit.'
                        ''')

    args = parser.parse_args()

    run(args)

    print("DONE")
