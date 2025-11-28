
import os
import re
import numpy as np
import pandas as pd
import subprocess
from io import StringIO
import csv
from tqdm import tqdm

perf_key_map = {0: "run", 1: "problem-progress", 2: "solution-progress", 3: "operation", 4: "problem-sizes", 5: "bias-type",
                6: "factor-dim", 7: "activation-type", 8: "solution", 9: "validation", 10: "time-us", 11: "gflops", 12: "empty",
                13: "total-gran", 14: "tiles-per-cu", 15: "num-cus", 16: "tile0-gran", 17: "tile1-gran", 18: "cu-gran", 19: "wave-gran",
                20: "mem-read-bytes", 21: "mem-write-bytes", 22: "temp-edge", 23: "clock-sys", 24: "clock-soc", 25: "clock-mem",
                26: "fan-rpm", 27: "hardware-samples", 28: "enqueue-time"}
extract_perf_map = {0: "%", 1: "gflops", 2: "freq", 3: "eff", 4: "validation", 5: "us", 6: "trans", 7: "type", 8: "run", 9: "problem-progress",
                    10: "operation", 11: "problem-sizes", 12: "bias-type", 13: "factor-dim", 14: "total-gran", 15: "tiles-per-cu", 16: "num-cus",
                    17: "tile0-gran", 18: "tile1-gran", 19: "cu-gran", 20: "wave-gran", 21: "mem-read-bytes", 22: "mem-write-bytes", 23: "temp-edge",
                    24: "clock-soc", 25: "clock-mem", 26: "hardware-samples", 27: "enqueue-time"}
# extract_perf_map += [kernel_param_idx for kernel_param_idx in range(95)]
for i in range(28, 28 + 23):
    kernel_param_idx = i - 28
    extract_perf_map[i] = kernel_param_idx

transA_map = {"Ailk": "N"}
transB_map = {"Bjlk": "T", "Bljk": "N"}

def parse_prob_map(root_dir: str, perf_file: str, output_dir: str, output_file: str) -> None:
    pattern = re.compile(r'\b\d+,\d+/\d+,\d+/\d+\b')
    kernel_map = {}

    file_path = os.path.join(root_dir, perf_file)

    # First, count total lines for tqdm
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)

    # Now process with progress bar
    with open(file_path, "r") as file:
        for line in tqdm(file, total=total_lines, desc="Parsing log"):
            if pattern.search(line):
                aggregate_best_kernel(line.strip(), kernel_map)

    perf_res = init_perf_res()
    for prob_size, attrs in kernel_map.items():
        # perf_res[key].append(kernel_map[key])
        for key in extract_perf_map.values():
            if key == "eff":
                perf_res[key].append(str(attrs[key]) + "%")
            else:
                perf_res[key].append(attrs[key])
    
    # print(perf_res)
    perf_res_df = pd.DataFrame(perf_res)
    
    # Save to CSV without header and without index
    perf_res_df.to_csv(os.path.join(output_dir, output_file), sep=' ', header=False, index=False)

    print(f"Perf result saved to {os.path.join(output_dir, output_file)}.")


def init_perf_res() -> dict:
    res = {}
    for key in extract_perf_map.values():
        res[key] = []
    return res
        
def aggregate_best_kernel(perf_line: str, kernel_map: dict) -> None:
    # Use csv.reader to split while preserving quoted groups
    reader = csv.reader(StringIO(perf_line))
    split_line = next(reader)

    problem_size = split_line[4]
    gflops = split_line[11]
    freq = split_line[23] # clock_sys
    # efflike = gflops / freq
    eff = calculate_eff(gflops, freq)
    if problem_size in kernel_map:
        if eff >= kernel_map[problem_size]["eff"]:
            kernel_map[problem_size] = wrap_perf_format(split_line, eff)
    else:
        kernel_map[problem_size] = wrap_perf_format(split_line, eff)


def calculate_eff(gflops: str, freq: str) -> float:
    cmd = f'python3 -c "$(curl -fsSL https://gist.githubusercontent.com/cliffxzx/457c2b814126370cb281724ff1af4e04/raw?nocache=$(date +%s))" eff f8 {gflops} {freq}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    split = str(result.stdout).split(' ')
    return float(split[-1][:-2])

def wrap_perf_format(split_line: list[str], eff: float) -> dict:
    sol_list = split_line[8].split('_') 
    res = {}
    res[extract_perf_map[0]] = 0 # %
    res[extract_perf_map[1]] = split_line[11] # gflops
    res[extract_perf_map[2]] = split_line[23] # freq
    res[extract_perf_map[3]] = eff # eff
    res[extract_perf_map[4]] = split_line[9]  # validation
    res[extract_perf_map[5]] = split_line[10] # us
    res[extract_perf_map[6]] = identify_trans(split_line[3]) # trans
    res[extract_perf_map[7]] = sol_list[3] # type
    res[extract_perf_map[8]] = split_line[0] # run
    res[extract_perf_map[9]] = split_line[1] # type
    res[extract_perf_map[10]] = split_line[3] # operation
    res[extract_perf_map[11]] = split_line[4] # problem-sizes
    res[extract_perf_map[12]] = split_line[5] # bias-type
    res[extract_perf_map[13]] = split_line[6] # factor-dim
    res[extract_perf_map[14]] = split_line[13] # total-gran
    res[extract_perf_map[15]] = split_line[14] # tiles-per-cu
    res[extract_perf_map[16]] = split_line[15] # num-cus
    res[extract_perf_map[17]] = split_line[16] # tile0-gran
    res[extract_perf_map[18]] = split_line[17] # tile1-gran
    res[extract_perf_map[19]] = split_line[18] # cu-gran
    res[extract_perf_map[20]] = split_line[19] # wave-gran
    res[extract_perf_map[21]] = split_line[20] # mem-read-bytes
    res[extract_perf_map[22]] = split_line[21] # mem-write-bytes
    res[extract_perf_map[23]] = split_line[22] # temp-edge
    res[extract_perf_map[24]] = split_line[24] # clock-soc
    res[extract_perf_map[25]] = split_line[25] # clock-mem
    res[extract_perf_map[26]] = split_line[27] # hardware-samples
    res[extract_perf_map[27]] = split_line[28] # enqueue-time
    for sol_param_idx in range(28, 28 + 23):
        j = sol_param_idx - 28
        res[extract_perf_map[sol_param_idx]] = sol_list[j] # might be different from hipblaslthelper
    return res

def identify_trans(op: str) -> str:
    contraction, l, at, bt, ct, dt = op.split('_')
    trans = ""
    if at in transA_map:
        trans += transA_map[at]
    else:
        print(f"Non-exiting trans {at}")
    if bt in transB_map:
        trans += transB_map[bt]
    else:
        print(f"Non-exiting trans {bt}")
    return trans

def main():
    input_dir = "/root/workspace2/rocm-libraries/projects/hipblaslt/tensilelite/navi_tuning/f8f8s_nn/perf/enlarge_small_piece_itv"
    perf_file = "f8f8s_NNsn_enlarge_p2_1.log"
    output_dir = "/root/apps/test_data/Navi48_tuning/f8f8s_NN/perf_results"
    output_file = "f8f8s_NNsn_enlarge_p2_1_distall.csv"
    parse_prob_map(input_dir, perf_file, output_dir, output_file)

if __name__ == "__main__":
    main()
