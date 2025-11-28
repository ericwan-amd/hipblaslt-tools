import os
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shutil
from tqdm import tqdm
from pathlib import Path
from itertools import product
import ast
import time
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="tensile-out logs directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=Path("./perf_results"),
        help="statistic perf results directory",
    )
    return parser.parse_args()

def parse_perf_result(perf_path: Path):
    gflops, freq, eff = [], [], []
    problem_size, m, n, b, k, size_label = [], [], [], [], [], []

    for perf_file in perf_path.iterdir():
        with open(perf_file, 'r') as file:
            for line in file:
                parts = line.split()
                size = ast.literal_eval(parts[11])
                if size == 0:
                    continue
                gflops.append(parts[1])
                freq.append(parts[2])
                eff.append(float(parts[3][:-1]))
                problem_size.append(size)
                m.append(problem_size[-1][0])
                n.append(problem_size[-1][1])
                b.append(problem_size[-1][2])
                k.append(problem_size[-1][3])
                size_label.append(f"{m[-1]}x{n[-1]}x{b[-1]}x{k[-1]}")

    df = pd.DataFrame({
        "problem_size": problem_size,
        "m": m,
        "n": n,
        "b": b,
        "k": k,
        "size_label": size_label,
        "gflops": gflops,
        "freq": freq,
        "eff": eff
    })
    df_sorted = df.sort_values(by='eff', ascending=True)

    return df_sorted

def plot_perf(output_dir: Path, df):
    scatter_pt_size = 20
    fig_size_x = 10
    fig_size_y = 10
    # Ensure 'gflops' is numeric
    df['gflops'] = pd.to_numeric(df['gflops'], errors='coerce')

    plt.figure(figsize=(fig_size_x, fig_size_y))

    scatter = plt.scatter(df['m'], df['n'], c=df['gflops'], cmap='viridis', s=scatter_pt_size)

    # Ticks (adjust step size to your data scale)
    x_ticks = np.arange(0, max(df['m']) + 1, step=128)
    y_ticks = np.arange(0, max(df['n']) + 1, step=128)
    plt.xticks(x_ticks, rotation=90, ha='right', fontsize=8)
    plt.yticks(y_ticks, fontsize=8)

    plt.title('GFLOPS vs. Problem Size (m, n)')
    plt.xlabel('m')
    plt.ylabel('n')

    cbar = plt.colorbar(scatter)
    cbar.set_label('GFLOPS', fontsize=12)

    # save_path = os.path.join(root_dir, '../gflops_vs_size.png')
    save_path = output_dir / f"gflops_vs_size.png"
    print(f"Saving plot to: {save_path}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # ----------------------------------------------------------------------
    # Plot Efficiency vs. Problem Size
    df['eff'] = pd.to_numeric(df['eff'], errors='coerce')

    plt.figure(figsize=(fig_size_x, fig_size_y))
    scatter = plt.scatter(df['m'], df['n'], c=df['eff'], cmap='viridis', s=scatter_pt_size)

    # Ticks (adjust step size to your data scale)
    x_ticks = np.arange(0, max(df['m']) + 1, step=128)
    y_ticks = np.arange(0, max(df['n']) + 1, step=128)
    plt.xticks(x_ticks, rotation=90, ha='right', fontsize=8)
    plt.yticks(y_ticks, fontsize=8)

    plt.title('Efficiency vs. Problem Size (m, n)')
    plt.xlabel('m')
    plt.ylabel('n')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Efficiency', fontsize=12)

    # save_path = os.path.join(root_dir, '../efficiency_vs_size_mn.png')
    save_path = output_dir / f"efficiency_vs_size_mn.png"
    print(f"Saving plot to: {save_path}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # ----------------------------------------------------------------------
    plt.figure(figsize=(fig_size_x, fig_size_y))

    scatter = plt.scatter(df['m'], df['k'], c=df['eff'], cmap='viridis', s=scatter_pt_size)

    # Ticks (adjust step size to your data scale)
    x_ticks = np.arange(0, max(df['m']) + 1, step=512)
    y_ticks = np.arange(0, max(df['k']) + 1, step=512)
    plt.xticks(x_ticks, rotation=90, ha='right', fontsize=8)
    plt.yticks(y_ticks, fontsize=8)

    plt.title('Efficiency vs. Problem Size (m, k)')
    plt.xlabel('m')
    plt.ylabel('k')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Efficiency', fontsize=12)

    # save_path = os.path.join(root_dir, '../efficiency_vs_size_mk.png')
    save_path = output_dir / f"efficiency_vs_size_mk.png"
    print(f"Saving plot to: {save_path}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # ----------------------------------------------------------------------
    plt.figure(figsize=(fig_size_x, fig_size_y))

    scatter = plt.scatter(df['n'], df['k'], c=df['eff'], cmap='viridis', s=scatter_pt_size)

    # Ticks (adjust step size to your data scale)
    x_ticks = np.arange(0, max(df['n']) + 1, step=512)
    y_ticks = np.arange(0, max(df['k']) + 1, step=512)
    plt.xticks(x_ticks, rotation=90, ha='right', fontsize=8)
    plt.yticks(y_ticks, fontsize=8)

    plt.title('Efficiency vs. Problem Size (n, k)')
    plt.xlabel('n')
    plt.ylabel('k')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Efficiency', fontsize=12)

    # save_path = os.path.join(root_dir, '../efficiency_vs_size_nk.png')
    save_path = output_dir / f"efficiency_vs_size_nk.png"
    print(f"Saving plot to: {save_path}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_eff_by_helper(perf_log_path: Path, eff_log_path: Path):
    
    # Step 1: Read perf_log.log
    with open(perf_log_path, "r") as f:
        log_data = f.read()

    # Step 2: Download the helper script
    url = "https://gist.githubusercontent.com/cliffxzx/457c2b814126370cb281724ff1af4e04/raw"
    nocache = str(int(time.time()))
    response = requests.get(f"{url}?nocache={nocache}")
    helper_script = response.text

    # Save helper script locally
    helper_path = "/tmp/hipblaslthelper.py"
    with open(helper_path, "w") as f:
        f.write(helper_script)

    # Step 3: Execute helper script with -vvv argument
    process = subprocess.Popen(
        ["python3", helper_path, "-vvv"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Send log data to stdin and capture output
    stdout, stderr = process.communicate(input=log_data)

    # Step 4: Write output to perf_log_distall.csv
    with open(eff_log_path, "w") as f:
        f.write(stdout)

    # Optional: Print errors if any
    if stderr:
        print("Errors:", stderr)

def regular_match_after_pattern(input_file: str, output_file: str, pattern: str = "%  gflops"):
    lines_to_write = []

    # Read all lines from the input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Find lines containing the pattern and capture the next line
    for i, line in enumerate(lines):
        if pattern in line and i + 1 < len(lines):
            lines_to_write.append(lines[i + 1])

    # Write the extracted lines to the output file
    with open(output_file, "w") as f:
        f.writelines(lines_to_write)

def aggregate_all_eff_results(perf_results: Path, output_dir: Path):
    df_perf = parse_perf_result(perf_results)
    plot_perf(output_dir, df_perf)
    df_perf.to_csv(output_dir / f"efficiency_report.csv", index=False)

def main():
    args = arg_parse()

    args.output.mkdir(parents=True, exist_ok=True)

    pref_log_dir = args.input

    perf_output_dir = args.output

    if perf_output_dir.exists():
        shutil.rmtree(perf_output_dir)

    intermidiate_output_dir = perf_output_dir / f"intermidiate_results"
    intermidiate_output_dir.mkdir(parents=True, exist_ok=True)

    eff_distall_output_dir = perf_output_dir / f"eff_distall_results"
    eff_distall_output_dir.mkdir(parents=True, exist_ok=True)

    if pref_log_dir.exists():
        perf_log_files = [f for f in pref_log_dir.iterdir() if f.is_file()]
        for log_path in perf_log_files:
            eff_log_path = intermidiate_output_dir / f"{log_path.stem}_eff.log"
            eff_log_distall_path = eff_distall_output_dir / f"{log_path.stem}_distall.csv"
            calculate_eff_by_helper(log_path, eff_log_path)
            regular_match_after_pattern(eff_log_path, eff_log_distall_path)
        aggregate_all_eff_results(eff_distall_output_dir, perf_output_dir)
    else:
        print(f"Path {pref_log_dir} does not exist logic yaml.")
    

if __name__ == "__main__":
    main()
