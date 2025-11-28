import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import yaml
import csv
import pandas as pd
import re

from tqdm import tqdm
from pathlib import Path
from typing import List, Set, Tuple
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

M_N_B_K_INDEX_IN_LOGIC_YAML = 7

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--logic",
        type=Path,
        required=True,
        help="logic yaml file",
    )
    parser.add_argument(
        "-t",
        "--template_bench_yaml",
        type=Path,
        required=True,
        help="template bench yaml",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=Path("./output"),
        help="output folder path",
    )
    return parser.parse_args()


def get_logic_yamls(logic_yaml_dir: Path):
    logic_yaml_dir = logic_yaml_dir.resolve()
    return [logic_yaml_path for logic_yaml_path in logic_yaml_dir.rglob("*.yaml")]

def get_m_n_b_k(logic_yaml_path: Path):
    problem_sizes: Set[Tuple[int, int, int, int]] = set()
    with open(logic_yaml_path, "r") as f:
        data = yaml.safe_load(f)
        assert data[M_N_B_K_INDEX_IN_LOGIC_YAML - 1] == [2, 3, 0, 1]
        for m_n_b_k, _ in data[M_N_B_K_INDEX_IN_LOGIC_YAML]:
            problem_sizes.add(tuple(m_n_b_k[:4]))
    return problem_sizes

def write_problem_sizes_to_csv(problem_sizes: Set[Tuple[int, int, int, int]], csv_path: Path):
    with csv_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['m', 'n', 'b', 'k'])
        for size in sorted(problem_sizes):
            writer.writerow(size)

def append_matrix_size_from_csv(problem_csv_path: Path, template_yaml_path: Path, output_yaml_path: Path):
    yaml = YAML()
    yaml.preserve_quotes = True

    # Load original YAML
    with template_yaml_path.open('r') as f:
        data = yaml.load(f)

    # Read CSV and build new sizes
    new_sizes = []
    with problem_csv_path.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            size_entry = CommentedMap()
            size_entry["M"] = int(row["m"])
            size_entry["N"] = int(row["n"])
            size_entry["K"] = int(row["k"])
            size_entry.fa.set_flow_style()
            new_sizes.append(size_entry)

    # Append to matrix_size
    test_block = data["Tests"][0]
    if "matrix_size" not in test_block or test_block["matrix_size"] is None:
        test_block["matrix_size"] = []
    test_block["matrix_size"].extend(new_sizes)

    # Write back preserving style
    with output_yaml_path.open('w') as f:
        yaml.dump(data, f)

    print(f"Updated YAML written to {output_yaml_path}")

def hipblaslt_bench_validate(bench_yaml: Path, log_path: Path):
    with open(log_path, "w") as log_file:
        subprocess.run(["hipblaslt-bench", "--yaml", f"{bench_yaml}"], stdout=log_file, stderr=log_file)

def parse_log_file(file_path: Path):
    rows = []
    columns = []
    solution = None
    kernel = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("[0]:"):
                # Extract column names
                columns = line.split("[0]:")[1].split(",")
                columns = [c.strip() for c in columns]
                columns += ["solution", "kernel"]
            elif re.match(r'^[A-ZNn,0-9.\-e]+', line) and "," in line:
                # Data row
                values = [v.strip() for v in line.split(",")]
                values += [solution, kernel]
                rows.append(values)
            elif line.startswith("--Solution name:"):
                solution = line.split("--Solution name:")[1].strip()
            elif line.startswith("--kernel name:"):
                kernel = line.split("--kernel name:")[1].strip()

    if rows and columns:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame()

def collect_fail_kernels(bench_log: Path):
    print(f"Processing {bench_log}")
    df = parse_log_file(bench_log)
    if df.empty:
        print(f"No data in {bench_log}")
        return

    # Filter rows where 'atol' or 'rtol' is 'failed'
    failed_df = df[(df['atol'] == 'failed') | (df['rtol'] == 'failed')]
    if failed_df.empty:
        print(f"No failed kernels in {bench_log}")

    # Save per-file CSV
    output_file =  bench_log.parent / f"{bench_log.stem}_failed.csv"
    failed_df.to_csv(output_file, index=False)
    print(f"Saved failed kernels to {output_file}")

def main():
    args = arg_parse()

    # Ensure output folder exists
    args.output.mkdir(parents=True, exist_ok=True)

    logic_yaml_path = args.logic
    output_dir = args.output
    template_yaml_path = args.template_bench_yaml
    problem_sizes_csv_path = output_dir / f"{logic_yaml_path.stem}_problems.csv"
    bench_yaml_path = output_dir / f"{logic_yaml_path.stem}_bench.yaml"
    bench_log_path = output_dir / f"{logic_yaml_path.stem}_bench_result.log"
    fail_kernels_csv_path = output_dir / f"{logic_yaml_path.stem}_fail_kernels.csv"

    problem_sizes: Set[Tuple[int, int, int, int]] = set()

    if logic_yaml_path.exists():
        print(f"Validate kernels containing in the log yaml: {logic_yaml_path}")
        problem_sizes = get_m_n_b_k(logic_yaml_path)
        write_problem_sizes_to_csv(problem_sizes, problem_sizes_csv_path)
        append_matrix_size_from_csv(problem_sizes_csv_path, template_yaml_path, bench_yaml_path)
        hipblaslt_bench_validate(bench_yaml_path, bench_log_path)
        collect_fail_kernels(bench_log_path)
    else:
        print(f"Path {logic_yaml_path} does not exist logic yaml.")


if __name__ == "__main__":
    main()