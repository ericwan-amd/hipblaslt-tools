import pandas as pd
import re
from pathlib import Path

def parse_log_file(file_path):
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

def main():
    input_dir = Path("../test_data/problem_size_bench/bench_results")
    output_dir = Path("../test_data/problem_size_bench/fail_analysis")
    log_files = list(input_dir.glob("*.log"))
    if not log_files:
        print("No log files found.")
        return

    # all_failed = []

    for file in log_files:
        print(f"Processing {file}")
        df = parse_log_file(file)
        if df.empty:
            print(f"No data in {file}")
            continue

        # Filter rows where 'atol' or 'rtol' is 'failed'
        failed_df = df[(df['atol'] == 'failed') | (df['rtol'] == 'failed')]
        if failed_df.empty:
            print(f"No failed kernels in {file}")
            continue

        # Save per-file CSV
        output_file =  output_dir / f"{file.stem}_failed.csv"
        failed_df.to_csv(output_file, index=False)
        print(f"Saved failed kernels to {output_file}")

    #     all_failed.append(failed_df)

    # # Combine all failed rows into one summary CSV
    # if all_failed:
    #     combined_df = pd.concat(all_failed, ignore_index=True)
    #     combined_df.to_csv("all_failed_kernels.csv", index=False)
    #     print("Saved combined summary: all_failed_kernels.csv")

if __name__ == "__main__":
    main()