import csv
import os
import numpy as np
import ast
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

MT0 = 256
MT1 = 256
CU_COUNT = 64
Magic_Num = 10

def parse_problem_size(root_dir: str, filename: str) -> dict:
    problem_coords = {}
    m_coords = []
    n_coords = []
    b_coords = []
    k_coords = []
    all_pts = []
    pts_all = 0
    with open(os.path.join(root_dir, filename), 'r') as file:
        for line in file:
            if line == '':
                continue
            lst = ast.literal_eval(line[6:])
            if (len(lst) < 4):
                continue
            pts_all += 1
            all_pts.append(lst)
            m, n, b, k = lst
            if filter_unused_size(m, n, b, k):
                continue
            m_coords.append(m)
            n_coords.append(n)
            b_coords.append(b)
            k_coords.append(k)
    problem_coords['m'] = m_coords
    problem_coords['n'] = n_coords
    problem_coords['b'] = b_coords
    problem_coords['k'] = k_coords

    print(f"All points: {pts_all} -> filtered points: {len(m_coords)}")
    print(f"smallest: [{min(m_coords)}, {min(n_coords)}, {min(b_coords)}, {min(k_coords)}]")
    print(f"largest: [{max(m_coords)}, {max(n_coords)}, {max(b_coords)}, {max(k_coords)}]")

    analyze_k_interval(root_dir, problem_coords)

    return problem_coords

def filter_unused_size(m: int, n: int, b: int, k: int) -> bool:
    if m * n > MT0 * MT1 * CU_COUNT * Magic_Num:
        return True
    if m > 20000 or n > 20000:
        return True
    return False

def analyze_k_interval(filePath: str, problem_coords: dict) -> None:
    problem_coords_df = pd.DataFrame(problem_coords)
    problem_coords_df_kSorted = problem_coords_df.sort_values(by='k', ascending=True)
    # problem_coords_df_kSorted.to_csv(filePath + "/gridbase_k_sorted.csv", index=False)
    unique_k = set(problem_coords_df['k'])
    unique_k = sorted(unique_k)
    
    for target_k in unique_k:
        # Call the function
        aggr_k = find_max_mn_row(problem_coords_df_kSorted, target_k)

        # Print the result
        print(f"Problem with the largest m * n for k = {target_k}:")
        print(aggr_k)



def find_max_mn_row(df: pd.DataFrame, target_k: int) -> pd.Series:
    """
    Finds the row in the DataFrame with the largest m * n value for a given k.

    Parameters:
    - df: pandas DataFrame with columns ['m', 'n', 'b', 'k']
    - target_k: the value of k to filter by

    Returns:
    - A pandas Series representing the row with the largest m * n
    """
    filtered_df = df[df['k'] == target_k].copy()
    filtered_df['m_times_n'] = filtered_df['m'] * filtered_df['n']
    max_row = filtered_df.loc[filtered_df['m_times_n'].idxmax()]
    return max_row

def summarize_dimension(values):
    """
    Given a list of integers, return:
    - [start, step, end] if it's an arithmetic sequence
    - [value] if all values are the same
    - sorted list if irregular (optional)
    """
    unique_vals = sorted(set(values))
    if len(unique_vals) == 1:
        return [unique_vals[0]]
    elif len(unique_vals) == 2:
        # Not enough points to confirm a pattern
        return unique_vals

    # Try to detect step
    diffs = [b - a for a, b in zip(unique_vals, unique_vals[1:])]
    step = diffs[0]

    if all(d == step for d in diffs):
        return [unique_vals[0], step, unique_vals[-1]]
    else:
        return unique_vals  # Irregular

def plot_problem_size_distribution(file_root_dir: str, problem_coords: dict) -> None:
    # Plot the points on a 2D grid
    m_coords = problem_coords['m']
    n_coords = problem_coords['n']
    b_coords = problem_coords['b']
    k_coords = problem_coords['k']

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(m_coords, n_coords, c=k_coords, cmap='viridis', s=5)
    plt.title('2D Grid Map of Points')
    plt.xlabel('m', fontsize=14)
    plt.ylabel('n', fontsize=14)

    x_ticks = np.arange(0, max(m_coords)+1, 500)
    y_ticks = np.arange(0, max(n_coords)+1, 500)
    x_tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(x_ticks)]
    y_tick_labels = [str(y) if i % 2 == 0 else '' for i, y in enumerate(y_ticks)]
    plt.xticks(x_ticks, x_tick_labels, fontsize=10, rotation=90)
    plt.yticks(y_ticks, y_tick_labels, fontsize=10, rotation=0)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(y_ticks, fontsize=10)  
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    # Add colorbar to show k values
    cbar = plt.colorbar(scatter)
    cbar.set_label('k value', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(file_root_dir, "grid_map.jpg"), format='jpg')

    # -------- Plotting k_coords ---------
    k_coords = sorted(k_coords)
    plt.figure(figsize=(10, 5))
    plt.plot(k_coords, marker='o', linestyle='-', color='blue')
    plt.title('Plot of k values')
    plt.xlabel('Index')
    plt.ylabel('k value')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(file_root_dir, "k_values_plot.jpg"), format='jpg')

def main():
    file_root_dir = "../test_data/gridBase/"
    csv_file_name = "problem_size_grid_based.csv"
    problem_coords = parse_problem_size(file_root_dir, csv_file_name)
    plot_problem_size_distribution(file_root_dir, problem_coords)

if __name__ == "__main__":
    main()