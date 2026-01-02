import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from problem_size_navi4x_gridbased import *


def generate_points(spec):
    """
    Accepts either:
    - [m, n, b, k] → exact point
    - [[m_start, m_step, m_end], [n_start, n_step, n_end], [1], [k]] → generate all combinations
    Returns: list of [m, n, b, k] points
    """
    # Exact single point
    if all(isinstance(dim, int) for dim in spec):
        return [spec]
    
    # Expand dimension ranges
    def expand(dim):
        if len(dim) == 1:
            return [dim[0]]
        elif len(dim) == 3:
            return list(range(dim[0], dim[2] + 1, dim[1]))
        else:
            raise ValueError(f"Unsupported dimension format: {dim}")
    
    expanded_dims = [expand(dim) for dim in spec]

    return [list(p) for p in product(*expanded_dims)]

def plot_points(points, save_path):
    m_coords = [p[0] for p in points]
    n_coords = [p[1] for p in points]
    k_coords = [p[3] for p in points]

    plt.figure(figsize=(8, 8))
    # scatter = plt.scatter(m_coords, n_coords, c=k_coords, cmap='viridis', s=20, marker='o')
    scatter = plt.scatter(m_coords, n_coords, cmap='blue', s=20, marker='o')
    plt.title('Scatter plot of generated points', fontsize=16)
    plt.xlabel('m', fontsize=14)
    plt.ylabel('n', fontsize=14)

    # Tick intervals: adjust as needed
    max_m, max_n = max(m_coords), max(n_coords)
    x_ticks = np.arange(0, max_m + 1, max(1, max_m // 64))
    y_ticks = np.arange(0, max_n + 1, max(1, max_n // 64))

    plt.xticks(x_ticks, fontsize=7, rotation=90)
    plt.yticks(y_ticks, fontsize=7, rotation=0)

    # cbar = plt.colorbar(scatter)
    # cbar.set_label('k', fontsize=12)

    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, format='jpg')
    print(f"Saved plot to {save_path}")
    plt.close()

def add_unique_problem_size_to_table(pt_range_ls, unique_pts, problem_size_table):
    for pr in pt_range_ls:
        for p in generate_points(pr):
            if tuple(p) in problem_size_table:
                continue
            unique_pts.append(p)
            problem_size_table.add(tuple(p))

def main():
    # Example exact point
    all_pts = []
    problem_size_table: set[Tuple[int, int, int, int]] = set()

    add_unique_problem_size_to_table(range_pts_basic_p1, all_pts, problem_size_table)

    add_unique_problem_size_to_table(range_pts_basic_p2_1, all_pts, problem_size_table)

    add_unique_problem_size_to_table(range_pts_basic_p2_2, all_pts, problem_size_table)

    add_unique_problem_size_to_table(range_pts_basic_p2_3, all_pts, problem_size_table)

    add_unique_problem_size_to_table(range_pts_basic_p2_4, all_pts, problem_size_table)

    add_unique_problem_size_to_table(range_pts_enlarge_p1, all_pts, problem_size_table)

    add_unique_problem_size_to_table(range_pts_enlarge_p2, all_pts, problem_size_table)

    add_unique_problem_size_to_table(range_pts_enlarge_p3, all_pts, problem_size_table)

    MT0 = 256
    MT1 = 256
    CU_COUNT = 64
    Heuristic_times = 12
    threshold_problem_size = MT0 * MT1 * CU_COUNT * Heuristic_times
    filtered_pts = [pt for pt in all_pts if pt[0] * pt[1] <= MT0 * MT1 * CU_COUNT * Heuristic_times]

    print(f"Number of unique points: {len(problem_size_table)}")
    print(f"Number of filtered_pts points: {len(filtered_pts)}")
    plot_points(filtered_pts, "../test_data/f8_points_enlarge_p3.jpg")

if __name__ == "__main__":
    main()
