import matplotlib.pyplot as plt
import numpy as np
from itertools import product

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

def main():
    exact_pts = [
                 [64, 64, 1, 64], # tune 1.1
                 [256, 256, 1, 256],
                 [512, 512, 1, 512],
                 [1024, 1024, 1, 1024],
                 [2048, 2048, 1, 2048],
                 [3072, 3072, 1, 3072],
                 [4096, 4096, 1, 4096]
                ]
    range_pts = [
                 [[64,64,1024],[64,64,1024],[1],[64]], # tune 1.2
                 [[64,64,1024],[64,64,1024],[1],[256]],
                 [[64,64,1024],[64,64,1024],[1],[512]],
                 [[64,64,1024],[64,64,1024],[1],[1024]],
                 [[64,64,1024],[64,64,1024],[1],[2048]],
                 [[64,64,1024],[64,64,1024],[1],[4096]],        
                 [[64,128,2048],[64,128,2048],[1],[64]],  # tune 1.3
                 [[64,128,2048],[64,128,2048],[1],[256]],
                 [[64,128,2048],[64,128,2048],[1],[512]],
                 [[64,128,2048],[64,128,2048],[1],[1024]],
                 [[64,128,2048],[64,128,2048],[1],[2048]],
                 [[64,128,2048],[64,128,2048],[1],[4096]],
                 [[64,256,4096],[64,256,4096],[1],[64]],
                 [[64,256,4096],[64,256,4096],[1],[256]],
                 [[64,256,4096],[64,256,4096],[1],[512]],
                 [[64,256,4096],[64,256,4096],[1],[1024]],
                 [[64,256,4096],[64,256,4096],[1],[2048]],
                 [[64,256,4096],[64,256,4096],[1],[4096]],
                 [[1024,64,2048],[1024,64,2048],[1],[64]],
                 [[1024,64,2048],[1024,64,2048],[1],[256]],
                 [[1024,64,2048],[1024,64,2048],[1],[1024]],
                 [[1024,64,2048],[1024,64,2048],[1],[4096]],
                 [[4096],[64,256,4096],[1],[64]],
                 [[4096], [64,256,4096],[1],[256]],
                 [[4096], [64,256,4096],[1],[512]],
                 [[4096], [64,256,4096],[1],[1024]],
                 [[4096], [64,256,4096],[1],[2048]],
                 [[4096], [64,256,4096],[1],[4096]],
                 [[64,256,4096],[4096],[1],[64]],
                 [[64,256,4096], [4096],[1],[256]],
                 [[64,256,4096], [4096],[1],[512]],
                 [[64,256,4096], [4096],[1],[1024]],
                 [[64,256,4096], [4096],[1],[2048]],
                 [[64,256,4096], [4096],[1],[4096]]
                 ]

    range_pts_edge = \
                [[[1],[1],[1],[1]]]
                #  [[2,2,16],[2,2,16],[1],[2,2,16]],
                #  [[2,2,16],[2,2,16],[1],[1024,1024,4160]]]

    range_pts_llama_deepseek = \
                [
                #  [[64, 512, 9216], [64, 512, 9216], [1], [512]],
                #  [[64, 512, 8192], [64, 512, 8192], [1], [2048]],
                #  [[64, 512, 8192], [64, 512, 8192], [1], [8192]],
                #  [[64, 512, 8192], [64, 512, 8192], [1], [16384]],
                #  [[64, 512, 8192], [64, 512, 8192], [1], [28672]],

                 [[64, 512, 9216], [64, 512, 9216], [1], [512]],

                #  [[64, 1024, 16384], [64, 1024, 16384], [1], [512]],
                #  [[64, 1024, 16384], [64, 1024, 16384], [1], [2048]],
                #  [[64, 1024, 16384], [64, 1024, 16384], [1], [8192]],
                #  [[64, 1024, 16384], [64, 1024, 16384], [1], [16384]],
                #  [[64, 1024, 16384], [64, 1024, 16384], [1], [28672]],

                #  [[64, 1024, 23552], [64, 1024, 23552], [1], [512]],
                #  [[64, 1024, 23552], [64, 1024, 23552], [1], [2048]],
                #  [[64, 1024, 23552], [64, 1024, 23552], [1], [8192]],
                #  [[64, 1024, 23552], [64, 1024, 23552], [1], [16384]],
                #  [[64, 1024, 23552], [64, 1024, 23552], [1], [28672]],

                ]

    # Example exact point
    all_pts = []
    for pt in exact_pts:
        all_pts += generate_points(pt)

    for pt_range in range_pts:
        all_pts += generate_points(pt_range)

    for pt_range in range_pts_edge:
        all_pts += generate_points(pt_range)

    for pt_range in range_pts_llama_deepseek:
        all_pts += generate_points(pt_range)
    
    MT0 = 256
    MT1 = 256
    CU_COUNT = 64
    Heuristic_times = 12
    threshold_problem_size = MT0 * MT1 * CU_COUNT * Heuristic_times
    print(threshold_problem_size)
    filtered_pts = [pt for pt in all_pts if pt[0] * pt[1] <= MT0 * MT1 * CU_COUNT * Heuristic_times]

    print(f"Number of points: {len(filtered_pts)}")
    plot_points(filtered_pts, "../test_data/f8_points_dense.jpg")

if __name__ == "__main__":
    main()
