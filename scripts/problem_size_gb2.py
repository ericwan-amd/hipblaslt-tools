import csv
import itertools
from typing import List, Tuple, Optional

def build_ranges_from_csv(csv_path: str,
                          m_step: int = 1,
                          n_step: int = 1,
                          b_step: int = 1,
                          k_step: int = 1,
                          limit_sample: Optional[int] = None):
    """
    Derive per-dimension ranges from a CSV with columns m, n, b, k.
    Returns:
      - ranges: List of [start, step, end] per axis in order [m, n, b, k].
      - minmax: tuple of (m_min, m_max, n_min, n_max, b_min, b_max, k_min, k_max)
    """
    m_vals, n_vals, b_vals, k_vals = [], [], [], []
    with open(csv_path, newline='') as f:
        # reader = csv.DictReader(f)
        # for i, row in enumerate(reader):
        #     if limit_sample is not None and i >= limit_sample:
        #         break
        #     def as_int(x): 
        #         if x is None or x == "": 
        #             raise ValueError("Missing value in CSV for a dimension.")
        #         return int(x)
        #     # Try common header styles
        #     try:
        #         # m_vals.append(as_int(row.get('m', row.get('M'))))
        #         # n_vals.append(as_int(row.get('n', row.get('N'))))
        #         # b_vals.append(as_int(row.get('b', row.get('B'))))
        #         # k_vals.append(as_int(row.get('k', row.get('K'))))
        #         print(row)
        #     except (KeyError, ValueError):
        #         # If headers differ, skip or you can extend this part
        #         pass
        for line in file:
            if line == '':
                continue
            lst = ast.literal_eval(line[6:])
            if (len(lst) < 4):
                continue
            pts_all += 1
            all_pts.append(lst)
            if (lst[0] * lst[1] > 41943040):
                continue
            m_vals.append(lst[0])
            n_vals.append(lst[1])
            b_vals.append(lst[2])
            k_vals.append(lst[3])

    if not m_vals:
        raise ValueError("No data found for m/n/b/k in CSV. Check headers.")

    m_min, m_max = min(m_vals), max(m_vals)
    n_min, n_max = min(n_vals), max(n_vals)
    b_min, b_max = min(b_vals), max(b_vals)
    k_min, k_max = min(k_vals), max(k_vals)

    ranges = [
        [m_min, m_step, m_max] if m_step else [m_min],
        [n_min, n_step, n_max] if n_step else [n_min],
        [b_min, b_step, b_max] if b_step else [b_min],  # if you want fixed, set b_step=0 and adjust code
        [k_min, k_step, k_max] if k_step else [k_min],
    ]
    minmax = (m_min, m_max, n_min, n_max, b_min, b_max, k_min, k_max)
    return ranges, minmax

def generate_grid_from_ranges(ranges: List[List[int]]) -> List[Tuple[int,int,int,int]]:
    """
    Given 4 axis ranges in the form:
      [[m_start, m_step, m_end], [n_start, n_step, n_end],
       [b_start, b_step, b_end] OR [b_value], 
       [k_start, k_step, k_end]]
    Generate the Cartesian product.
    """
    def axis_values(axis_range: List[int]) -> List[int]:
        if len(axis_range) == 1:
            return axis_range
        start, step, end = axis_range
        if step == 0:
            return [start]
        # inclusive range
        vals = list(range(start, end + 1, step))
        return vals

    M = axis_values(ranges[0])
    N = axis_values(ranges[1])
    B = axis_values(ranges[2])
    K = axis_values(ranges[3])

    grid = list(itertools.product(M, N, B, K))
    return grid

def save_grid_to_csv(grid: List[Tuple[int,int,int,int]], path: str):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m','n','b','k'])
        writer.writerows(grid)

def main():
    # Example Mode A: derive from CSV
    csv_path = "../test_data/problem_size_grid_based.csv"  # adjust as needed
    # Derive ranges from CSV, specify your granularity (steps)
    try:
        ranges, _ = build_ranges_from_csv(csv_path, limit_sample=10000)
        grid = generate_grid_from_ranges(ranges)
        print(f"Generated grid with {len(grid)} points from CSV-derived ranges.")
        # save_grid_to_csv(grid, "problem_size_grid_derived_from_csv.csv")
        # print("Saved grid to problem_size_grid_derived_from_csv.csv")
    except Exception as e:
        print(f"Could not derive from CSV: {e}")

    # Example Mode B: explicit ranges (uncomment and customize)
    # ranges_explicit = [
    #     [64, 64, 1024],  # m: 64..1024 step 64
    #     [64, 64, 1024],  # n: 64..1024 step 64
    #     [1],               # b fixed at 1
    #     [4096, 4096, 4096] # k fixed at 4096 (expressed as [start, step, end] or [value])
    # ]
    # grid2 = generate_grid_from_ranges(ranges_explicit)
    # save_grid_to_csv(grid2, "problem_size_grid_explicit.csv")

if __name__ == "__main__":
    main()