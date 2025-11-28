#!/usr/bin/env python3
import sys
import argparse
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from typing import Any, List, Dict
from itertools import product

def load_yaml(path: str) -> Any:
    yaml = YAML()
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)
    return data

def save_yaml(path: str, data: Any) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(sequence=4, offset=2)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

def ensure_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return [x]

def walk_and_amend_problem_sizes(node: Any, new_points: List[int], mode_append: bool, mode_replace: bool):
    """
    Recursively walk the YAML structure and amend all ProblemSizes lists found
    in any BenchmarkProblems/.../BenchmarkFinalParameters/ProblemSizes.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            # if k == "ProblemSizes" and isinstance(v, list):
            if k == "ProblemSizes":
                new_v = amend_problem_sizes(v, new_points, mode_append, mode_replace)
                node[k] = new_v
            else:
                walk_and_amend_problem_sizes(v, new_points, mode_append, mode_replace)
    elif isinstance(node, list):
        for item in node:
            walk_and_amend_problem_sizes(item, new_points, mode_append, mode_replace)

def amend_problem_sizes(items: List[Any], new_points: List[int], mode_append: bool, mode_replace: bool) -> List[Any]:
    """
    items: list of ProblemSizes entries (each is expected to be a dict)

    Robust handling:
    - If items is empty, we append a new Exact entry (depending on your policy you can change this)
    - Safely create an inline (flow) list for Exact using ruamel.yaml objects
    """
    if items is None:
        items = []
    # if not isinstance(items, list):
        # return

    for point in new_points:
        new_exact_problem: CommentedMap = CommentedMap()

        # Create a flow-style (inline) list for Exact
        flow_list = CommentedSeq(list(point))
        # ruamel.yaml versions differ; try multiple approaches
        try:
            flow_list.fa.set_flow_style()  # newer versions (no-arg)
        except TypeError:
            try:
                flow_list.fa.set_flow_style(True)  # older versions
            except TypeError:
                # Fallback: leave as default; we still assign the list
                pass

        new_exact_problem['Exact'] = flow_list
        items.append(new_exact_problem)

    return items

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

def create_problems() -> List[List[int]]:
    range_pts_llama_deepseek = \
                [
                    [[64, 1024, 23552], [64, 1024, 23552], [1], [28672]]
                ]

    pts = []
    for pt_range in range_pts_llama_deepseek:
        pts += generate_points(pt_range)

    MT0 = 256
    MT1 = 256
    CU_COUNT = 64
    Heuristic_times = 12
    threshold_problem_size = MT0 * MT1 * CU_COUNT * Heuristic_times
    filtered_pts = [pt for pt in pts if pt[0] * pt[1] <= MT0 * MT1 * CU_COUNT * Heuristic_times]
    print(f"Generate {len(filtered_pts)} problems")
    return filtered_pts
    
def main():
    parser = argparse.ArgumentParser(
        description="Amend all ProblemSizes entries under BenchmarkProblems by adding/ensuring an 'Exact' key."
    )
    parser.add_argument("input_yaml", help="Path to input YAML file")
    parser.add_argument("output_yaml", help="Path to write amended YAML file")
    # parser.add_argument("--points", nargs='+', required=True,
    #                     help="New points to add, e.g. m n b k. These will be used as the list for 'Exact'.")
    parser.add_argument("--append", action="store_true",
                        help="If an item already has 'Exact', append points (deduplicated).")
    parser.add_argument("--replace", action="store_true",
                        help="If an item has 'Exact', replace its contents with the new points.")
    args = parser.parse_args()

    # new_points = list(args.points)
    new_points = create_problems()
    # print(new_points)

    data = load_yaml(args.input_yaml)
    walk_and_amend_problem_sizes(data, new_points, args.append, args.replace)
    save_yaml(args.output_yaml, data)
    print(f"Amendment complete. Output written to: {args.output_yaml}")

if __name__ == "__main__":
    main()
