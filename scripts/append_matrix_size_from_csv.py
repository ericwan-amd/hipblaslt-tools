from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import yaml
import csv
from pathlib import Path


def read_problem_from_csv(csv_path: Path, problem_sizes: )
def main():
    template_yaml_path = Path("../test_data/problem_size_bench/template_bench.yaml")
    problem_csv_path = Path("../test_data/problem_size_bench/gfx1201_Cijk_Ailk_Bljk_F8F8S_BH_Bias_SHB_HA_S_SAB_SCD_SAV_UserArgs.csv")
    output_yaml_path = Path("../test_data/problem_size_bench/nn_f8f8s.yaml")

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

if __name__ == "__main__":
    main()