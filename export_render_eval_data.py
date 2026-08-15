import csv
import json
import shutil
from pathlib import Path


BASE_PATH = Path("./data/different_types")
OUTPUT_PATH = Path("./data/render_eval_data")
DATA_CONFIG_PATH = Path("./data_config.csv")
CONTROLLER_NAME = "hand"


def ensure_dir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


def copytree_merge(src: Path, dst: Path):
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_case(case_name: str):
    case_path = BASE_PATH / case_name
    if not case_path.exists():
        return

    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    case_output = OUTPUT_PATH / case_name
    mask_output = case_output / "mask"
    ensure_dir(case_output)
    ensure_dir(mask_output)

    # RGB frames are shared across all views, so copy them once per case.
    copytree_merge(case_path / "color", case_output / "color")

    for view_idx in range(3):
        with open(case_path / "mask" / f"mask_info_{view_idx}.json", "r") as f:
            data = json.load(f)

        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)

        if obj_idx is None:
            raise ValueError(f"No object found for case={case_name} view={view_idx}")

        copytree_merge(
            case_path / "mask" / str(view_idx) / str(obj_idx),
            mask_output / str(view_idx),
        )

    shutil.copy2(case_path / "split.json", case_output / "split.json")


def main():
    ensure_dir(OUTPUT_PATH)

    with open(DATA_CONFIG_PATH, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            case_name = row[0]
            _category = row[1]
            _shape_prior = row[2]
            copy_case(case_name)


if __name__ == "__main__":
    main()
