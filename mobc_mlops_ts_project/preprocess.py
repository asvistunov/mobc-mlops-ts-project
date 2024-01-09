from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from dvc.repo import Repo
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"
CONF_PATH = PROJECT_ROOT / "conf"


@hydra.main(version_base=None, config_path=str(CONF_PATH), config_name="preprocess")
def main(cfg: DictConfig):
    sales = pd.read_csv(DATA_PATH / "raw/sales_train_evaluation.csv").query(f'state_id == "{cfg.filters.state_id}"')

    sales_long = sales.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=[col for col in sales.columns if "d_" in col],
        var_name="d",
        value_name="qty_sold",
    )

    sales_long.to_csv(DATA_PATH / "processed/sales_long.csv", index=False)

    calendar = pd.read_csv("data/raw/calendar.csv", parse_dates=["date"])

    sales_clean = sales_long.merge(calendar[["date", "d"]], on="d", validate="many_to_one")[
        [
            "date",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "qty_sold",
        ]
    ]

    sales_clean["item_id"] = sales_clean["item_id"].str.slice(-3)
    sales_clean["dept_id"] = sales_clean["dept_id"].str.slice(-1)
    sales_clean["store_id"] = sales_clean["store_id"].str.slice(-1)

    sales_clean_filtered = sales_clean[sales_clean.dept_id == str(cfg.filters.dept_id)]

    (
        sales_clean_filtered.query(f'dept_id == "{cfg.filters.dept_id}"').to_csv(
            DATA_PATH / "processed/sales_clean.csv", index=False
        )
    )

    sales_clean_filtered.set_index(OmegaConf.to_object(cfg.filters.group_cols), inplace=True)

    ids = sales_clean_filtered.index.drop_duplicates()

    np.random.seed(cfg.train_test_split.random_seed)

    test_ids = np.random.choice(ids, int(len(ids) * cfg.train_test_split.test_size), replace=False)

    test_index = sales_clean_filtered.index.isin(test_ids)

    train = sales_clean_filtered[~test_index]
    test = sales_clean_filtered[test_index]

    for data, split_type in zip([train, test], ["train", "test"]):
        (data.reset_index().to_csv(DATA_PATH / f"processed/{split_type}.csv", index=False))


if __name__ == "__main__":
    Repo(PROJECT_ROOT).pull()
    main()
