from pathlib import Path

import hydra
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.catboost_model import CatBoostModel
from darts.utils.model_selection import train_test_split
from dvc.repo import Repo
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"
CONF_PATH = PROJECT_ROOT / "conf"

Repo(PROJECT_ROOT).pull()


@hydra.main(version_base=None, config_path=str(CONF_PATH), config_name="train")
def main(cfg: DictConfig):

    sales = pd.read_csv(
        DATA_PATH / "processed" / "sales_clean.csv", parse_dates=["date"]
    )

    sales_ts = TimeSeries.from_group_dataframe(
        sales,
        group_cols=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        time_col="date",
        value_cols="qty_sold",
    )

    train_items, _ = train_test_split(
        sales_ts, test_size=cfg.train_test_split.test_size, axis=0
    )

    train, test = train_test_split(
        train_items,
        test_size=cfg.train_test_split.test_size,
        horizon=cfg.model.horizon,
        axis=1,
    )

    catboost_model = CatBoostModel(
        lags=cfg.model.lags,
        use_static_covariates=cfg.model.use_static_covariates,
        output_chunk_length=cfg.model.output_chunk_length,
    )

    catboost_model.fit(train, verbose=3)


if __name__ == "__main__":
    main()
