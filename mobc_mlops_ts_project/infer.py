from pathlib import Path

import hydra
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.catboost_model import CatBoostModel
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"
CONF_PATH = PROJECT_ROOT / "conf"


@hydra.main(version_base=None, config_path=str(CONF_PATH), config_name="modeling")
def main(cfg: DictConfig):
    sales = pd.read_csv(DATA_PATH / "processed" / "train.csv", parse_dates=["date"])

    sales_ts = TimeSeries.from_group_dataframe(
        sales,
        group_cols=OmegaConf.to_object(cfg.model.group_cols),
        time_col=cfg.model.time_col,
        value_cols=cfg.model.value_cols,
    )

    catboost_model = CatBoostModel.load(DATA_PATH / "models" / cfg.model.name)

    sales_forecast_ts = catboost_model.predict(cfg.model.horizon, sales_ts)

    sales_forecast = []

    for ts in sales_forecast_ts:

        static_covs = ts.static_covariates.to_dict(orient="records")[0]

        forecast_tmp = ts.pd_dataframe().reset_index().assign(**static_covs)

        sales_forecast.append(forecast_tmp)

    (pd.concat(sales_forecast).to_csv(DATA_PATH / cfg.inference.output_file, index=False))


if __name__ == "__main__":
    main()
