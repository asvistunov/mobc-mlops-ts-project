from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, rmse, rmsle
from darts.models.forecasting.catboost_model import CatBoostModel
from darts.utils.model_selection import train_test_split
from dvc.repo import Repo
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"
CONF_PATH = PROJECT_ROOT / "conf"


@hydra.main(version_base=None, config_path=str(CONF_PATH), config_name="modeling")
def main(cfg: DictConfig):

    mlflow.set_tracking_uri(uri=f"http://{cfg.mlflow.host}:{cfg.mlflow.port}")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    sales = pd.read_csv(DATA_PATH / "processed" / "train.csv", parse_dates=["date"])

    sales_ts = TimeSeries.from_group_dataframe(
        sales,
        group_cols=OmegaConf.to_object(cfg.model.group_cols),
        time_col=cfg.model.time_col,
        value_cols=cfg.model.value_cols,
    )

    train, test = train_test_split(
        sales_ts,
        test_size=cfg.train_test_split.test_size,
        horizon=cfg.model.horizon,
        axis=1,
    )

    catboost_model = CatBoostModel(
        lags=cfg.model.lags,
        use_static_covariates=cfg.model.use_static_covariates,
        output_chunk_length=cfg.model.output_chunk_length,
    )
    with mlflow.start_run():

        catboost_model.fit(train, verbose=3)

        forecast = catboost_model.predict(cfg.model.horizon, train)

        for metric in [rmse, mae, rmsle]:

            mlflow.log_metric(
                metric.__name__,
                metric(test, forecast, inter_reduction=np.mean),
            )

            mlflow.log_params(catboost_model.model.get_all_params())

        catboost_model.save(DATA_PATH / "models" / cfg.model.name)


if __name__ == "__main__":
    Repo(PROJECT_ROOT).pull()
    main()
