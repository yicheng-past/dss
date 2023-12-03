import shutil
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import savgol_filter
from tqdm import tqdm

from src.conf import PrepareDataConfig
from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "step",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
    "anglez_sin",
    "anglez_cos",
    "anglez_abs_diff",
    "anglez_savgol_filter_30",
    "anglez_savgol_filter_120",
    # "rolling_unique_anglez_sum",
    "anglez_abs_diff_rolling_720",
    "anglez_abs_diff_rolling_480",
    "anglez_abs_diff_rolling_60",
    "anglez_abs_diff_rolling_30",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def deg_to_rad(x: pl.Expr) -> pl.Expr:
    return np.pi / 180 * x


def rolling_nunique(series: pl.Series, window_size: int) -> pl.Series:
    # Create a rolling window that collects lists of values within the window
    rolling_lists = series.rolling(window_size).arr()

    # Use apply to count the unique values within each list
    unique_counts = rolling_lists.apply(lambda x: len(set(x)))

    return unique_counts


def rolling_nunique_efficient(series: pl.Series, window: int) -> pl.Series:
    # Convert Polars Series to Pandas Series
    pandas_series = series.to_pandas()

    values = pandas_series.values
    unique_counts = np.zeros(len(values))
    unique_values = set()
    rolling_nunique_result = []
    for i in range(len(values)):
        if i >= window:
            unique_values.discard(values[i - window])
        unique_values.add(values[i])
        unique_counts[i] = len(unique_values)
        if i >= window - 1:
            rolling_nunique_result.append(unique_counts[i])
        else:
            rolling_nunique_result.append(np.nan)

    # Convert the result back to Polars Series
    result_series = pd.Series(rolling_nunique_result, index=pandas_series.index)
    return pl.Series(result_series.name, result_series.values)


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = series_df.with_columns(
        pl.col('anglez').cast(pl.Int64).alias('anglez_int'),
        pl.col('anglez').diff(1).abs().fill_null(0).alias("anglez_abs_diff")
    )
    
    # rolling_unique_counts_5min = rolling_nunique_efficient(series_df['anglez_int'], 60)
    # series_df = series_df.with_columns((pl.Series('rolling_unique_anglez_5min_window', rolling_unique_counts_5min)).fill_null(0))
    
    # shift_intervals = [5, 10, 15, 20, 25, 30]
    # for minutes_earlier in shift_intervals:
    #     shift_amount = minutes_earlier * 12
    #     shifted_column_name = f"rolling_unique_anglez_5min_window_{minutes_earlier}minEarlier"
    #     series_df = series_df.with_columns(
    #         series_df["rolling_unique_anglez_5min_window"].shift(shift_amount).fill_null(0).alias(shifted_column_name)
    #     )
    # sum_columns_expr = [pl.col(f"rolling_unique_anglez_5min_window_{minutes}minEarlier") for minutes in shift_intervals]
    # sum_columns_expr.append(pl.col("rolling_unique_anglez_5min_window"))
    # sum_expr = sum_columns_expr[0]
    # for col_expr in sum_columns_expr[1:]:
    #     sum_expr += col_expr
    # series_df = series_df.with_columns(sum_expr.alias("rolling_unique_anglez_sum"))
    
    # series_df = series_df.with_columns(
    #     (pl.Series('rolling_unique_anglez_5min_window_5minEarlier', series_df['rolling_unique_anglez_5min_window'].shift(60))).fill_null(0),
    #     (pl.Series('rolling_unique_anglez_5min_window_10minEarlier', series_df['rolling_unique_anglez_5min_window'].shift(120))).fill_null(0),
    # )

    window_length_30 = min(360 - 1, len(series_df['anglez_abs_diff']) - 1)
    if window_length_30 % 2 == 0:
        window_length_30 -= 1
    window_length_30 = max(1, window_length_30)
    anglez_savgol_filter_30 = savgol_filter(series_df['anglez_abs_diff'].to_numpy(), window_length_30, 3)
    window_length_120 = min(1440 - 1, len(series_df['anglez_abs_diff']) - 1)
    if window_length_120 % 2 == 0:
        window_length_120 -= 1
    window_length_120 = max(1, window_length_120)
    anglez_savgol_filter_120 = savgol_filter(series_df['anglez_abs_diff'].to_numpy(), window_length_120, 3)
    series_df = series_df.with_columns(pl.Series('anglez_savgol_filter_30', anglez_savgol_filter_30),
                                       pl.Series('anglez_savgol_filter_120', anglez_savgol_filter_120))
    
    series_df = (
        series_df
        .with_columns(
            pl.col("anglez_abs_diff").rolling_mean(8640).alias("anglez_abs_diff_rolling_720"),
            pl.col("anglez_abs_diff").rolling_mean(5760).alias("anglez_abs_diff_rolling_480"),
            pl.col("anglez_abs_diff").rolling_mean(720).alias("anglez_abs_diff_rolling_60"),
            pl.col("anglez_abs_diff").rolling_mean(360).alias("anglez_abs_diff_rolling_30"),
        )
    )
   
    series_df = (
        series_df.with_row_count("step")
        .with_columns(
            *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
            *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
            *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
            pl.col("step") / pl.count("step"),
            pl.col('anglez_rad').sin().alias('anglez_sin'),
            pl.col('anglez_rad').cos().alias('anglez_cos'),
        )
        .select("series_id", *FEATURE_NAMES)
    )
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                    pl.col("anglez_rad"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df)

            # 特徴量をそれぞれnpyで保存
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()
