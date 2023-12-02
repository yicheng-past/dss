import numpy as np
import polars as pl
from scipy.signal import find_peaks


def non_maximum_suppression(df, iou_threshold=100):
    grouped = df.groupby(['event', 'series_id'])
    nms_predictions = []

    for _, group in grouped:
        sorted_group = group.sort('score', reverse=True)

        while not sorted_group.is_empty():
            highest = sorted_group[0]
            nms_predictions.append(highest)
            sorted_group = sorted_group.filter((sorted_group['step'] - highest['step']).abs() > iou_threshold)

    nms_df = pl.DataFrame(nms_predictions).sort('row_id').with_column(pl.arange(0, len(nms_predictions)).alias('index')).reset_index(drop=True)
    return nms_df


def post_process_for_seg(
    keys: list[str], preds: np.ndarray, score_th: float = 0.01, distance: int = 5000
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])

    sub_df = non_maximum_suppression(sub_df)
    return sub_df
