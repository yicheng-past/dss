from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.conf import InferenceConfig
from src.datamodule import load_chunk_features
from src.dataset.common import get_test_ds
from src.models.base import BaseModel
from src.models.common import get_model
from src.utils.common import nearest_valid_size, trace
from src.utils.post_process import post_process_for_seg


def load_model(cfg: InferenceConfig, path=None) -> BaseModel:
    num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    model = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )

    # load weights
    if cfg.weight is not None:
        if path:
            weight_path = path
        else:
            weight_path = (
                Path(cfg.dir.model_dir) / cfg.weight.exp_name / cfg.weight.run_name / "best_model.pth"
            )
        model.load_state_dict(torch.load(weight_path))
        print('load weight from "{}"'.format(weight_path))
    return model


def get_test_dataloader(cfg: InferenceConfig) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    feature_dir = Path(cfg.dir.processed_dir) / cfg.phase
    series_ids = [x.name for x in feature_dir.glob("*")]
    chunk_features = load_chunk_features(
        duration=cfg.duration,
        feature_names=cfg.features,
        series_ids=series_ids,
        processed_dir=Path(cfg.dir.processed_dir),
        phase=cfg.phase,
    )
    test_dataset = get_test_ds(cfg, chunk_features=chunk_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def inference(
    duration: int, loader: DataLoader, models, device: torch.device, use_amp
) -> tuple[list[str], np.ndarray]:
    # Check if models is a single model or a list of models
    if isinstance(models, BaseModel):
        models = [models]

    preds_sum = None
    num_models = len(models)

    for model in models:
        model = model.to(device)
        model.eval()

        preds = []
        keys = []
        for batch in tqdm(loader, desc="inference"):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    x = batch["feature"].to(device)
                    output = model.predict(
                        x,
                        org_duration=duration,
                    )
                if output.preds is None:
                    raise ValueError("output.preds is None")
                else:
                    key = batch["key"]
                    preds.append(output.preds.detach().cpu().numpy())
                    keys.extend(key)

        preds = np.concatenate(preds)

        if preds_sum is None:
            preds_sum = np.zeros_like(preds)

            preds_sum += np.concatenate(preds)

    preds_avg = preds_sum / num_models

    return keys, preds_avg  # type: ignore


def make_submission(
    keys: list[str], preds: np.ndarray, score_th, distance
) -> pl.DataFrame:
    sub_df = post_process_for_seg(
        keys,
        preds,  # type: ignore
        score_th=score_th,
        distance=distance,  # type: ignore
    )

    return sub_df


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)

    with trace("load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)
    with trace("load model"):
        models = []
        for model_path in cfg.path_models:
            models.append(load_model(cfg, model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with trace("inference"):
        keys, preds = inference(cfg.duration, test_dataloader, models, device, use_amp=cfg.use_amp)

    with trace("make submission"):
        sub_df = make_submission(
            keys,
            preds,
            score_th=cfg.pp.score_th,
            distance=cfg.pp.distance,
        )
    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    main()
