import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE
from torch.utils.data import DataLoader
import torch
import numpy as np
# import polars as pl
import psutil
import os
from tqdm import tqdm


class TFTTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_gpu()

    def setup_gpu(self):
        if self.config.use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(self.config.gpu_device)
            print(f"Using GPU: {torch.cuda.get_device_name(self.config.gpu_device)}")
        else:
            print("GPU not available, using CPU.")
            self.config.use_gpu = False

    def r_squared(self, y_true, y_pred, weights=None):
        if weights is None:
            weights = np.ones_like(y_true)
        weighted_mean = np.average(y_true, weights=weights)
        total_ss = np.sum(weights * (y_true - weighted_mean) ** 2)
        residual_ss = np.sum(weights * (y_true - y_pred) ** 2)
        r2 = 1 - (residual_ss / total_ss)
        return r2

    def train(self, data, val):
        # training_cutoff = data["time_idx"].max() - self.config.max_prediction_length
        training = TimeSeriesDataSet(
            data.to_pandas(use_pyarrow_extension_array=True),
            time_idx="time_id",
            target="responder_6",
            group_ids=["symbol_id"],
            max_encoder_length=self.config.max_encoder_length,
            max_prediction_length=self.config.max_prediction_length,
            # static_categoricals=["symbol_id"],
            time_varying_known_reals=["time_idx"]
            + [f"feature_{str(i).zfill(2)}" for i in range(79)],
            time_varying_unknown_reals=["responder_6"],
            target_normalizer=NaNLabelEncoder(add_nan=True),
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            val.to_pandas(use_pyarrow_extension_array=True),
            predict=True,
            stop_randomization=True,
        )

        train_dataloader = DataLoader(
            training, batch_size=self.config.batch_size, shuffle=True
        )

        val_dataloader = DataLoader(
            validation, batch_size=self.config.batch_size*10, shuffle=False
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_continuous_size,
            output_size=self.config.output_size,
            loss=SMAPE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=self.config.max_epochs,
            enable_model_summary=True,
            gradient_clip_val=0.1,
        )

        def get_all_subclasses(cls):
            all_subclasses = []

            for subclass in cls.__subclasses__():
                all_subclasses.append(subclass)
                all_subclasses.extend(get_all_subclasses(subclass))

            return all_subclasses
        print(get_all_subclasses(tft.__class__))
        trainer.fit(
            tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

        # Log the training progress
        for epoch in range(self.config.max_epochs):
            tqdm.write(f"Epoch {epoch + 1}/{self.config.max_epochs}")

        print("Training completed!")
