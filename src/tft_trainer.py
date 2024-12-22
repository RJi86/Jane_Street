import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE
from torch.utils.data import DataLoader
import torch
import numpy as np
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

    def train(self, data):
        training_cutoff = data["time_idx"].max() - self.config.max_prediction_length
        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            max_encoder_length=self.config.max_encoder_length,
            max_prediction_length=self.config.max_prediction_length,
            static_categoricals=["group_id"],
            time_varying_known_reals=["time_idx", "price"],
            time_varying_unknown_reals=["target"],
            target_normalizer=NaNLabelEncoder(add_nan=True),
        )

        train_dataloader = DataLoader(training, batch_size=self.config.batch_size, shuffle=True)

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

        trainer = pl.Trainer(gpus=1 if self.config.use_gpu else 0, max_epochs=self.config.max_epochs)
        trainer.fit(tft, train_dataloader)

        # Log the training progress
        for epoch in range(self.config.max_epochs):
            tqdm.write(f"Epoch {epoch + 1}/{self.config.max_epochs}")

        print("Training completed!")

# Example usage
if __name__ == "__main__":
    config = {
        "max_encoder_length": 60,
        "max_prediction_length": 30,
        "batch_size": 64,
        "learning_rate": 0.03,
        "hidden_size": 16,
        "attention_head_size": 4,
        "dropout": 0.3,
        "hidden_continuous_size": 8,
        "output_size": 7,
        "use_gpu": True,
        "gpu_device": 0,
        "max_epochs": 30,
    }

    trainer = TFTTrainer(config)
    # Assuming `data` is a preprocessed DataFrame ready for training
    trainer.train(data)