pip install -r requirements.txt

# Set up Torch Forecasting
pip install pytorch-forecasting pytorch-lightning


# Start New Training
## Basic training with default settings
python train.py

## Training with GPU
python train.py --use_gpu

##Training with specific model name and partitions
python train.py --model_name my_model_v1 --partitions 0 1 2 --use_gpu

# Resume Training from Checkpoint
## Resume from latest checkpoint
python train.py --resume

## Resume from specific checkpoint
python train.py --checkpoint lgb_baseline_epoch5_20240121_011015

## Resume with GPU
python train.py --resume --use_gpu

# View available checkpoints
python train.py --list_checkpoints

# Common Training Scenarios
## Start new training with GPU on first 3 partitions
python train.py --model_name first_try --use_gpu --partitions 0 1 2

## Resume interrupted training
python train.py --resume --use_gpu

## Start fresh training with different model name
python train.py --model_name second_try --use_gpu

## View all saved checkpoints
python train.py --list_checkpoints