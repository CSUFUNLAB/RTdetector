# RTdetector
This repository supplements our paper "RTdetector: Deep Transformer Networks for Time Series Anomaly Detection based on Reconstruction Trend" .
## Installation
This code needs Python-3.7 or higher.
```bash
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```

## Dataset Preprocessing
Preprocess all datasets using the command
```bash
python3 preprocess.py SMAP SMD UCR MBA
```
Distribution rights to some datasets may not be available. Check the readme files in the `./data/` folder for more details. If you want to ignore a dataset, remove it from the above command to ensure that the preprocessing does not fail.

## Result Reproduction
All RTdetector experiments are implemented in PyTorch on an NVIDIA GeForce RTX 2080 Ti GPU.

To run a model on a dataset, run the following command:
```bash
python3 main.py --model <model> --dataset <dataset> --retrain
```

We provide a trained model for verification, which can be run directly
```bash
python3 main.py --model RTdetector --dataset <dataset> --test
```

where `<model>` can be either of 'RTdetector','TranAD', 'GDN', 'MAD_GAN', 'MTAD_GAT', 'MSCRED', 'USAD', 'OmniAnomaly', 'LSTM_AD', and dataset can be one of 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'MSDS', 'MBA', 'UCR' and 'NAB. 

