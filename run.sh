#!/bin/bash

# Specify your datasets
datasets=("MBA" "SMAP" "SMD" "MSL" "SWaT" "NAB")
#datasets=("UCR" "MBA" "SMAP" "MSL" "SWaT" "SMD")
#datasets=("")

models=("RTdetector" "RTdetector_DeNormalization" "RTdetector_Destation" "RTdetector_Normalization")
#models=("TranAD_5")
# Loop through each model
for model in "${models[@]}"
do
    # Loop through each dataset and run the command
    for dataset in "${datasets[@]}"
    do
        echo "Running $model on $dataset dataset..."
        python main.py --model $model --dataset $dataset --retrain
        echo "Finished $model on $dataset dataset."
    done
done
