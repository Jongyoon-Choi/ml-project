#!/bin/bash

model="MF"
seed=42
batch_size=64
epochs=(1 2 3)
ks=(1 2 3)
learning_rate=0.001
save_prediction=False

output_dir="experiment_results"
mkdir -p "$output_dir"

for epoch in "${epochs[@]}"; do
  for k in "${ks[@]}"; do
    output_file="$output_dir/${model}_epoch${epoch}_k${k}.log"
    python -u -m main --model "$model" \
                    --seed "$seed" \
                    --batch_size "$batch_size" \
                    --epochs "$epoch" \
                    --k "$k" \
                    --learning_rate "$learning_rate" \
                    --save_predictions "$save_predictions"  > "$output_file" 2>&1 &
  done
done

wait  # 모든 백그라운드 작업이 완료될 때까지 대기