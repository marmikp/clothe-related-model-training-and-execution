#bin/bash
python run_image_classification.py \
    --model_name_or_path google/vit-base-patch16-224-in21k\
    --train_dir datasets/fabric-dataset/train/ \
    --validation_dir datasets/fabric-dataset/test/ \
    --output_dir ./clothes-fabric-detection/ \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 88 \
    --per_device_eval_batch_size 88 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --metric_for_best_model f1 \
    --ignore_mismatched_sizes True
    
    