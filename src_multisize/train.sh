python main.py --data_url '../dataset/train_val' \
    --train_url './model_snapshots' \
    --data_local '../dataset/data2' \
    --train_data 'train' \
    --val_data 'val' \
    --deploy_script_path './deploy_scripts' \
    --arch 'efficientnet-b5' \
    --num_classes 54 \
    --workers 4 \
    --epochs 120 \
    --pretrained True \
    --seed 0 \
    --batch-size 6 \
    --train_local ./checkpoints \
    --learning-rate 0.005 \
    --weight-decay 1e-4 \
#    --resume ./checkpoints/epoch_6_87.0.pth




# se_resnext101_32x4d, resnet50, pnasnet5large, se_resnet152, resnext101_32x48d_wsl
# efficientnet-b5