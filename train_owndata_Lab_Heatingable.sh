GPU=0
PORT_BASE=6001
GT_PATH=/home/jiangzhenghan/project/Lab

DATASET=Lab
SAVE_PATH=output

SCENE_LIST=(
    Heatingtable
)
for SCENE in "${SCENE_LIST[@]}"; do
    echo "scene: $SCENE"
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s $GT_PATH/$SCENE/rgb --port $(expr $PORT_BASE + $GPU) --model_path $SAVE_PATH/$DATASET/"$SCENE"_rgb_down2x --expname $DATASET/$SCENE --configs arguments/$DATASET/Heatingtable.py -r 1
    CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/"$SCENE"_rgb_down2x  --skip_train --configs arguments/$DATASET/Heatingtable.py
    # CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/"$SCENE"_rgb_down2x
    CUDA_VISIBLE_DEVICES=$GPU python metrics_v1.py --model_path $SAVE_PATH/$DATASET/"$SCENE"_rgb_down2x --batch_size 16 --rgb

    CUDA_VISIBLE_DEVICES=$GPU python train.py -s $GT_PATH/$SCENE/thermal --port $(expr $PORT_BASE + $GPU) --model_path $SAVE_PATH/$DATASET/"$SCENE"_thermal_down2x --expname $DATASET/$SCENE --configs arguments/$DATASET/Heatingtable.py -r 1
    CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/"$SCENE"_thermal_down2x  --skip_train --configs arguments/$DATASET/Heatingtable.py
    # CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/"$SCENE"_thermal_down2x
    CUDA_VISIBLE_DEVICES=$GPU python metrics_v1.py --model_path $SAVE_PATH/$DATASET/"$SCENE"_thermal_down2x --batch_size 16 --thermal
done
