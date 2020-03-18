#!/bin/sh -x
MONODEPTH_PATH=/media/tom/seagate-8tb-ext4/models/monodepth/monodepth/monodepth_simple.py
CHECKPOINT_PATH=/media/tom/seagate-8tb-ext4/models/monodepth/models/model_kitti/model_kitti
IMAGE_PATH=/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/manual_experiments/for_paper

python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/000147_10_dog.png
python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/000147_10_fridge.png
python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/000147_10_dog_with_shadow.png
python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/000147_10_fridge_with_shadow.png

python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/car_outline_normal.png
python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/car_outline_hollow.png
python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/car_outline_top_bottom.png
python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/car_outline_sides.png
python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/car_outline_bottom_sides.png
python3 $MONODEPTH_PATH --checkpoint_path $CHECKPOINT_PATH --image_path $IMAGE_PATH/car_outline_top_sides.png

