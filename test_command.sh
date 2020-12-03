#!/bin/sh
python test.py --name coco_pretrained --dataset_mode coco --dataroot ./datasets/coco_stuff --gpu_ids -1 --no_pairing_check --use_generated_label_as_input
