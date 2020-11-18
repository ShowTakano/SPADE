import cv2
import numpy as np
import glob
import json

# クラスインデックス定義
# https://github.com/nightrome/cocostuff/blob/master/labels.md

json_file = open("./datasets/coco_stuff/label_map.json", "r")
class_name = json.load(json_file)

label_dir = "./datasets/coco_stuff/val_label/"
globed = glob.glob(label_dir + "*.png")

for g in globed:
    print(g)
    label = cv2.imread(g)
    label = label[:,:,0] # 3chennel同じ値が入っている。
    print(label.shape)
    unique = np.unique(label)
    print(unique)
    for class_index in unique:
        if class_index > 181: # クラスインデックスは0~181まで。255はunlabeled
            class_index = 255
        print(class_name[str(class_index)])
    
