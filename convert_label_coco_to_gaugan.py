import cv2
import numpy as np
import glob
import json
import os

# クラスインデックス定義
# https://github.com/nightrome/cocostuff/blob/master/labels.md

json_file = open("./datasets/coco_stuff/label_map.json", "r")
label_map = json.load(json_file)

label_dir = "./datasets/coco_stuff/val_label/"
globed = glob.glob(label_dir + "*.png")

dst_dir = "./datasets/coco_stuff/val_label_gaugan/"

print(label_map[str(0)]["class_name"])
print(label_map[str(0)]["rgb_label"])

for g in globed:
    print("input ", g)
    label = cv2.imread(g)
    label = label[:,:,0] # 3chennel同じ値が入っている。
    unique = np.unique(label)
    nums = []
    for u in unique:
        mask = np.where(label==u, 1, 0)
        num = np.sum(mask)
        nums.append(num)
    maxnum = np.max(nums)
    maxidx = nums.index(maxnum)
    label_filled = np.where(label==255, unique[maxidx], label)
    unique_filled = np.unique(label_filled)

    label_r = np.zeros((label.shape[0], label.shape[1]))
    label_g = np.zeros((label.shape[0], label.shape[1]))
    label_b = np.zeros((label.shape[0], label.shape[1]))
    for u in unique_filled:
        rgb_label = label_map[str(u)]["rgb_label"]
        label_r = np.where(label_filled==u, rgb_label[0], label_r)
        label_g = np.where(label_filled==u, rgb_label[1], label_g)
        label_b = np.where(label_filled==u, rgb_label[2], label_b)
    label_bgr = cv2.merge([label_b, label_g, label_r])
    filename = dst_dir + os.path.basename(g)
    cv2.imwrite(filename, label_bgr)
    print("saved ", filename)
print("The unlabeled class has been replaced by the most frequent class.")
    
