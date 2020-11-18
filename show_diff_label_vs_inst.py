import cv2
import numpy as np
import glob
import json

json_file = open("./datasets/coco_stuff/label_map.json", "r")
class_name = json.load(json_file)

label_dir = "./datasets/coco_stuff/val_label/"
inst_dir = "./datasets/coco_stuff/val_inst/"
globed_l = glob.glob(label_dir + "*.png")
globed_i = glob.glob(inst_dir + "*.png")

for l in range(len(globed_l)):
    print(globed_l[l])
    print(globed_i[l])
    
    label = cv2.imread(globed_l[l])
    inst = cv2.imread(globed_i[l])
    
    label = label[:,:,0]
    inst = inst[:,:,0]
    print(label.shape)
    unique_l = np.unique(label)
    unique_i = np.unique(inst)
    print(unique_l)
    print(unique_i)
    for class_index in unique_l:
        if class_index > 181:
            class_index = 255
        print(class_name[str(class_index)])

    for inst_index in unique_i:
        inst_mask = np.where(inst == inst_index, 255, 0)
        filename = globed_i[l].split("/")[-1].split(".png")[0] + "_" + str(inst_index) + ".png"
        cv2.imwrite(filename, inst_mask)
    
