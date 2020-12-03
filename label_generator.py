import cv2
import numpy as np
import json
import copy
import scipy.stats as stats
import os


class LabelGenerator():
    def __init__(
        self, 
        filename=None, 
        target_class_name=None, 
        debug=False):

        self.debug = debug
        self.filename = filename
        self.target_class_name = target_class_name
        self.label_map = self._get_label_map()
        self.target_class_idx = self._get_target_class_idx()
        self.label, self.img = self._get_label()
        self.label_eraised = self._get_erased_label()
        self.label_shifted = copy.deepcopy(self.label)

    def _get_label_map(self):
        json_file = open("./datasets/coco_stuff/label_map.json", "r")
        label_map = json.load(json_file)
        return label_map

    def _get_target_class_idx(self):
        target_class_idx = -1
        for key in self.label_map.keys():
            _class_name = self.label_map[key]["class_name"]
            if _class_name == self.target_class_name:
                target_class_idx = int(key)
        assert target_class_idx != -1, "not sutch target class name."
        return target_class_idx

    def _get_label(self):
        label = cv2.imread(self.filename)
        label = label[:, :, 0]
        img = None
        val_img_filepath = self.filename.replace("label", "img").replace("png", "jpg")
        if os.path.exists(val_img_filepath):
            img = cv2.imread(val_img_filepath)
        if self.debug:
            if img is not None:
                image = cv2.hconcat([img, cv2.merge([label, label, label])])
                cv2.imshow("image", image)
                cv2.waitKey(0)
            else:
                cv2.imshow("label", label)
                cv2.waitKey(0)
        return label, img

    def _get_erased_label(self):
        label_eraised = copy.deepcopy(self.label)
        for y in range(self.label.shape[0]):
            for x in range(self.label.shape[1]):
                if self.label[y, x] == self.target_class_idx:
                    neighbor = label_eraised[y-1:y+2, x-2:x+0]
                    flatten = neighbor.flatten()
                    flatten = np.delete(flatten, np.where(flatten == self.target_class_idx))
                    mode = stats.mode(flatten)
                    if len(mode[0]) == 0:
                        pass
                    else:
                        label_eraised[y, x] = mode[0][0]
        if self.debug:
            cv2.imshow("label_eraised", label_eraised)
            cv2.waitKey(0)
        return label_eraised

    def _get_mask(self):
        mask = np.where(self.label_shifted == self.target_class_idx, 255, 0)
        mask = mask.astype(np.uint8)
        return mask

    def _get_contours(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours_rmdouble = []
        for c in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[c])
            APPEND = True
            for _c in range(c+1, len(contours)):
                _x, _y, _w, _h = cv2.boundingRect(contours[_c])
                # 自分が相手より小さい場合追加しない
                if x > _x and y > _y and (x+w) < (_x+_w) and (y+h) < (_y+_h):
                    APPEND = False
            if APPEND:
                contours_rmdouble.append(contours[c])
        return contours_rmdouble

    def _shift(self, mask, contours, pix):
        label_shifted = copy.deepcopy(self.label_eraised)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if pix is None:
                x_shift = int(w)
            else:
                x_shift = pix
            _x = x - x_shift
            if _x >= 0:
                mask_moved = np.zeros_like(mask)
                mask_moved[y:y+h, _x:_x+w] = mask[y:y+h, x:x+w]
                label_shifted = np.where(mask_moved == 255, self.target_class_idx, label_shifted)
        if self.debug:
            cv2.imshow("label_shifted", label_shifted)
            cv2.waitKey(0)
        return label_shifted

    def shift(self, pix=None):
        mask = self._get_mask()
        contours = self._get_contours(mask)
        self.label_shifted = self._shift(mask, contours, pix)
        return self.label_shifted


filename = "./datasets/coco_stuff/val_label/000000414385.png" # 路肩のスケートボーダー
#filename = "./datasets/coco_stuff/val_label/000000226417.png" # バイク軍団
#filename = "./datasets/coco_stuff/val_label/000000493772.png" # 霧の中の傘のおばけ
#filename = "./datasets/coco_stuff/val_label/000000292997.png" # 雪の中の事故車

#filename = "./datasets/coco_stuff/val_label/000000000139.png" # 部屋の中の人
#filename = "./datasets/coco_stuff/val_label/000000000785.png" # スキーの人
#filename = "./datasets/coco_stuff/val_label/000000001268.png" # ペリカンを撮っている人
#filename = "./datasets/coco_stuff/val_label/000000001490.png" # SUPの人
#filename = "./datasets/coco_stuff/val_label/000000122166.png" # 自転車の人
#filename = "./datasets/coco_stuff/val_label/000000165518.png" # ライダー

target_class_name = "person"

lgen = LabelGenerator(filename, target_class_name, debug=False)


for i in range(15):
    label_generated = lgen.shift(pix=10)

    filename_dst = filename.replace(".png", "_" + str(i) + ".png")
    cv2.imwrite(filename_dst, label_generated)

    print("lavel generated ", filename_dst)
