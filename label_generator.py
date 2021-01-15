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
        label_map_filepath="./datasets/coco_stuff/label_map.json", 
        debug=False):

        self.debug = debug
        self.filename = filename
        self.target_class_name = target_class_name
        self.label_map_filepath = label_map_filepath
        self.label_map = self._get_label_map()
        self.target_class_idx = self._get_target_class_idx()
        self.label, self.img = self._get_label()

    def _get_label_map(self):
        json_file = open(self.label_map_filepath, "r")
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

    def _get_mask(self, label_shifted):
        # label画像から、ターゲットクラスに該当する箇所を255 非該当の箇所を0にしたmask画像を得る
        mask = np.where(label_shifted == self.target_class_idx, 255, 0)
        mask = mask.astype(np.uint8)
        return mask

    def _get_contours(self, mask):
        # ターゲットクラスのmask画像から、各連結領域の四隅の座標群を得る
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


class LabelShifter(LabelGenerator):
    def __init__(
        self, 
        filename=None, 
        target_class_name=None, 
        label_map_filepath="./datasets/coco_stuff/label_map.json", 
        debug=False):
        super().__init__(filename, target_class_name, label_map_filepath, debug)

        self.label_map = self._get_label_map()
        self.target_class_idx = self._get_target_class_idx()
        self.label, self.img = self._get_label()
        self.label_eraised = self._get_erased_label()
        self.label_shifted = copy.deepcopy(self.label)

    def _get_erased_label(self):
        # ターゲットラベルを消したlabelを得る
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

    def _shift(self, mask, contours, pix):
        # labelをずらす
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
        # 指定されたpix数labelを繰り返しずらします
        mask = self._get_mask(self.label_shifted)
        contours = self._get_contours(mask)
        self.label_shifted = self._shift(mask, contours, pix)
        return self.label_shifted


class LabelCopyandPaster():
    def __init__(
        self, 
        filename_src=None, 
        target_class_names=None, 
        target_class_indexes=None, 
        filename_dst=None, 
        label_map_filepath="./datasets/coco_stuff/label_map.json", 
        debug=False):

        self.filename_src = filename_src
        self.filename_dst = filename_dst
        self.target_class_names = target_class_names
        self.target_class_indexes = target_class_indexes
        self.label_map_filepath = label_map_filepath
        self.debug = debug

        self.lgens = []
        for target_class_name in self.target_class_names:
            self.lgens.append(LabelGenerator(self.filename_src, target_class_name, self.label_map_filepath))

    def copypaste(self, scale=2.5, offset_x=60, offset_y=140):
        # リサイズしてオフセット位置に貼り付けたlabelとimgを作成する
        lgen_dst = LabelGenerator(self.filename_dst, "person", self.label_map_filepath)
        label_dst = lgen_dst.label
        img_dst = lgen_dst.img

        label_tmp = np.zeros_like(label_dst)
        img_tmp = np.zeros_like(img_dst)
        mask_tmp = np.zeros_like(label_dst)

        for l in range(len(self.lgens)):
            label = self.lgens[l].label
            label = cv2.resize(label, (int(label.shape[1]/scale), int(label.shape[0]/scale)), interpolation=cv2.INTER_NEAREST)
            img = self.lgens[l].img
            img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
            mask_target_class = self.lgens[l]._get_mask(label)
            contours = self.lgens[l]._get_contours(mask_target_class)
            mask_contour = np.zeros_like(mask_target_class)
            idx = self.target_class_indexes[l]
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask_contour[y:y+h, x:x+w] = 255
            mask_tclass_n_cntr = np.where(mask_contour == 255, mask_target_class, mask_contour)
            label_tmp[offset_y:label.shape[0]+offset_y, offset_x:label.shape[1]+offset_x] = label
            img_tmp[offset_y:label.shape[0]+offset_y, offset_x:label.shape[1]+offset_x, :] = img
            mask_tmp[offset_y:label.shape[0]+offset_y, offset_x:label.shape[1]+offset_x] = mask_tclass_n_cntr
            label_dst = np.where(mask_tmp == 255, label_tmp, label_dst)
            label_dst = np.where(lgen_dst.label==142, 142, label_dst)
            img_dst = np.where(cv2.merge([mask_tmp, mask_tmp, mask_tmp]) == [255,255,255], img_tmp, img_dst)
            img_dst = np.where(cv2.merge([lgen_dst.label, lgen_dst.label, lgen_dst.label]) == [142,142,142], lgen_dst.img, img_dst)
        if self.debug:
            cv2.imshow("test", img_dst)
            cv2.waitKey(0)
        return label_dst, img_dst
        

filename1 = "./datasets/coco_stuff/val_label/000000414385.png" # 路肩のスケートボーダー
#filename = "./datasets/coco_stuff/val_label/000000226417.png" # バイク軍団
#filename = "./datasets/coco_stuff/val_label/000000493772.png" # 霧の中の傘のおばけ
#filename = "./datasets/coco_stuff/val_label/000000292997.png" # 雪の中の事故車
#filename = "./datasets/coco_stuff/val_label/000000000139.png" # 部屋の中の人
#filename = "./datasets/coco_stuff/val_label/000000000785.png" # スキーの人
#filename = "./datasets/coco_stuff/val_label/000000001268.png" # ペリカンを撮っている人
#filename = "./datasets/coco_stuff/val_label/000000001490.png" # SUPの人
filename2 = "./datasets/coco_stuff/val_label/000000122166.png" # 自転車の人
#filename = "./datasets/coco_stuff/val_label/000000165518.png" # ライダー


# 路肩の人間を道路へ飛び出させる
target_class_name = "person"
lshif = LabelShifter(filename1, target_class_name, debug=False)
for i in range(41):
    label_generated = lshif.shift(pix=10)

    filename_dst = filename1.replace(".png", "_" + str(i) + ".png")
    cv2.imwrite(filename_dst, label_generated)

    print("lavel generated ", filename_dst)


# 渋滞中の道路に自転車を逆走させる
target_class_names = ["person", "bicycle", "backpack"]
target_class_indexes = [-1, 0, 0]
lcoppaste = LabelCopyandPaster(filename2, target_class_names, target_class_indexes, filename1, debug=False)

scales =     [6,   5.8, 5.6, 5.4, 5.2,   5, 4.8, 4.6, 4.4, 4.2,  4, 3.8, 3.6, 3.4, 3.2, 3.0,  2.8,  2.6,  2.4,  2.2,  2.0, 1.8, 1.6, 1.4, 1.2]
offset_xs =  [135, 131, 127, 123, 119, 115, 111, 107, 103,  99, 95,  94,  90,  86,  82,  75,   67,   59,   51,   43,   35,  25,  15,   5,   0]
offset_ys =  [225, 223, 221, 219, 217, 215, 213, 211, 209, 207,205, 199, 193, 187, 181, 175,  163,  151,  139,  127,  115, 105,  95,  85,  75]

for i in range(len(scales)):
    label_generated, image_generated = lcoppaste.copypaste(scales[i], offset_xs[i], offset_ys[i])

    filename_label_dst = filename2.replace(".png", "_" + str(i) + ".png")

    filename_img = filename2.replace("label", "img").replace("png", "jpg")
    filename_img_dst = filename_img.replace(".jpg", "_" + str(i) + ".jpg")

    cv2.imwrite(filename_label_dst, label_generated)
    cv2.imwrite(filename_img_dst, image_generated)

    print("lavel and image generated ", filename_label_dst, filename_img_dst)