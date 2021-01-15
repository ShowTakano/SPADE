"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from util.visualizer import Visualizer
from util import html

import glob
import re

import pandas as pd
import numpy as np
import json

opt = TestOptions().parse()

visualizer = Visualizer(opt)

web_dir = './results'

webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

base_label_dir = opt.base_label_dir
print(base_label_dir)
csv_data = pd.read_csv('./results/metrics/eval_each_LABELvsSS1.csv', index_col=0)
csv_data2 = pd.read_csv('./results/metrics/eval_each_LABELvsSS2.csv', index_col=0)
csv_data3 = pd.read_csv('./results/metrics/eval_each_SS1vsSS2.csv', index_col=0)
f = open('./datasets/coco_stuff/label_map.json','r')
label_map = json.load(f)
files = sorted(glob.glob(os.path.join(base_label_dir, '*.png')), key=natural_keys)
for file in files:
	filename = os.path.basename(file)
	filename_no_extension = os.path.splitext(filename)[0]
	webpage.add_header(filename_no_extension)
	ims = []
	txts = []
	links = []
	
	# original image
	image_name = '../val_img/%s.jpg' % (filename_no_extension)
	label = 'orginal-image'
	ims.append(image_name)
	txts.append(label)
	links.append(image_name)
	
	# original label
	image_name = '../SPADE_org/coco_pretrained/test_latest/images/input_label/%s.png' % (filename_no_extension)
	label = 'original-label'
	ims.append(image_name)
	txts.append(label)
	links.append(image_name)
	
	# deeplab-pytorch label
	image_name = '../SPADE_deeplab-pytorch/coco_pretrained/test_latest/images/input_label/%s.png' % (filename_no_extension)
	label = 'deeplab-pytorch-label'
	ims.append(image_name)
	df = csv_data.query('ImgName == "%s"' % (filename))
	mean_accuracy_val = round(df['MeanAccuracy'].iloc[-1], 3) if df.size > 0 else 'NaN'
	frequency_weighted_iou_val = round(df['FrequencyWeightedIoU'].iloc[-1], 3) if df.size > 0 else 'NaN'
	mean_iou_val = round(df['MeanIoU'].iloc[-1], 3) if df.size > 0 else 'NaN'
	class_iou_val_str = '{\n'
	tmp_length = 0
	for key, class_names in label_map.items():
		if key in df.columns:
			if df.size > 0 and not np.isnan(df[key].iloc[-1]):
				if len(class_iou_val_str) - tmp_length > 30:
					class_iou_val_str += '\n'
					tmp_length = len(class_iou_val_str)
				class_iou_val_str += class_names['class_name'] + ":" + str(round(df[key].iloc[-1], 3)) + ','
	sub_str = class_iou_val_str[:len(class_iou_val_str) - 1]
	class_iou_val_str = sub_str + "\n}"
	txts.append(label + '\n' + 'original-label vs ss1' + '\n' + \
	'MeanAccuracy=' + str(mean_accuracy_val) + '\n' + \
	'FrequencyWeightedIoU=' + str(frequency_weighted_iou_val) + '\n' + \
	'MeanIoU=' + str(mean_iou_val) + '\n'+ 'ClassIoU=' + class_iou_val_str)
	links.append(image_name)
	
	# SPADE image
	image_name = '../SPADE_org/coco_pretrained/test_latest/images/synthesized_image/%s.png' % (filename_no_extension)
	label = 'spade-image'
	ims.append(image_name)
	txts.append(label)
	links.append(image_name)
	
	# SS+SPADE image
	image_name = '../SPADE_deeplab-pytorch/coco_pretrained/test_latest/images/synthesized_image/%s.png' % (filename_no_extension)
	label = 'ss-spade-image'
	ims.append(image_name)
	txts.append(label)
	links.append(image_name)
	
	# deeplab-pytorch label from SPADE image
	image_name = '../ss_predicted_label_2/%s.png' % (filename_no_extension)
	label = 'deeplab-pytorch-label from spade-image'
	ims.append(image_name)
	df = csv_data2.query('ImgName == "%s"' % (filename))
	mean_accuracy_val = round(df['MeanAccuracy'].iloc[-1], 3) if df.size > 0 else 'NaN'
	frequency_weighted_iou_val = round(df['FrequencyWeightedIoU'].iloc[-1], 3) if df.size > 0 else 'NaN'
	mean_iou_val = round(df['MeanIoU'].iloc[-1], 3) if df.size > 0 else 'NaN'
	class_iou_val_str = '{\n'
	tmp_length = 0
	for key, class_names in label_map.items():
		if key in df.columns:
			if df.size > 0 and not np.isnan(df[key].iloc[-1]):
				if len(class_iou_val_str) - tmp_length > 30:
					class_iou_val_str += '\n'
					tmp_length = len(class_iou_val_str)
				class_iou_val_str += class_names['class_name'] + ":" + str(round(df[key].iloc[-1], 3)) + ','
	sub_str = class_iou_val_str[:len(class_iou_val_str) - 1]
	class_iou_val_str = sub_str + "\n}"
	txts.append(label + '\n' + 'original-label vs ss2' + '\n' + \
	'MeanAccuracy=' + str(mean_accuracy_val) + '\n'+ \
	'FrequencyWeightedIoU=' + str(frequency_weighted_iou_val) + '\n' + \
	'MeanIoU=' + str(mean_iou_val) + '\n' + 'ClassIoU='+ class_iou_val_str)
	links.append(image_name)
	
	# deeplab-pytorch label from SPADE image
	image_name = '../ss_predicted_label_2/%s.png' % (filename_no_extension)
	label = 'deeplab-pytorch-label from spade-image'
	ims.append(image_name)
	df = csv_data3.query('ImgName == "%s"' % (filename))
	mean_accuracy_val = round(df['MeanAccuracy'].iloc[-1], 3) if df.size > 0 else 'NaN'
	frequency_weighted_iou_val = round(df['FrequencyWeightedIoU'].iloc[-1], 3) if df.size > 0 else 'NaN'
	mean_iou_val = round(df['MeanIoU'].iloc[-1], 3) if df.size > 0 else 'NaN'
	class_iou_val_str = '{\n'
	tmp_length = 0
	for key, class_names in label_map.items():
		if key in df.columns:
			if df.size > 0 and not np.isnan(df[key].iloc[-1]):
				if len(class_iou_val_str) - tmp_length > 30:
					class_iou_val_str += '\n'
					tmp_length = len(class_iou_val_str)
				class_iou_val_str += class_names['class_name'] + ":" + str(round(df[key].iloc[-1], 3)) + ','
	sub_str = class_iou_val_str[:len(class_iou_val_str) - 1]
	class_iou_val_str = sub_str + "\n}"
	txts.append(label + '\n' + 'ss1 vs ss2' + '\n' + \
	'MeanAccuracy=' + str(mean_accuracy_val) + '\n'+ \
	'FrequencyWeightedIoU=' + str(frequency_weighted_iou_val) + '\n' + \
	'MeanIoU=' + str(mean_iou_val) + '\n' + 'ClassIoU='+ class_iou_val_str)
	links.append(image_name)
	
	# 描画
	webpage.add_images_and_lines(ims, txts, links, width=opt.display_winsize)

webpage.save()
