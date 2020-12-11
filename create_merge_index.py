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

opt = TestOptions().parse()

visualizer = Visualizer(opt)

web_dir = './results'

webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

files = sorted(glob.glob("./datasets/coco_stuff/val_label_org/*.png"))
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
	txts.append(label)
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
	webpage.add_images(ims, txts, links, width=opt.display_winsize)

webpage.save()
