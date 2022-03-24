import streamlit as st

# from fastai.vision.all import * 
# from fastai.vision.data import * 
# from fastai.vision.core import * 
# from fastai2_extensions.interpret.all import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings 
import logging

data_path = 'streamlit_app_images/'

img = data_path + '04_inv16.tiff'
st.image(img)
# data_path = Path('streamlit_app_images/')
# data = ImageDataLoaders.from_folder(data_path, train_name='train',
#                                     valid_name='test', 
#                                     batch_tfms=[*aug_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.0, max_warp=0, p_affine=0),Normalize.from_stats(*imagenet_stats)], 
#                                     bs=32, 
#                                     resize_method=ResizeMethod.Squish, 
#                                     size=(512, 512),
#                                     num_workers=2)

# learn = cnn_learner(data, models.resnet50, metrics=[error_rate])

torch.load('resources/models/020232021_512_res50_round1.h5.pth')
# learn.predict(img)

# gcam = GradCam(learn, img, None)
# gcam.plot(full_size=True, plot_original=False, figsize(12, 6))
