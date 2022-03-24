import streamlit as st

from fastai.vision.all import * 
from fastai.vision.data import * 
from fastai.vision.core import * 
from fastai2_extensions.interpret.all import *
import numpy as np
import matplotlib.pyplot as plt
import warnings 
import logging

data_path = 'resources/'

img = data_path + '04_inv16.tiff'
st.image(img)
data_path = Path(data_path)
data = ImageDataLoaders.from_folder(data_path, train='val',
                                    valid='test', 
                                    batch_tfms=[*aug_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.0, max_warp=0, p_affine=0),Normalize.from_stats(*imagenet_stats)], 
                                    bs=32, 
                                    resize_method=ResizeMethod.Squish, 
                                    size=(512, 512),
                                    num_workers=2)

learn = cnn_learner(data, models.resnet50, metrics=[error_rate])
learn.load('020232021_512_res50_round1.h5', map_location='cpu')
learn.predict(img)

gcam = GradCam(learn, img, None)
st.image(gcam.plot(full_size=True, plot_original=False, figsize=(12, 6)))
