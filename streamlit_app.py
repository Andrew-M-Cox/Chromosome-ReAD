import streamlit as st


#import fastai.vision.all and vision.widgets to create widgets
from fastai.vision.all import *
from fastai.vision.widgets import *#Make the two text comments below a markdown in your notebook#Malaria Parasite Species Classifier#You need to know the species of malaria for effective treatment? #Then upload an image of malaria parasite.#declare path and load our export.pkl file

# from fastai.vision.all import * 
# from fastai.vision.data import * 
# from fastai.vision.core import * 
# from fastai2_extensions.interpret.all import *
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings 
# import logging


# data_path = 'resources/'
# img = data_path + '04_inv16.tiff'
# st.image(img)
# data_path = Path(data_path)
# data = ImageDataLoaders.from_folder(data_path, train='val',
#                                     valid='test', 
#                                     batch_tfms=[*aug_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.0, max_warp=0, p_affine=0),Normalize.from_stats(*imagenet_stats)], 
#                                     bs=32, 
#                                     resize_method=ResizeMethod.Squish, 
#                                     size=(512, 512),
#                                     num_workers=2)

# learn = cnn_learner(data, models.resnet50, metrics=[error_rate])
# defaults.device = torch.device('cpu')
# learn.load('020232021_512_res50_round1.h5')
# learn.predict(img)

# gcam = GradCam(learn, img, None)
# st.image(gcam.plot(full_size=True, plot_original=False, figsize=(12, 6)))



#Make the two text comments below a markdown in your notebook
# #Malaria Parasite Species Classifier
# #You need to know the species of malaria for effective treatment? #Then upload an image of malaria parasite.

# #declare path and load our export.pkl file
path = Path('resources')
learn_inf = load_learner(path/'fastai_resnet50.pkl', cpu=True)

#declare a button,output,label widget
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()

#define an on_data_change function which execute when an image is #uploaded.It gets the image uploaded,display the image,make a #prediction of the image and output prediction, probability of #predictions

def on_data_change(change):    
    lbl_pred.value = ''     
    img = PILImage.create(btn_upload.data[-1])     
    out_pl.clear_output()     
    with out_pl: display(img.to_thumb(128,128))     
    pred,pred_idx,probs = learn_inf.predict(img)    
    lbl_pred.value = f'Prediction: {pred}; Probability:{probs[pred_idx]:.04f}'#button to click to upload image
btn_upload.observe(on_data_change, names=['data'])#display label,btn_upload,out_pl,lbl_pred vertically
display(VBox([widgets.Label('Select an Image of Malaria Parasite!'), btn_upload, out_pl, lbl_pred]))