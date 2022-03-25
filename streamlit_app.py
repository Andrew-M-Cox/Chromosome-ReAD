import streamlit as st
# #import fastai.vision.all and vision.widgets to create widgets
# from fastai.vision.all import *
# from fastai.vision.widgets import *#Make the two text comments below a markdown in your notebook#Malaria Parasite Species Classifier#You need to know the species of malaria for effective treatment? #Then upload an image of malaria parasite.#declare path and load our export.pkl file

from fastai.vision.all import * 
from fastai.vision.data import * 
from fastai.vision.core import * 
from fastai2_extensions.interpret.all import *
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
import numpy as np
import matplotlib.pyplot as plt
import warnings 
import logging


from io import BytesIO


import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageFile, ImageDraw, ImageFont
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


# st.header("Generate ASCII images using GAN")
# st.write("Choose any image and get corresponding ASCII art:")
st.title("Chromosome Predictor")

st.write("This Chromosome Recurrent Abnormality Detector (ReAD) generates quality predictions from single chromosome images.")

st.header("")

# data_path = 'resources/'

# images = {
#         "Chr16": data_path+'65_chr16b.png',
#         "Inv16": data_path+'04_inv16.tiff' ,
#         "Unknown": data_path+'04_unk_ab_7.tiff',
#         }

# img = st.sidebar.selectbox("Select your chart.", list(images.keys()))

# st.image(images[img])

    
uploaded_file = st.file_uploader("Choose an image...")

def asciiart(in_f, SC, GCF,  out_f, color1='black', color2='blue', bgcolor='white'):

    # The array of ascii symbols from white to black
    chars = np.asarray(list(' .,:irs?@9B&#'))

    # Load the fonts and then get the the height and width of a typical symbol 
    # You can use different fonts here
    font = ImageFont.load_default()
    letter_width = font.getsize("x")[0]
    letter_height = font.getsize("x")[1]

    WCF = letter_height/letter_width

    #open the input file
    img = Image.open(in_f)


    #Based on the desired output image size, calculate how many ascii letters are needed on the width and height
    widthByLetter=round(img.size[0]*SC*WCF)
    heightByLetter = round(img.size[1]*SC)
    S = (widthByLetter, heightByLetter)

    #Resize the image based on the symbol width and height
    img = img.resize(S)
    
    #Get the RGB color values of each sampled pixel point and convert them to graycolor using the average method.
    # Refer to https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/ to know about the algorithm
    img = np.sum(np.asarray(img), axis=2)
    
    # Normalize the results, enhance and reduce the brightness contrast. 
    # Map grayscale values to bins of symbols
    img -= img.min()
    img = (1.0 - img/img.max())**GCF*(chars.size-1)
    
    # Generate the ascii art symbols 
    lines = ("\n".join( ("".join(r) for r in chars[img.astype(int)]) )).split("\n")

    # Create gradient color bins
    nbins = len(lines)
    #colorRange =list(Color(color1).range_to(Color(color2), nbins))

    #Create an image object, set its width and height
    newImg_width= letter_width *widthByLetter
    newImg_height = letter_height * heightByLetter
    newImg = Image.new("RGBA", (newImg_width, newImg_height), bgcolor)
    draw = ImageDraw.Draw(newImg)

    # Print symbols to image
    leftpadding=0
    y = 0
    lineIdx=0
    for line in lines:
        color = 'blue'
        lineIdx +=1

        draw.text((leftpadding, y), line, '#0000FF', font=font)
        y += letter_height

    # Save the image file

    #out_f = out_f.resize((1280,720))
    newImg.save(out_f)


def load_image(filename, size=(512,512)):
	# load image with the preferred size
	pixels = Image.open(filename) 
    
	# convert to numpy array
	# pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	# pixels = (pixels - 127.5) / 127.5
	# # reshape to 1 sample
	# pixels = expand_dims(pixels, 0)
	return pixels


def imgGen2(img1):

    # img = data_path + '04_inv16.tiff'
    st.write('Predicting chromosome identity for image.')
    st.image(img1)

    data_path = Path('resources')
    data = ImageDataLoaders.from_folder(data_path, train='val',
                                        valid='test', 
                                        batch_tfms=[*aug_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.0, max_warp=0, p_affine=0),Normalize.from_stats(*imagenet_stats)], 
                                        bs=32, 
                                        resize_method=ResizeMethod.Squish, 
                                        size=(512, 512),
                                        num_workers=0)
    learn = cnn_learner(data, models.resnet50, metrics=[error_rate])
    defaults.device = torch.device('cpu')
    learn.load('03242022_resnet50.h5')
    # learn_inf = load_learner('resources/models/03242022_resnet50.pkl')
    preds=learn.predict(img1)
    st.write(preds)


    gcam = GradCam(learn, img1, None)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(gcam.plot(full_size=False, plot_original=True, figsize=(12, 6)))
    return None
    # inputf = img1  # Input image file name

    # SC = 0.1    # pixel sampling rate in width
    # GCF= 2      # contrast adjustment

    # asciiart(inputf, SC, GCF, "results.png")   #default color, black to blue
    # asciiart(inputf, SC, GCF, "results_pink.png","blue","pink")
    # img = Image.open(img1)
    # img2 = Image.open('results.png').resize(img.size)
    # #img2.save('result.png')
    # #img3 = Image.open('results_pink.png').resize(img.size)
    # #img3.save('resultp.png')
    # return img2	

if uploaded_file is not None:
    src_image = load_image(uploaded_file)
    # image = Image.open(uploaded_file)	
    src_image.save('predict.png')
    # st.image(uploaded_file, caption='Input Image', use_column_width=True)
    #st.write(os.listdir())
    im = imgGen2('predict.png')	
    # st.image(im, caption='Predicted', use_column_width=True) 	



