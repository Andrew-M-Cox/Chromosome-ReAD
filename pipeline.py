import streamlit as st

from fastai.vision.all import * 
from fastai.vision.data import * 
from fastai.vision.core import * 
from fastai2_extensions.interpret.all import *
import numpy as np
import matplotlib.pyplot as plt
import warnings 
import logging

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
logger = logging.getLogger(__name__)

#TODO: Check this actually loads the model using Gradient
def load_model(): 
    path = 'resources/models'
    learn = load_learner(path, 'fastai_resnet50.pkl')
    return learn
PreTrainedModel = load_model

class ResNet50Pipeline: 
    """Poor man's ResNet pipeline"""
    def __init__(
        self, 
        model: PreTrainedModel, 
        use_cuda: bool
    ):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

    def __call__(self, inputs: array): 
        inputs = inputs

import requests

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


if __name__ == "__main__":
    import sys
    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)