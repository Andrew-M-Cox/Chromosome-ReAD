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

