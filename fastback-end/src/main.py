import pickle
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from pydantic import BaseModel
from typing import Any
from PIL import Image
import requests
import uvicorn

def AlterLearner(learn, n_channels=1):
  "Adjust a `Learner`'s model to accept `1` channel"
  layer = learn.model[0][0]
  layer.in_channels=n_channels
  layer.weight = nn.Parameter(layer.weight[:,1,:,:].unsqueeze(1))
  learn.model[0][0] = layer
def getLabel(x):
  return x.name.split('_')[0]

if __name__ == "__main__":
    uvicorn.run("app.app:app", host="0.0.0.0", port = 8000, log_level = "debug", proxy_headers=True, reload=True)