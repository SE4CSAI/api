from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
from pydantic import BaseModel
from typing import Any
from PIL import Image
import requests
import uvicorn
import os
import pickle
def AlterLearner(learn, n_channels=1):
  "Adjust a `Learner`'s model to accept `1` channel"
  layer = learn.model[0][0]
  layer.in_channels=n_channels
  layer.weight = nn.Parameter(layer.weight[:,1,:,:].unsqueeze(1))
  learn.model[0][0] = layer
def getLabel(x):
  return x.name.split('_')[0]


app = FastAPI(title= " Bird or Bug Classifier")

# Endpoint for classifiying an image from a URL
@app.post('/predict/image_model/from_url', status_code=200)
async def predict_from_url(request: str):
    return {'status_code': 200, 'predicted_label': 'insect'}
    # from fastai.learner import load_learner
    # learn = load_learner('src/imgClassifier.pkl')
    # try:
    #     img = PILImage.create(requests.get(request, stream=True).raw)
    # except Exception as e:
    #     return {"status_code": 400,
    #             "message": "file could not be opened"
    #             }
    # prediction = learn.predict(img)
    # return {"status_code": 200,
    #         "predicted_label": prediction,
    #         }

# End point for classfying an image from file upload.
@app.post("/upload_image")
def upload_image(file : UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        print('test')
        return {"message" : "There was an error uploading the file"}
    finally:
        file.file.close()
    # learn = load_learner('src/imgClassifier.pkl')
    # try:
    #     img = PILImage.create(file.filename)
    # except Exception as e:
    #     os.remove(file.filename)
    #     return {"status_code": 400,
    #             "message": "file could not be opened"
    #             }
    # prediction, _, probs = learn.predict(img)
    # os.remove(file.filename)
    # return {"status_code": 200,
    #         "predicted_label": prediction,
    #         "probs" : {probs[0].tolist(), probs[1].tolist()}
    #         }
# End point for classifying an audio file
# @app.post("/upload_audio")
# def upload_audio(file : UploadFile = File(...)):
#     try:
#         contents = file.file.read()
#         with open(file.filename, 'wb') as f:
#             f.write(contents)
#     except Exception:
#         return {"message" : "There was an error uploading the file"}
#     finally:
#         file.file.close()
#     learn = load_learner('src/audClassifier.pkl')
#     try:
#         audio = AudioTensor.create(file.filename)
#     except:
#         return {"status_code": 400,"message": "file could not be opened"}
#     prediction = learn.predict(audio)
#     os.remove(file.filename)
#     return {"status_code": 200,
#              "predicted_label": prediction[0],
#              }

