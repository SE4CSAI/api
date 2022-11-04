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

app = FastAPI(title= " Bird or Bug Classifier")

# Endpoint for classifiying an image from a URL
@app.post('/predict/image_model/from_url', status_code=200)
async def predict_from_url(request: str):
    learn = load_learner('src/imgClassifier.pkl')
    try:
        img = PILImage.create(requests.get(request, stream=True).raw)
    except Exception as e:
        return {"status_code": 400,
                "message": "file could not be opened"
                }
    prediction = learn.predict(img)
    return {"status_code": 200,
            "predicted_label": prediction,
            }

# End point for classfying an image from file upload.
@app.post("/upload_image")
def upload_image(file : UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message" : "There was an error uploading the file"}
    finally:
        file.file.close()
    learn = load_learner('src/imgClassifier.pkl')
    try:
        img = PILImage.create(file.filename)
    except Exception as e:
        return {"status_code": 400,
                "message": "file could not be opened"
                }
    prediction, _, probs = learn.predict(img)
    return {"status_code": 200,
            "predicted_label": prediction,
            "probs" : {probs[0].tolist(), probs[1].tolist()}
            }

# End point for classifying an audio file
@app.post("/upload_audio")
def upload_audio(file : UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message" : "There was an error uploading the file"}
    finally:
        file.file.close()
    learn = load_learner('src/audClassifier.pkl')
    ## TODO : Setup the pipeline to convert each new sample to the required format for classificaiton,
    ##        Create clean up function to remove downloaded files
    cfg = AudioConfig.BasicSpectrogram()
    aud2spec = AudioToSpec.from_cfg(cfg)
    pipe = Pipeline([AudioTensor.create, aud2spec])
    itemTfms = [ResizeSignal(7000), aud2spec]