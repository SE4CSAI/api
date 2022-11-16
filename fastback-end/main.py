import pickle
import uvicorn
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
# Functions required for the classification model. These were used during training so they are required when loading the saved model.

if __name__ == "__main__":
    uvicorn.run("src.app.app:app", host="0.0.0.0", port = 8000, log_level = "debug", proxy_headers=True, reload=True)