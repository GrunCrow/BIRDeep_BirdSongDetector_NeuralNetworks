import comet_ml # pip install comet_ml
from comet_ml.integration.pytorch import watch
from comet_ml.integration.pytorch import log_model

# Pytorch: https://pytorch.org/get-started/locally/

from ultralytics import YOLO # pip install ultralytics
from constants import *
from constants_unshared import MY_API_KEY_COMET
import os

os.environ["COMET_API_KEY"] = MY_API_KEY_COMET
os.environ["COMET_AUTO_LOG_GRAPH"] = "true"
os.environ["COMET_AUTO_LOG_PARAMETERS"] = "true"
os.environ["COMET_AUTO_LOG_METRICS"] = "true"
os.environ["COMET_LOG_PER_CLASS_METRICS"] = "true"
os.environ["COMET_MAX_IMAGE_PREDICTIONS"] = "50"

# initialize experiment in comet
comet_ml.init("BIRDeep") # it get the name of the project name on training

# Create a new YOLO model from scratch
model = YOLO(MODEL_NAME)

# Load a pretrained YOLO model (recommended for training)
model = YOLO(MODEL_WEIGHTS)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
model.train(
                    data=DATASET_YAML, 
                    device = 0,                   # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
                    epochs = 500,
                    patience = 30,
                    name = "2_BaseExperimentBinary_Small",      # experiment name
                    resume = False,	            # resume training from last checkpoint
                    single_cls = True,	        # train multi-class data as single-class -> def = False
                    cfg="Detector/config/config.yaml",
                    pretrained=True
                    )

model.val(
    conf = 0.5,  # confidence threshold
)

# What Manuel wanted :´) -> for classification model, will it work with detection???????????????????
'''# Run inference on an image
results = model('bus.jpg')  # results list

# View results
for r in results:
    print(r.probs)  # print the Probs object containing the detected class probabilities'''

# Export the model to ONNX format
#success = model.export(format='onnx')