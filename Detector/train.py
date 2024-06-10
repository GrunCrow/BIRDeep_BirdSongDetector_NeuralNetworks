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
#model = YOLO(MODEL_NAME)

# Load a pretrained YOLO model (recommended for training)
#model = YOLO(MODEL_WEIGHTS)

# Nano model
model = YOLO("yolov8s.pt")

# Train the model using the 'coco128.yaml' dataset for 3 epochs
model.train(
    data=DATASET_YAML, 
    device = 0,                   # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    epochs = 500,
    patience = 50,
    name = "8_AugmentedBG_Small", # "2_BaseExperimentBinary_Small",      # experiment name
    resume = False,	            # resume training from last checkpoint
    single_cls = True,	        # train multi-class data as single-class -> def = False
    cfg="Detector/config/config.yaml",
    pretrained=True,

    optimizer = "auto", # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
)

# Grid search optimization
optimizers = ["SGD", "Adam", "RMSProp"]
lr0_values = [0.001, 0.01, 0.1]
momentum_values = [0.9, 0.937, 0.99]
weight_decay_values = [0.0001, 0.0005, 0.001]

'''for optimizer in optimizers:
    for lr0 in lr0_values:
        for momentum in momentum_values:
            for weight_decay in weight_decay_values:
                print("====================================== EXPERIMENT ======================================")
                print("Optimizer: ", optimizer)
                print("lr0: ", lr0)
                print("momentum: ", momentum)
                print("weight_decay: ", weight_decay)
                print("==========================================================================================")

                comet_ml.init("BIRDeep") # it get the name of the project name on training

                model = YOLO("yolov8s.pt")

                # Train the model using the 'coco128.yaml' dataset for 3 epochs
                model.train(
                    data=DATASET_YAML, 
                    device = 0,                   # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
                    epochs = 500,
                    patience = 50,
                    name = "6_Binary_" + optimizer + "_lr0" + str(lr0) + "_momentum" + str(momentum) + "_wd" + str(weight_decay) + "_Small", # "2_BaseExperimentBinary_Small",      # experiment name
                    resume = False,	            # resume training from last checkpoint
                    single_cls = True,	        # train multi-class data as single-class -> def = False
                    cfg="Detector/config/config.yaml",
                    pretrained=True,

                    optimizer = optimizer, # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
                    
                    cos_lr = True,
                    lr0 = lr0, # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                    lrf = 0.01, # (float) final learning rate (lr0 * lrf)
                    momentum = momentum, # (float) SGD momentum/Adam beta1
                    weight_decay = weight_decay, # (float) optimizer weight decay 5e-4
                    warmup_epochs = 3.0, # (float) warmup epochs (fractions ok)
                    warmup_momentum = 0.8, # (float) warmup initial momentum
                    warmup_bias_lr = 0.1, # (float) warmup initial bias lr
                    box = 7.5, # (float) box loss gain
                    cls = 0.5, # (float) cls loss gain (scale with pixels)
                    dfl = 1.5 # (float) dfl loss gain
                    )'''

'''model.val(
    split = "val", # val, test or train
    conf = 0.4,  # confidence threshold
)'''

# What Manuel wanted :´) -> for classification model, will it work with detection???????????????????
'''# Run inference on an image
results = model('bus.jpg')  # results list

# View results
for r in results:
    print(r.probs)  # print the Probs object containing the detected class probabilities'''

# Export the model to ONNX format
#success = model.export(format='onnx')