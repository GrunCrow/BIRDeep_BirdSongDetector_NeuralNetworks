# PATHS
PATH = "Detector/"
DATASET_PATH = "../" # "../../desarrollo/"

# DATASET
# When running a jupyter notebook, it works like from the jupyter notebook location, 
# when runned from jupyter file, it runs from where user is at the terminal
DATASET_YAML = PATH + "birdeep.yaml"
TRAIN_TXT = DATASET_PATH + "Data/Dataset/TXTs/train.txt"
VAL_TXT = DATASET_PATH + "Data/Dataset/TXTs/validation.txt"
TEST_TXT = "Data/Dataset/TXTs/test.txt"
TEST_CSV = DATASET_PATH + "Data/Dataset/CSVs/test.csv"

# MODEL
MODEL_NAME = 'yolov8.yaml'

RESUME = False
MODEL_WEIGHTS_INITIAL = PATH + 'weights/yolov8s.pt' # created on path folder
MODEL_WEIGHTS_BEST = "BIRDeep/1_test2/weights/best.pt"

MODEL_WEIGHTS = MODEL_WEIGHTS_INITIAL # created on path folder

if MODEL_WEIGHTS == MODEL_WEIGHTS_BEST:
    RESUME = True

# ULTRALYTICS CODE

ULTRALYTICS_MULTI_LABEL = False  # Default Value = False -> to obtain all classes conf vector??

def get_best_model_weights(model_name):
    return PATH + "Trainings/YOLOv8/" + model_name + "/weights/best.pt"