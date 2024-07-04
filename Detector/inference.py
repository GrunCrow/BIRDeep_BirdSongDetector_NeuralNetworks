from ultralytics import YOLO
from constants import *
import os
import pandas as pd

# load best model from training results
# best_model = YOLO(MODEL_WEIGHTS_BEST)
#best_model = YOLO("BIRDeep/4_Binary_Small/weights/best.pt") # BIRDeep/test/weights/best.pt

#best_model = YOLO("BIRDeep/6_Binary_RMSProp_lr00.001_momentum0.9_wd0.001_Small/weights/best.pt") # Todos los valores = 0
# 12_AllBG_LessESC50_Small      9_AugmentedBG_NewLabels_Small
# 11_AllBG__FF_Small    
best_model = YOLO("BIRDeep/12_AllBG_LessESC50_FF_Small/weights/best.pt")



'''df = pd.read_csv(TEST_CSV)

imgs_list = df.transpose().values.tolist()

#new = list(np.concatenate(imgs_list))

# iterate through the sublist using List comprehension
flatList = [element for innerList in imgs_list for element in innerList]'''

'''best_model.val(
    data = "Detector/birdeep.yaml",
    conf = 0.1,  # confidence threshold
    iou = 0.4, # default = 0.6
    split = "val", # val, test or train
    save_json = True,  # save a COCO-JSON results file
    save_hybrid = False,  # save hybrid grid results - ALWAYS SET TO FALSE (https://github.com/ultralytics/ultralytics/issues/6976) - when this flag is set to true it merges the GT with the predictions from the model
    plots = True, # save plots	of prediction vs ground truth
    save_conf = True  # save results with confidence scores
)'''

# get predictions on best model
results = best_model.predict(
    source=TEST_TXT, #"Dataset/multispecies.jpeg", # (str, optional) source directory for images or videos
    save=True, 
    conf=0.15,
    iou=0.2,
    save_txt = True,  # (bool) save results as .txt file
    save_conf = True,  # (bool) save results with confidence scores
    save_crop = False,  # (bool) save cropped images with results

    show = False,  # (bool) show results if possible
    show_labels = True,  # (bool) show object labels in plots
    show_conf = True,  # (bool) show object confidence scores in plots

    visualize = False,  # (bool) visualize model features
    
    
    #vid_stride = 1,  # (int) video frame-rate stride
    #line_width = ,   # (int, optional) line width of the bounding boxes, auto if missing
    #augment = False,  # (bool) apply image augmentation to prediction sources
    # agnostic_nms = False,  # (bool) Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.
    #classes = , # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
    #retina_masks = False,  # (bool) use high-resolution segmentation masks
    #boxes = True  # (bool) Show boxes in segmentation predictions
)

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')
'''

# What Manuel wanted :´) -> for classification model, will it work with detection???????????????????
# Run inference on an image
# results = best_model(source='Detector/AM1_20230510_083000.PNG')  # results list'''

'''# View results
for r in results:
    print(r.B)  # print the Probs object containing the detected class probabilities'''