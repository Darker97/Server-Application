
import argparse
import time
import cv2
import os
import numpy as np

Path_cfg  = "Data/yolov3_custom.cfg"
Path_weight = "Data/yolov3.weights"
Path_classes = "Data/class.names"
confidence = 0.5

def Auswertung_Yolo(Path_image):
    image = cv2.imread(Path_image)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # read class names from text file
    classes = None
    with open(Path_classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # read pre-trained model and config file
    net = cv2.dnn.readNet(Path_weight, Path_cfg)

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    return boxes, confidences, indices, classes, image, class_ids

def CutTable(image, indices, boxes, confidences, class_ids, classes):
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = round(box[0])
        y = round(box[1])
        w = round(box[2])
        h = round(box[3])

        newImage = image[y:y+h, x:x+w]
        label = str(classes[class_ids[i]])
        
        if label == "table": 
            return newImage
    pass

def Paint_Yolo(image, indices, boxes, confidences, class_ids, classes):
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes)
        
    # save output image to disk
    return image

# ------------------------------------------------------------------------------------

def get_output_layers(net):  
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ------------------------------------------------------------------------------------
# Functions for the API

def getPaintedImage(Path_image):
    boxes, confidences, indices, classes, image, class_ids  = Auswertung_Yolo(Path_image)
    image = Paint_Yolo(image, indices, boxes, confidences, class_ids, classes)
    p = cv2.imwrite("result.jpg", image)
    print(p)
    return "result.jpg"

def getTable(Path_image):
    boxes, confidences, indices, classes, image, class_ids = Auswertung_Yolo(Path_image)
    image = CutTable(image, indices, boxes, confidences, class_ids, classes)

    try:
        p = cv2.imwrite("table.jpg", image)
    except Exception:
        return 0
    print(p)
    return "table.jpg"

def getBoxes(Path_image):
    boxes, confidences, indices, classes, image, class_ids  = Auswertung_Yolo(Path_image)
    
    array = []
    print (indices)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = round(box[0])
        y = round(box[1])
        w = round(box[2])
        h = round(box[3])

        label = str(classes[class_ids[i]])

        temp = {}
        i = i[0]
        temp['class'] = label
        temp['box'] = [x,y,w,h]
        array.append(temp)
    
    return array