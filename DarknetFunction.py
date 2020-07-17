
import argparse
import time
import cv2
import os
import numpy as np

Path_cfg  = "Data/yolov3_custom.cfg"
Path_weight = "Data/YoloV3.weights"
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
    Path_image = FindeDocument(Path_image)

    boxes, confidences, indices, classes, image, class_ids  = Auswertung_Yolo(Path_image)
    image = Paint_Yolo(image, indices, boxes, confidences, class_ids, classes)
    p = cv2.imwrite("result.jpg", image)
    print(p)
    return "result.jpg"

def getTable(Path_image):
    Path_image = FindeDocument(Path_image)

    boxes, confidences, indices, classes, image, class_ids = Auswertung_Yolo(Path_image)
    image = CutTable(image, indices, boxes, confidences, class_ids, classes)

    try:
        p = cv2.imwrite("table.jpg", image)
    except Exception:
        return 0
    print(p)
    return "table.jpg"

def getBoxes(Path_image):
    Path_image = FindeDocument(Path_image)
    
    boxes, confidences, indices, classes, image, class_ids  = Auswertung_Yolo(Path_image)
    
    array = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = round(box[0])
        y = round(box[1])
        w = round(box[2])
        h = round(box[3])

        label = str(classes[class_ids[i]])

        temp = {}
        temp['class'] = label
        temp['box'] = [x,y,w,h]
        array.append(temp)
    
    return array

# ------------------------------------------------------------------------------------
# Functionen zum verbessern des Inputs
def FindeDocument(image_path):
    image = cv2.imread(image_path)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

    img = cv2.medianBlur(img, 11)
    
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    edges = cv2.Canny(img, 200, 250)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding contour of biggest rectangle
    # Otherwise return corners of original image
    # Don't forget on our 5px border!
    height = edges.shape[0]
    width = edges.shape[1]
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    # Page fill at least half of image, then saving max area found
    maxAreaFound = MAX_COUNTOUR_AREA * 0.5

    # Saving page contour
    pageContour = np.array([[5, 5], [5, height-5], [width-5, height-5], [width-5, 5]])

    # Go through all contours
    for cnt in contours:
        # Simplify contour
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        # Page area must be bigger than maxAreaFound 
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):

            maxAreaFound = cv2.contourArea(approx)
            pageContour = approx

    # Result in pageConoutr (numpy array of 4 points):

    def fourCornersSort(pts):
        """ Sort corners: top-left, bot-left, bot-right, top-right """
        # Difference and sum of x and y value
        # Inspired by http://www.pyimagesearch.com
        diff = np.diff(pts, axis=1)
        summ = pts.sum(axis=1)
        
        # Top-left point has smallest sum...
        # np.argmin() returns INDEX of min
        return np.array([pts[np.argmin(summ)],
                        pts[np.argmax(diff)],
                        pts[np.argmax(summ)],
                        pts[np.argmin(diff)]])


    def contourOffset(cnt, offset):
        """ Offset contour, by 5px border """
        # Matrix addition
        cnt += offset
        
        # if value < 0 => replace it by 0
        cnt[cnt < 0] = 0
        return cnt


    # Sort and offset corners
    pageContour = fourCornersSort(pageContour[:, 0])
    pageContour = contourOffset(pageContour, (-5, -5))

    # Recalculate to original scale - start Points
    sPoints = pageContour.dot(image.shape[0] / 800)
    
    # Using Euclidean distance
    # Calculate maximum height (maximal length of vertical edges) and width
    height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
                np.linalg.norm(sPoints[2] - sPoints[3]))
    width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
                np.linalg.norm(sPoints[3] - sPoints[0]))

    # Create target points
    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)

    # Wraping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints) 
    newImage = cv2.warpPerspective(image, M, (int(width), int(height)))

    cv2.imwrite("Zwischenergebnis.jpg", cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
    
    return "Zwischenergebnis.jpg"