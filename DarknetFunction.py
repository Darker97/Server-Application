import cv2
import numpy as np
import json

net = cv2.dnn.readNet("Data/yolov3.weights", "Data/yolov3_custom.cfg")

def Transform(LinkToFile):
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #loading image
    img = cv2.imread(LinkToFile)
    img = cv2.resize(img,None,fx=0.4,fy=0.3)
    height,width,channels = img.shape


    blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)

    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #onject detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
            
                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                
                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    Temp_Array = []

    for i in range(len(boxes)-1):
        temp = {}
        temp['box'] = boxes[i]
        temp['confidece'] = confidence[i]
        temp['class'] = class_ids[i]
        Temp_Array.append(temp)

    return json.dumps(Temp_Array)

