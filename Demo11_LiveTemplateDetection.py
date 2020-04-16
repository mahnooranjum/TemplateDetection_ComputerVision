##==============================================================================
##   Demo By: Mahnoor Anjum
##   Date: 31/03/2019
##   Codes inspired by:
##   Rajeev Ratab
##   Official Documentation
##==============================================================================
import cv2
import numpy as np

def detector_sift(input, template_gray):
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray,None)
    keypointsT, descriptorsT =sift.detectAndCompute(template_gray,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 20)
    search_params = dict(checks = 200)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors, descriptorsT, k=2)
    good_match = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_match.append(m)
            
    return good_match


image_template = cv2.imread('imgs/demo11b.jpg',0)

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('Demo11.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    
    x0 = int(width*0.2)
    y0 = int(height*0.1)
    x1 = int(width*0.8)
    y1 = int(height*0.6)
#    
    cv2.rectangle(frame, (x0,y0),(x1,y1), (0,0,255),4)
    input = frame[y0:y1, x0:x1]
#    
    frame = cv2.flip(frame,1)
    matches = detector_sift(input, image_template)

    cv2.putText(frame, str(len(matches)), (int(width/10), int(height/3)), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
    threshold = 10
    if (len(matches)>threshold):
        cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0),4)
        cv2.putText(frame, "OBJECT FOUND", (int(width/10), int(height*0.9)), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
        
    cv2.imshow('Object Detector', frame)
    out.write((frame))

    if cv2.waitKey(1) == 32: 
        break

out.release()
cap.release()
cv2.destroyAllWindows()
