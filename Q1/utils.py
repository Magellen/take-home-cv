import numpy as np
import cv2
import glob
import os

def v2img(video='traffic.mp4'):
   if not os.path.exists('imgs'):
       os.makedirs('imgs')
   vidcap = cv2.VideoCapture(video)
   def getFrame(sec):
       vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
       hasFrames,image = vidcap.read()
       if hasFrames:
           cv2.imwrite("imgs/"+"{:04d}".format(count)+".jpg", image)
       return hasFrames
   sec = 0
   frameRate = 0.1
   count=1
   success = getFrame(sec)
   while success:
       count = count + 1
       sec = sec + frameRate
       sec = round(sec, 2)
       success = getFrame(sec)

def createMask():
    imgPath = glob.glob("imgs/*.jpg")
    imgPath.sort()
    imgs = [cv2.imread(path) for path in imgPath]
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=1000)
    for img in imgs:
        fg_mask = bs.apply(img)
    h,w,c = imgs[0].shape
    heat = np.zeros((h,w))
    for img in imgs:
        fg_mask = bs.apply(img)
        th = cv2.threshold(fg_mask.copy(), 253, 255, cv2.THRESH_BINARY)[1]          
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)), iterations=2)        
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)), iterations=2)
        heat += dilated 
    heat = heat.astype(np.uint8)
    th = cv2.threshold(heat, 128, 255, cv2.THRESH_BINARY)[1]
    heat2 = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8)), iterations=5)
    cv2.imwrite("mask.jpg",heat2)

def createHeatMap():
    imgPath = glob.glob("imgs/*.jpg")
    imgPath.sort()
    imgs = []
    for path in imgPath:
        img = cv2.imread(path)
        h,w,c = img.shape
        img = cv2.resize(img,(w*2,h*2))
        imgs.append(img)
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=1000)
    for img in imgs:
        fg_mask = bs.apply(img)
    heat = np.zeros((1440,2560))
    for img in imgs:
        fg_mask = bs.apply(img)
        th = cv2.threshold(fg_mask.copy(), 253, 255, cv2.THRESH_BINARY)[1]         
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)), iterations=2)        
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)), iterations=2) 
        heat += dilated
    heat2 = heat/heat.max()*255
    heat2 = heat2.astype(np.uint8)
    heatmap = cv2.applyColorMap(heat2, cv2.COLORMAP_HOT)
    heatmap = cv2.resize(heatmap,(w,h))
    cv2.imwrite("heatmap.jpg",heatmap)

def run(staticCar = 60, speedLimit = 2):
    cap = cv2.VideoCapture('traffic.mp4')
    if not os.path.isfile("mask.jpg"):
        createMask()
    mask = cv2.imread("mask.jpg",cv2.IMREAD_GRAYSCALE)
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=1000)
    h,w = mask.shape
    pre = np.zeros((w,h))
    ret = True
    first = True
    while(ret):
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_and(img, mask)
        img = bs.apply(img)
        if first:
            pre = img
            first = False
            continue
        contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moving = 0
        for c in contours:                     
            x, y, w, h = cv2.boundingRect(c)                    
            area = cv2.contourArea(c)        
            if 100 < w*h < 300 and 1/3 < w/h < 3:                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                moving += 1
        img2 = cv2.bitwise_and(img,pre)
        img2 = cv2.bitwise_xor(img,img2)
        contours, hier = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        speeding = 0
        for c in contours:                     
            x, y, w, h = cv2.boundingRect(c)                    
            area = cv2.contourArea(c) 
            speed = np.sqrt(area)   
            if speed > speedLimit and w*h > 10:                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
                speeding += 1
        pre = img
        cv2.putText(frame,f'car number: {staticCar+moving}',(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
