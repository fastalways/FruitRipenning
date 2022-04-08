import numpy as np
import time
import cv2
from time import strftime
from tkinter import *
from tkinter import filedialog

#-------------------------------------------------------------------
diffThreshold = 750
#-------------------------------------------------------------------


cap = cv2.VideoCapture(0)

def renderHistogram(colorImage):
    # Cal histogram
    bgr_planes = cv2.split(colorImage)
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX) # normalize -> 0 - 1
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX) # normalize -> 0 - 1
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX) # normalize -> 0 - 1
    # plot histogram
    for i in range(1, histSize):
        cv2.line(hist_img, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(b_hist[i]) ),
                ( 255, 0, 0), thickness=2)
        cv2.line(hist_img, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(g_hist[i]) ),
                ( 0, 255, 0), thickness=2)
        cv2.line(hist_img, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(r_hist[i]) ),
                ( 0, 0, 255), thickness=2)
    return hist_img,[r_hist,g_hist,b_hist]

filenameCalibratedImage = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select a File",
                                          filetypes = (("Calibrated Image",
                                                        "*.png*"),
                                                       ("all files",
                                                        "*.*")))
print(f"Loaded Calibrated Image:{filenameCalibratedImage}")
CalibratedImage = cv2.imread(filenameCalibratedImage)

CalibratedImage_HistImage,CalibratedImage_Hist = renderHistogram(CalibratedImage)

#relay
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(4,GPIO.OUT)

GPIO.output(4, True)

#----- Load Calibrate Image (GreenPaper)


print("Press c to capture calibration image")
print("Press e to exit")

def laplaceDetector(img,kernel_size = 3,ddepth = cv2.CV_16S):
    src = img.copy()
    src = cv2.GaussianBlur(src, (3, 3), 0)
    dst = cv2.Laplacian(src, ddepth, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    return abs_dst

def cornerDetector(img,show_result=False):
    _,thres_img = cv2.threshold(img,30,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thres_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_area_id = -1
    for id,contour in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(contour)
        area = w*h
        if(area>max_area):
            max_area = area
            max_area_id = id
    ShowContour = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
    cv2.drawContours(ShowContour,contours,max_area_id,(0,0,255),3)
    approx_points = cv2.approxPolyDP(contours[max_area_id],200,True)
    four_points = []
    for i,values in enumerate(approx_points):
        four_points.append(values[0]) # add conner
    if(show_result):
        for i,values in enumerate(four_points):
            cv2.circle(ShowContour,values,5,(255,0,255),2)
        cv2.imshow("thres_img",thres_img)
        cv2.imshow("SHOWMaxCountour",ShowContour)
    return four_points

def doWarpPerspective(img,four_points):
    '''https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/'''
    '''https://theailearner.com/tag/cv2-getperspectivetransform/'''
    # order points -> top-left, top-right, bottom-right, bottom-left
    pts = np.array(four_points)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

warp_frame = None

cv2.namedWindow("Realtime_Camera",cv2.WINDOW_NORMAL)
cv2.namedWindow("Current_Histogram",cv2.WINDOW_NORMAL)
cv2.namedWindow("Calibrated_Reference",cv2.WINDOW_NORMAL)
cv2.namedWindow("Realtime_CalibratePaper",cv2.WINDOW_NORMAL)

while (True):
   
    ret , frame = cap.read()
    
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_laplace = laplaceDetector(frame_gray)
    connerFourPoints = cornerDetector(frame_laplace)
    if(len(connerFourPoints)==4): # if found valid conner
        warp_frame = doWarpPerspective(frame,connerFourPoints)
        cv2.imshow("Realtime_CalibratePaper",warp_frame)
        warp_frame_HistImage,warp_frame_Hist = renderHistogram(warp_frame)
        sum_r_square = np.sum(np.square(warp_frame_Hist[0] - CalibratedImage_Hist[0])) # Red
        sum_g_square = np.sum(np.square(warp_frame_Hist[1] - CalibratedImage_Hist[1])) # Green
        sum_b_square = np.sum(np.square(warp_frame_Hist[2] - CalibratedImage_Hist[2])) # Blue
        diffCalibatedValue = np.sqrt(sum_r_square + sum_g_square + sum_b_square)
        #print(f"DiffCalibatedValue={diffCalibatedValue}")
        cv2.putText(frame,"diff:"+str(diffCalibatedValue),(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
        if(diffCalibatedValue>=0 and diffCalibatedValue<=diffThreshold): # OK
            cv2.putText(frame,"Calibration -> OK",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,0),2)# OK
        else :
            cv2.putText(frame,"Calibration -> not passed",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,255),2)# OK
        cv2.imshow("Current_Histogram",warp_frame_HistImage)
    else :
        cv2.putText(frame,"Not found Paper",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)# Not found Paper
        cv2.putText(frame,"(more darker/lighter)",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)# Not found Paper

    cv2.imshow("Realtime_Camera",frame)
    
    cv2.imshow("Calibrated_Reference",CalibratedImage_HistImage)
    #cv2.imshow("frame_laplace",frame_laplace)
    


    key = cv2.waitKey(1)

    
    if key == ord("c"):
        timefile = strftime("%d_%m_%Y_%H-%M-%S")
        if(warp_frame.all()!=None):
            calibFileName = 'CalibratedImage - '+timefile + '.png'
            cv2.imwrite( calibFileName , warp_frame) # png for lossless/compression
            cv2.imwrite( timefile + '.jpg', frame)
            print("Saved -> "+calibFileName)
    if key & 0xFF == ord("e"):
        GPIO.output(4, False)
        break
    

cap.release()
cv2.destroyAllWindows

