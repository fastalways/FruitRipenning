import numpy as np
import time
import cv2
from time import strftime
cap = cv2.VideoCapture(0)
#### 1) เปิดไฟ ####
#relay
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(4,GPIO.OUT)

GPIO.output(4, True)

#### 2) พิมพ์ข้อความ กด c เพื่อแคปภาพ กด e เพื่อออก ####
print("Press c to capture calibration image")
print("Press e to exit")

#### ฟังก์ชันหาขอบภาพ ####
def laplaceDetector(img,kernel_size = 3,ddepth = cv2.CV_16S):
    src = img.copy()
    src = cv2.GaussianBlur(src, (3, 3), 0)
    dst = cv2.Laplacian(src, ddepth, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    return abs_dst

#### ฟังก์ชันหามุมของภาพ ####
def cornerDetector(img,show_result=False):
    #### แปลงเป็นขาวดำ ####
    _,thres_img = cv2.threshold(img,30,255,cv2.THRESH_BINARY)
    #cv2.imshow("thres_img",thres_img) 
    #### หาcontour ####
    contours, hierarchy = cv2.findContours(thres_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #### วนหา contourที่มีขนาดใหญ่ที่สุด ####
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

#### ฟังก์ชันปรับภาพให้เป็นสี่เหลี่ยม ####
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



while (True):
    #### 3) รับภาพจากกล้อง เก็บในตัวแปร frame ####
    ret , frame = cap.read()
    #### 4) แปลงเป็นภาพขาวเทา ####
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #### 5) หาขอบของภาพกระดาษสีเขียว ####
    frame_laplace = laplaceDetector(frame_gray)
    #### 6) เรียกฟังก์ชันหามุมของกระดาษ ####
    connerFourPoints = cornerDetector(frame_laplace,show_result=True)
    #### 7) ถ้าเจอจุด4จุดแสดงว่าหากระดาษเจอ ให้ทำการทำภาพกระดาษที่บิดงอกลับมาตรงเป็นสี่เหลี่ยม ####
    if(len(connerFourPoints)==4): # if found valid conner
        #### ทำการทำภาพกระดาษที่บิดงอกลับมาตรงเป็นสี่เหลี่ยม ####
        warp_frame = doWarpPerspective(frame,connerFourPoints)
        cv2.imshow("Warped",warp_frame)
    cv2.imshow("Output",frame)
    cv2.imshow("frame_laplace",frame_laplace)

    key = cv2.waitKey(1)

    #### 8) ถ้ากด ปุ่ม c ให้บันทึกภาพ ####
    if key == ord("c"):
        timefile = strftime("%d_%m_%Y_%H-%M-%S")
        if(warp_frame.all()!=None):
            calibFileName = 'CalibratedImage-'+timefile + '.png'
            cv2.imwrite( calibFileName , warp_frame) # png for lossless/compression
            cv2.imwrite( timefile + '.jpg', frame)
            print("Saved -> "+calibFileName)
    #### 8) ถ้ากด ปุ่ม e ให้ออกจากโปรแกรมโดยการbreakออกจากloop ####
    if key & 0xFF == ord("e"):
        GPIO.output(4, False)
        break
    

cap.release()
cv2.destroyAllWindows

