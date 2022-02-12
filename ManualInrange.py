import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import copy

img_path = './BananaForSegment_12Feb2022/'

alpha_value = .7 # 0.1-1


diff_thres = np.array([50,50,50]) # [70,110,40]  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class cvRect:
    def __init__(self, xywh):
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
        self.xmin = self.x
        self.ymin = self.y
        self.xmax = self.x + self.w
        self.ymax = self.y + self.h
    def area(self):
        return self.w * self.h
    def tl(self):
        return [self.x,self.y]
    def br(self):
        return [self.x+self.w,self.y+self.h]
    def center(self):
        return [self.x+(self.w//2),self.y+(self.h//2)]
    def get_xywh(self):
        return  [self.x,self.y,self.w,self.h]


def pointBG(src_img):
    img = src_img.copy()
    mean = np.array([50,50,50])
    low_H = np.int16(np.clip(mean[0] - diff_thres[0],0,255)).item()
    low_S = np.int16(np.clip(mean[1] - diff_thres[1],0,255)).item()
    low_V = np.int16(np.clip(mean[2] - diff_thres[2],0,255)).item()
    high_H = np.int16(np.clip(mean[0] + diff_thres[0],0,255)).item()
    high_S = np.int16(np.clip(mean[1] + diff_thres[1],0,255)).item()
    high_V = np.int16(np.clip(mean[2] + diff_thres[2],0,255)).item()
    lowerb = (low_H, low_S, low_V)
    upperb = (high_H, high_S, high_V)
    print(f"lowerb{lowerb}")
    print(f"upperb{upperb}")
    inrange_img = cv.inRange(img, lowerb, upperb)
    return inrange_img

def locateBG(inrange_img):
    kernel_ELLIPSE_2x2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
    _,thres_img = cv.threshold(inrange_img,127,255,cv.THRESH_BINARY_INV)
    thres_img = cv.erode(thres_img,kernel_ELLIPSE_2x2,iterations=1)
    contours, _ = cv.findContours(thres_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    OBJS_RECT = []
    OBJS_CENTER = []
    IMG_CENTER = [inrange_img.shape[1]//2,inrange_img.shape[0]//2] # [x_center, y_center]
    OBJS_DIFF_CENTER = []
    # reject small contour (noise)
    for i,cnt in enumerate(contours):
        x,y,w,h = cv.boundingRect(cnt)
        if(w>=50 or h>=50):
            OBJS_RECT.append([x,y,w,h]) # [x,y,w,h]
            obj_center = [ (x+(w//2)) , (y+(h//2)) ]
            OBJS_CENTER.append(obj_center) # [x_obj_center, y_obj_center]
            OBJS_DIFF_CENTER.append(abs(IMG_CENTER[0]-obj_center[0])+abs(IMG_CENTER[1]-obj_center[1])) # diff = [ X_IMG_CENTER - x_obj_center, Y_IMG_CENTER - y_obj_center]
            '''print("_________________________________________________")
            print(f"IMG center{IMG_CENTER}")
            print(f"OBJ center{obj_center}")
            print(f"x,y,w,h obj{[x,y,w,h]}")
            print(f"OBJS_DIFF_CENTER center{abs(IMG_CENTER[0]-obj_center[0])+abs(IMG_CENTER[1]-obj_center[1])}")'''
    # find middlest RECT
    middlest_RECT = [inrange_img.shape[1]//4,inrange_img.shape[0]//4,IMG_CENTER[0],IMG_CENTER[1]] #in case if not found RECT -> be use default
    if(len(OBJS_RECT)==1): # if have only one RECT
        middlest_RECT = OBJS_RECT[0]
    elif(len(OBJS_RECT)>1): # find middlest RECT
        tmp_min_middle = OBJS_DIFF_CENTER[0]
        for i,val in enumerate(OBJS_DIFF_CENTER):
            #print(f"{val}",end=',')
            if(val <= tmp_min_middle):
                tmp_min_middle = val
                middlest_RECT = OBJS_RECT[i]
        #print(f"select {tmp_min_middle}")
    #cv.rectangle(thres_img,(middlest_RECT[0],middlest_RECT[1]),(middlest_RECT[0]+middlest_RECT[2],middlest_RECT[1]+middlest_RECT[3]),(255,255,255),2)
    # if middlest obj has area >90% (false positive) all white image
    allArea = inrange_img.shape[1]*inrange_img.shape[0] # w * h
    objArea = middlest_RECT[2] * middlest_RECT[2] # w * h
    if(objArea/allArea>0.9):
        [inrange_img.shape[1]//4,inrange_img.shape[0]//4,IMG_CENTER[0],IMG_CENTER[1]] #in case all white image -> be use default
    return thres_img,middlest_RECT
    


def main():
    global img_path,alpha_value
    divideHeight = 1
    divideWidth = 1
    list_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    del_lists = []
    for i,fname in enumerate(list_files):
        last = len(fname) - 1
        file_ext = fname[-3:]
        if(file_ext!='png' and file_ext!='jpg'): # and file_ext!='JPG'
            del_lists.append(fname) # mark as delete
            #print(file_ext)
    for val in del_lists:
        list_files.remove(val)
            
    print(f"After del other file ext:{list_files}")
    imgs = []
    #  ,
    # Read images from lists
    for i,fname in enumerate(list_files):
        tmp_img = cv.imread(img_path+fname)
        h = tmp_img.shape[0]//divideHeight
        w = tmp_img.shape[1]//divideWidth
        imgs.append(cv.resize(tmp_img,(w,h)))
    # Set low contrast
    lowct_imgs = []
    for i,img in enumerate(imgs):
        lowct_imgs.append(cv.convertScaleAbs(img,alpha=alpha_value, beta=0))
    # Convert to HSV
    RGB_imgs = []
    for i,img in enumerate(lowct_imgs):
        RGB_imgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # Call func pointBG
    inrange_imgs = []
    for i,img in enumerate(RGB_imgs):
        tmp_point_img = pointBG(img)
        inrange_imgs.append(tmp_point_img)

    locateBG_imgs = []
    locateBG_xywh = []
    for i,img in enumerate(inrange_imgs):
        ret_img,ret_xywh = locateBG(img)
        locateBG_imgs.append(ret_img)
        locateBG_xywh.append(ret_xywh)
        xywh = locateBG_xywh[i]
        tl_point = (xywh[0],xywh[1])
        br_point = (xywh[0]+xywh[2],xywh[1]+xywh[3])
        #cv.rectangle(imgs[i],tl_point,br_point,(0,255,0),2) # (x,y),(x+w,y+h)
        cv.imwrite(img_path+"/seg/"+list_files[i]+"_segment.jpg",imgs[i])
        cv.imwrite(img_path+"/seg/"+list_files[i]+"_segment.png",locateBG_imgs[i])
    
    # Display by plt
    plt_index = 1
    num_imgs = len(imgs)
    col = 4
    plt.rcParams["figure.figsize"] = (30,40)
    for i in range(num_imgs):
        if i==1 :
            plt.subplot(num_imgs,col,plt_index),plt.imshow(imgs[i]),plt.title("Original"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(lowct_imgs[i]),plt.title("LowContrast"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(RGB_imgs[i]),plt.title("HSV"),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(inrange_imgs[i]),plt.title("pointBG"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(locateBG_imgs[i]),plt.title("Eroded"),plt.xticks([]),plt.yticks([])
            plt_index+=1
        else :
            plt.subplot(num_imgs,col,plt_index),plt.imshow(imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(lowct_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(RGB_imgs[i]),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(inrange_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(locateBG_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
    plt.show()




if __name__ == "__main__":
    main()