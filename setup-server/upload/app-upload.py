# from crypt import methods
import imp
from lib2to3.pgen2 import token
from sqlite3 import Time
# import string
from time import time
from tkinter import Image
from unittest import result
from unittest.mock import patch
from urllib import response
from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
from flask_pymongo import PyMongo 
from bson.objectid import ObjectId

#line Notify  import
import requests, urllib.parse
import io
from PIL import Image

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv
import math

token = "qb3ndQYktV7wci2Y5laK1FhxKaRUUmcR9VQnnh5AgPQ"
url = 'https://notify-api.line.me/api/notify'
HEADERS = {'Authorization': 'Bearer ' + token}



loaded_model = tf.keras.models.load_model(
    '../'+'FullImage-Incepv3-epoch0800-val_acc0.94.pb')
# D:\year4\project\Flask\FlaskAPIuplodeFile\FullImage-Incepv3-epoch0400-val_acc0.91.pb


def bsegment_wider_range(img):
    alpha_value = .6
    lowct_img = cv.convertScaleAbs(img, alpha=alpha_value, beta=0)
    # rgb_img=cv.cvtColor(lowct_img, cv.COLOR_BGR2RGB)
    lowerb = (0, 0, 0)
    upperb = (80, 80, 90)
    inrange_img = cv.inRange(lowct_img, lowerb, upperb)
    invert_banana = 255-inrange_img
    bgr_banana = cv.cvtColor(invert_banana, cv.COLOR_GRAY2BGR)
    segmented_bgr_banana = np.bitwise_and(bgr_banana, img)
    return segmented_bgr_banana


def cropOutBlack(seg_bgr_banana, bgr_banana):
    imgray = cv.cvtColor(seg_bgr_banana, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 10, 255, 0)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Find max Contour
    max_area = 0
    max_idx = -1
    (im_h, im_w) = seg_bgr_banana.shape[:2]
    img_area = im_h*im_w
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)
        if((w*h) >= max_area):
            max_area = w*h
            max_idx = idx
    if(max_idx == -1):
        return bgr_banana
    (x, y, w, h) = cv.boundingRect(contours[max_idx])
    for idx, contour in enumerate(contours):
        (xSub, ySub, wSub, hSub) = cv.boundingRect(contour)
        if (wSub*hSub) >= (img_area*0.015):
            # expand area by sub img
            if(x > xSub):
                x = xSub
                w += abs(xSub-x)
            if(y > ySub):
                y = ySub
                h += abs(ySub-y)
            if(x+w < xSub+wSub):
                w += ((xSub+wSub)-(x+w))
            if(y+h < ySub+hSub):
                h += ((ySub+hSub)-(y+h))
    return bgr_banana[y:y+h, x:x+w]


def predictRipenessLV(banana_path):
    img_height = 299
    img_width = 299
    im = cv.imread(banana_path)
    seg_banana = bsegment_wider_range(im)
    only_banana = cropOutBlack(seg_banana, im)
    only_banana = cv.resize(
        only_banana, (img_height, img_width), interpolation=cv.INTER_CUBIC)
    only_banana = cv.cvtColor(only_banana, cv.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(only_banana, dtype=tf.float32)
    banana_img = tf.cast(img_tensor, tf.float32) / 255.0
    result = loaded_model.predict(banana_img[tf.newaxis, ...])
    class_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    max_cf_class = np.argmax(result)
    return int(class_names[max_cf_class])

def filterOutBlack4(seg_bgr_banana):
  
    (h,w)=seg_bgr_banana.shape[:2]
    arr=np.reshape(seg_bgr_banana,(h*w,3))
 
    del_arr = np.delete(arr, np.where(np.all(arr == np.array([0, 0, 0],dtype=np.uint8), axis=1)), axis=0)
  
    size_2d=int(math.sqrt(del_arr.shape[0]))
    result=np.reshape(del_arr[0:(size_2d*size_2d)],(size_2d,size_2d,3))
    return result

def bsegment(img):
    alpha_value=.7
    lowct_img=cv.convertScaleAbs(img,alpha=alpha_value, beta=0)
    #rgb_img=cv.cvtColor(lowct_img, cv.COLOR_BGR2RGB)
    lowerb = (0, 0, 0)
    upperb = (80, 80, 90)
    inrange_img = cv.inRange(lowct_img, lowerb, upperb)
    invert_banana = 255-inrange_img
    bgr_banana = cv.cvtColor(invert_banana,cv.COLOR_GRAY2BGR)
    segmented_bgr_banana=np.bitwise_and(bgr_banana,img)
    return segmented_bgr_banana

def calRipenessScore(Path_img):
    img = cv.imread(Path_img)
    mean, std = cv.meanStdDev(filterOutBlack4(bsegment(img)))
    return ((mean[0]*1.9) + (mean[1]*1.6) + (mean[2]*7.4))[0].item()


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/DBprojectfinal"
mongo = PyMongo(app)
app.secret_key = "caircocoders-ednalan"

UPLOAD_FOLDER = '../../../WebsiteProject/banana-website/build/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


'''@app.route('/')
def main():
    resp = jsonify({'message': ' successfully '})
    resp.status_code = 200
    return resp'''
    
@app.route('/')
def show_index():
    str =  """<!DOCTYPE html>
<html>
<head>
<title>BANANA-UPLOAD-API</title>
</head>
<body>

<h1>Welcome to BANANA-UPLOAD-API</h1>
<p>This is BANANA-UPLOAD-API</p>

</body>
</html>"""
    return str


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    
    files = request.files.getlist('files[]')
    
    
   
    tem = request.form['tem']
    hum = request.form['hum']
    date = request.form['date']
    time = request.form['time']
    namebox =  request.form['namebox']

    # ID_Box = jsonify(mongo.db.Boxs.find({{'Status':"rippening"}}))
    # print("id_box -> " + ID_Box).
    # id_box = str(ObjectId(ID_Box))
    id_Boxs = mongo.db.Boxs.aggregate([{ "$match" : { "Status" : "rippening" , "Name_Box" : namebox } } , { "$project" : { "_id" : 1 , "Alert_LV": 1 } }])
     # print(ID_Boxs)
    for id_Box in id_Boxs:
        print(type(id_Box['_id']))
        print(type(id_Box['Alert_LV']))

    

    errors = {}
    success = False

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True

            Path_Image = UPLOAD_FOLDER + '/' + filename
            LVbanana = -1
            RipenessScore = -1
            try:
                LVbanana = predictRipenessLV(Path_Image)
                RipenessScore = calRipenessScore(Path_Image)
            except:
                LVbanana = -1
                RipenessScore = -1

           


            RipenessScore2f = float("{:.2f}".format(RipenessScore))
            mongo.db.DataLog.insert_one(
                {'id_Box': ObjectId((id_Box['_id'])), 'image': '../uploads/'+filename, 'tem': tem,
                    'hum': hum, 'Date': date, 'time': time, 'LV': LVbanana, 'score' :  RipenessScore2f }
            )
            print(type(namebox))


            # print(Fscore)
            
            print ("Alert Leavel:"+str(id_Box['Alert_LV']))
            print ("Lavel Banana:"+str(LVbanana))
            print("                    ")

            if  LVbanana  >= (id_Box['Alert_LV']) :
                msg = "👋ผลไม้ของคุณไกล้สุกเเล้ว ❗""\n""🍌 คะแนนความสุกของกล้วย :"+ str(RipenessScore2f) + "\n" "🍌 ระดับความสุก :"+str(LVbanana)

                img = Image.open(Path_Image)
                img.load()
                myimg = np.array(img)

                f = io.BytesIO()
                Image.fromarray(myimg).save(f, 'png')
                data = f.getvalue()

                response = requests.post(url,headers=HEADERS,params={"message": msg}
                                    ,files={"imageFile": data})
                print(response)

                


# D:\year4\project\Flask\FlaskAPIuplodeFile\app\static\uploads\
        else:
            errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({'message': 'Files successfully uploaded'})
        resp.status_code = 201
        print(filename)
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5002,debug=False)

    # note run IP
    # flask run -h localhost -p 3000
    # python app.py
