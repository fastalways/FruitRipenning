# from crypt import methods
# from distutils.log import debug
# # from crypt import methods
# from crypt import methods
# from crypt import methods
import os
import imp
from time import time
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from flask_cors import CORS
from bson.objectid import ObjectId
from itsdangerous import json


from flask_jwt_extended import create_access_token
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager



app = Flask(__name__)
app.config['MONGO_URI']='mongodb://localhost:27017/DBprojectfinal'
mongo = PyMongo(app)

app.config["JWT_SECRET_KEY"] = "super-secret"  # Change this!
jwt = JWTManager(app)

CORS(app)


dbUsers = mongo.db.Users
dbBoxs = mongo.db.Boxs
dbDataLog = mongo.db.DataLog

@app.route('/')
def show_index():
    str =  """<!DOCTYPE html>
<html>
<head>
<title>BANANA-BACKEND-API</title>
</head>
<body>

<h1>Welcome to BANANA-BACKEND-API</h1>
<p>This is BANANA-BACKEND-API</p>

</body>
</html>"""
    return str

# -----------------------------------test-----------------------------------------------------------
# insertOne users
@app.route('/users', methods=['POST'])
def createUsers():
    id = dbUsers.insert_one({
        'User':request.json['User'],
        'Password': request.json['Password'],
        'Fname' : request.json['Fname'],
        'Lname' : request.json['Lname'],
        'Email' : request.json['Email']

    })

    # print(str(ObjectId(id.inserted_id)))
    return jsonify(str(ObjectId(id.inserted_id)))

# find users
@app.route('/users', methods=['GET'])
def getusers():
    users = []
    for doc in dbUsers.find():
        users.append({
            '_id': str(ObjectId(doc['_id'])),
            'User':doc['User'],
            'Password': doc['Password'],
            'Fname' : doc['Fname'],
            'Lname' : doc['Lname'],
            'Email' : doc['Email']
        })
    return jsonify(users)

# find_One user
@app.route('/user/<id>', methods=['GET'])
def getuser(id):
    user = dbUsers.find_one({'_id': ObjectId(id)})
    return jsonify({
        '_id': str(ObjectId(user['_id'])),
        'User':user['User'],
        'Password': user['Password'],
        'Fname' : user['Fname'],
        'Lname' : user['Lname'],
        'Email' : user['Email']
    })

#delete_one user
@app.route('/user/<id>', methods=['DELETE'])
def deleteUser(id):
    dbUsers.delete_one({'_id': ObjectId(id)})
    return jsonify({'msg': 'User deleted'})

# update_one user
@app.route('/user/<id>', methods=['PUT'])
def updateUser(id):
    dbUsers.update_one({'_id': ObjectId(id)}, {'$set': {
        'User':request.json['User'],
        'Password': request.json['Password'],
        'Fname' : request.json['Fname'],
        'Lname' : request.json['Lname'],
        'Email' : request.json['Email']
    }})
    return jsonify({'msg': 'User Updated'})

# ----------------------------------------------------------------------------------------------

# Create a route to authenticate your users and return JWTs. The
# create_access_token() function is used to actually generate the JWT.
@app.route("/token", methods=["POST"])
def create_token():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    if username != "test" or password != "test":
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

#insertOne boxs
@app.route('/box', methods=['POST'])
def createBox():

    id = dbBoxs.insert_one({
        'User_id': ObjectId('624343ff751be92383acb4a1'), #auten user _id --> User_id
        'Name_Box': request.json['Name_Box'],
        'Status' : 'rippening',
        'Date_Start' : request.json['Date_Start'],
        'Alert_LV': int(request.json['Alert_LV'])
        # ,
        # 'Alert_Score': int(request.json['Alert_Score'])
    })
    return jsonify(str(ObjectId(id.inserted_id)))


# find boxs rippening
@app.route('/boxsRP/<id>', methods=['GET'])
def getboxs_rippening(id):
    boxs = []
    for doc in dbBoxs.find({'User_id': ObjectId(id),'Status':"rippening"}):
        boxs.append({
            '_id':str(ObjectId(doc['_id'])),
            'User_id':str(ObjectId(doc['User_id'])),
            'Name_Box':doc['Name_Box'],
            'Status': doc['Status'],
            'Date_Start':doc['Date_Start'],
            'Alert_LV': doc['Alert_LV']
            # ,'Alert_Score': doc['Alert_Score']
        })
    return jsonify(boxs)


        
# find boxs All
@app.route('/boxsAll/<id>', methods=['GET'])
def getboxs_all(id):
    
    boxs = []
    for doc in dbBoxs.find({'User_id': ObjectId(id)}):
        boxs.append({
            '_id':str(ObjectId(doc['_id'])),
            'User_id':str(ObjectId(doc['User_id'])),
            'Name_Box':doc['Name_Box'],
            'Status': doc['Status'],
            'Date_Start':doc['Date_Start'],
            'Alert_LV': doc['Alert_LV'],
            # 'Alert_Score': doc['Alert_Score']
        })
    return jsonify(boxs)

#update box
@app.route('/boxEdit/<id>', methods=['PUT'])
def editbox(id):
    dbBoxs.update_one({'_id': ObjectId(id)}, {'$set': {
        'Name_Box':request.json['Name_Box'],
        'Status' : request.json['Status'],
        'Date_Start': request.json['Date_Start'],
        'Alert_LV' : request.json['Alert_LV'],
        # 'Alert_Score' : request.json['Alert_Score']
    }})
    return jsonify({'msg': 'boxs Update'})

#Delete_one box
@app.route('/boxsDel/<id>', methods=['DELETE'])
def delbox(id):
    dbBoxs.delete_one({'_id': ObjectId(id)})
    return jsonify({'msg': 'Box deleted'})

#find datalog ref --> id box
@app.route('/datalog/<id_box>', methods=['GET'])
def getdatalog(id_box):

    Fscore = mongo.db.DataLog.find({'id_Box':ObjectId(id_box)},{"score":1,"_id":0}).limit(1).sort("_id",1)
    for score in Fscore:
        print(type(score['score']))


    datalog = []
    for doc in dbDataLog.find({'id_Box': ObjectId(id_box)}):
        datalog.append({
            '_id':str(ObjectId(doc['_id'])),
            'id_Box':str(ObjectId(doc['id_Box'])),
            'image': doc['image'],
            'tem' : doc['tem'], 
            'hum' : doc['hum'],
            'Date':doc['Date'],
            'time' : doc['time'],
            'score': "%.2f" %(doc['score']-score['score']),
            'LV': doc['LV']
        })
    return jsonify(datalog)

#find datalog latest record
@app.route('/datalogLT/<id_box>', methods=['GET'])
def getdataLatest(id_box):

    Fscore = mongo.db.DataLog.find({'id_Box':ObjectId(id_box)},{"score":1,"_id":0}).limit(1).sort("_id",1)
    for score in Fscore:
        print(type(score['score']))

    datalogLT = []
    for doc in dbDataLog.find({'id_Box':ObjectId(id_box)}).limit(1).sort("_id",-1):
        datalogLT.append({
            '_id':str(ObjectId(doc['_id'])),
            'id_Box':str(ObjectId(doc['id_Box'])),
            'image': doc['image'],
            'tem' : doc['tem'],
            'hum' : doc['hum'],
            'Date':doc['Date'],
            'time' : doc['time'],
            'score': "%.2f" %(doc['score']-score['score']),
            'LV': doc['LV']
        })

    return jsonify(datalogLT)

#Delete datalog
@app.route('/datalogDel/<id>', methods=['DELETE'])
def deldatalog(id):
    dbDataLog.delete_one({'_id': ObjectId(id)})
    return jsonify({'msg': 'Datalog Deleted'})

if __name__ == "__main__":
    app.run(port=5001,debug=False)
