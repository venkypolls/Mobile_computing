import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib as plt

def get_data():
    types = ["book","car","gift","movie","sell","total"]
    poseObjects = []

    for i in range(6):
        files = os.listdir('/Users/venkateshpolepally/education/mobile computing/Thursday_Assignment_2_json/'+types[i])
        file_name = []
        for file in files:
            file_name.append(os.path.join('/Users/venkateshpolepally/education/mobile computing/Thursday_Assignment_2_json/',types[i], file))


        for file in file_name:
            with open(file, encoding="utf-8") as json_file:
                # print(file)
                data = json.load(json_file)
                p = PosenetPoints(types[i])
                p.parse_data(data)
                poseObjects.append(p)

    return poseObjects





class PosenetPoints:


    def __init__(self, name):
        self.type = name
        self.leftShoulder_x = []
        self.rightShoulder_x = []
        self.leftElbow_x = []
        self.rightElbow_x = []
        self.leftWrist_x = []
        self.rightWrist_x = []
        self.leftShoulder_y = []
        self.rightShoulder_y = []
        self.leftElbow_y = []
        self.rightElbow_y = []
        self.leftWrist_y = []
        self.rightWrist_y = []
        self.type = ""

    def parse_data(self,data):
        for pose in data:
            # print(pose)
            key = pose['keypoints']
            for point in key:
                if(point['part'] == "leftShoulder"):
                    self.leftShoulder_x.append(round(point['position']['x']))
                    self.leftShoulder_y.append(round(point['position']['y']))
                if (point['part'] == "rightShoulder"):
                    self.rightShoulder_x.append(round(point['position']['x']))
                    self.rightShoulder_y.append(round(point['position']['y']))
                if (point['part'] == "leftElbow"):
                    self.leftElbow_x.append(round(point['position']['x']))
                    self.leftElbow_y.append(round(point['position']['y']))
                if (point['part'] == "rightElbow"):
                    self.rightElbow_x.append(round(point['position']['x']))
                    self.rightElbow_y.append(round(point['position']['y']))
                if (point['part'] == "leftWrist"):
                    self.leftWrist_x.append(round(point['position']['x']))
                    self.leftWrist_y.append(round(point['position']['y']))
                if (point['part'] == "rightWrist"):
                    self.rightWrist_x.append(round(point['position']['x']))
                    self.rightWrist_y.append(round(point['position']['y']))





objects = get_data()
mainData = []
classData = []
for object in objects:
    temp = []
    temp = object.leftElbow_x
    temp = object.leftElbow_y
    temp+=object.rightElbow_x
    temp += object.rightElbow_y
    temp+=object.leftShoulder_x
    temp += object.leftShoulder_y
    temp+=object.rightShoulder_x
    temp += object.rightShoulder_y
    temp+=object.leftWrist_x
    temp += object.leftWrist_y
    temp+=object.rightWrist_x
    temp += object.rightWrist_y
    mainData.append(temp)
    classData.append(object.type)

# for i in range(len(mainData)):
#     print(len(mainData[i]))





