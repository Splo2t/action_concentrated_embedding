import pandas as pd
import numpy as np
import os
import threading 
from concurrent import futures
from scipy import signal

import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str    ,   default="../")
parser.add_argument('--output_path',    type=str    ,   default="../processed_data")

args = parser.parse_args()

projectPath = args.data_path

output_path = args.output_path

if not os.path.isdir(output_path):
    os.mkdir(path=output_path)

# video_path = projectPath + 'rawdata/원천데이터/1.mp4_TS001/1.mp4/1.Sangle/1.자연재난/COLDWAVE/1_1'
data_root_path = projectPath + '02_XML_TL/'
label_root_path = projectPath + '03_JSON_TrL/'

xml_path_dict = {}
json_path_dict = {}

for lv1 in os.listdir(data_root_path):          # tact / tact_hand / untact
    lv1 = os.path.join(data_root_path, lv1) 
    for lv2 in os.listdir(lv1):                 # 자연재난 / ...
        lv2 = os.path.join(lv1, lv2)
        for lv3 in os.listdir(lv2):             # COLDWAVE ... 
            lv3 = os.path.join(lv2, lv3)
            for lv4 in os.listdir(lv3):         # 1_1 / ...
                lv4 = os.path.join(lv3, lv4)
                for filename in os.listdir(lv4):
                    if filename[-5] == 'F':
                        xml_path_dict[filename[:-6]] = os.path.join(lv4, filename)
                    else:
                        xml_path_dict[filename[:-4]] = os.path.join(lv4, filename)

for lv1 in os.listdir(label_root_path):          # tact / untact
    lv1 = os.path.join(label_root_path, lv1) 
    for lv2 in os.listdir(lv1):                 # 자연재난 / ...
        lv2 = os.path.join(lv1, lv2)
        for lv3 in os.listdir(lv2):             # COLDWAVE ... 
            lv3 = os.path.join(lv2, lv3)
            for lv4 in os.listdir(lv3):         # 1_1 / ...
                lv4 = os.path.join(lv3, lv4)
                for filename in os.listdir(lv4):
                    json_path_dict[filename[:-5]] = os.path.join(lv4, filename)

id_list = list(xml_path_dict.keys())

keypoint_list = [
    list(range(0, 8))+list(range(15, 19)),
    list(range(0, 70)),
    list(range(0, 21)),
    list(range(0, 21))
]
divider = np.cumsum(list(map(len, keypoint_list)))*2

import json
class Chunk:
    def __init__(self, name):
        self.name = name
        self.points = self.getDataFromXml(xml_path_dict[name])
        self.label = pd.read_json(json_path_dict[name])['korean_text'].iloc[0]
        self.feature = []

    def getDataFromXml(self, path):
        f = open(path, 'r')
        s = f.read()
        
        df_set = [
            pd.read_xml(s, xpath='./track/body', parser='etree'),       # 0: body_df         
            pd.read_xml(s, xpath='./track/face', parser='etree'),       # 1: face_df         
            pd.read_xml(s, xpath='./track/leftHand', parser='etree'),   # 2: left_hand_df    
            pd.read_xml(s, xpath='./track/rightHand', parser='etree')   # 3: right_hand_df   
        ]
        f.close()
        for i in range(len(df_set)):
            df = df_set[i]
            dfo = df['outside']==1
            
            df_set[i] = df[dfo][['frame','points']]
            df_set[i].set_index(keys=['frame'], drop=True, inplace=True)
            df_set[i] = df_set[i].sort_index()
        
        frames = [df.size for df in df_set ]
        if (sum(frames)/4 * 10)%10 != 0:
            raise Exception(f'Number of Frames are Not Same {frames}')
 
        keypoints = [[], [], [], []]
        for i in range(len(df_set)):
            for df in df_set[i].iterrows():
                points = []
                for pos_str in df[1]['points'].split(';'):
                    arr = np.fromstring(pos_str, sep=',')[:-1]
                    if len(arr) == 0:
                        continue
                    points.append(arr)
        
                keypoints[i].append(points)

        for j in range(4):
            keypoints[j] = np.array(keypoints[j])

        return keypoints # ( frame, joint_idx, xyc )

    def process(self):
        points = [self.points[i][:, keypoint_list[i], :] for i in range(4)]
        abs_xpos = np.hstack([ p[:, :, 0] for p in points ])
        abs_ypos = np.hstack([ p[:, :, 1] for p in points ])
        mx = np.expand_dims(np.mean(abs_xpos, axis=1), axis=1)
        sx = np.expand_dims(np.std (abs_xpos, axis=1), axis=1)
        my = np.expand_dims(np.mean(abs_ypos, axis=1), axis=1)
        sy = np.expand_dims(np.std (abs_ypos, axis=1), axis=1)

        norm_xpos = np.subtract(abs_xpos, mx)/sx
        norm_ypos = np.subtract(abs_ypos, my)/sy

        self.feature = np.dstack((norm_xpos, norm_ypos)).reshape((mx.size, -1))

    def makeDict(self):
        if len(self.feature) == 0:
            return
        d = {}
        d['label'] = self.label
        feature = {}
        # print(f'Interval: {interval}')
        feature['body']  = self.feature[:, :divider[0]]
        feature['face']  = self.feature[:, divider[0]:divider[1] ]
        feature['left']  = self.feature[:, divider[1]:divider[2] ]
        feature['right'] = self.feature[:, divider[2]: ]
        d['feature'] = feature
        return d
        
def getDict(id):
    pkl_path = os.path.join(output_path, f"{id}.pickle")
    if os.path.isfile(pkl_path):
        return 
    else:
        try:
            c = Chunk(id)
            c.process()
            if len(c.feature) == 0:
                return 
            with open(pkl_path, 'wb') as pkl:
                pickle.dump(c.makeDict(), pkl)
        except Exception as err:
            print(f"{id}Error at {xml_path_dict[id]} with {err}")
            return

import time

print(f"ID SIZE: {len(id_list)}")

id_len = len(id_list)

tot = 0

start = time.time()
for i in range(32):
    print(i) 
    with futures.ProcessPoolExecutor() as exec:
        res = exec.map(getDict, id_list[i*id_len//32:(i+1)*id_len//32])
        r = list(filter(None, list(res)))
        tot += sum(r)
end = time.time()
print(f"latency: {(end - start)/10}")
print(f"total data cnt: {tot}")
