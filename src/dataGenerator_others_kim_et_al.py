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
    [
        [0]+list(range(15, 19)),# face
        [1, 2, 5],
        [6, 7],
        [3, 4]
    ],
    list(range(0, 21)), # left
    list(range(0, 21))  # right 
]
# divider = np.cumsum(list(map(len, keypoint_list)))*2

import json
class Chunk:
    def __init__(self, name):
        self.name = name
        self.points = self.getDataFromXml(xml_path_dict[name])
        # s = json.load(open(json_path_dict[name], 'r', encoding='UTF-8'))['korean_text']
        # # if type(s) == list:
        # #     print(s)
        # #     s = s[0]
        # self.label = s
        self.label = pd.read_json(json_path_dict[name])['korean_text'].iloc[0]
        self.feature = []

    def getDataFromXml(self, path):
        f = open(path, 'r')
        s = f.read()
        
        # self.num_frame = pd.read_xml(s, xpath='./meta/task')['stop_frame'].values[0]

        df_set = [
            pd.read_xml(s, xpath='./track/body', parser='etree'),       # 0: body_df         
            # pd.read_xml(s, xpath='./track/face', parser='etree'),       # 1: face_df         
            pd.read_xml(s, xpath='./track/leftHand', parser='etree'),   # 2: left_hand_df    
            pd.read_xml(s, xpath='./track/rightHand', parser='etree')   # 3: right_hand_df   
        ]
        f.close()
        for i in range(len(df_set)):
            df = df_set[i]
            dfo = df['outside']==1
            # dfi = df['outside']==0 

            # idx_list = set(df[dfi]['frame'].tolist()).difference(set(df[dfo]['frame'].tolist()))
            # df_out = df[dfo]
            # for idx in idx_list:
            #     df_out = pd.concat([df_out, df[df['frame']==idx]])
            
            df_set[i] = df[dfo][['frame','points']]
            df_set[i].set_index(keys=['frame'], drop=True, inplace=True)
            df_set[i] = df_set[i].sort_index()
        
        frames = [df.size for df in df_set ]
        if (sum(frames)/3 * 10)%10 != 0:
            raise Exception(f'Number of Frames are Not Same {frames}')
 
        keypoints = [[], [], []]
        for i in range(len(df_set)):
            for df in df_set[i].iterrows():
                points = []
                for pos_str in df[1]['points'].split(';'):
                    arr = np.fromstring(pos_str, sep=',')[:-1]
                    if len(arr) == 0:
                        continue
                    points.append(arr)
        
                keypoints[i].append(points)

        for j in range(3):
            keypoints[j] = np.array(keypoints[j])

        return keypoints # ( frame, joint_idx, xyc )

    def process(self):
        eps = 1e-8
        body_points = [ [] for i in range(len(keypoint_list[0]))]
        for i in range(len(keypoint_list[0])):
            body_points[i] = self.points[0][:, keypoint_list[0][i], :]

        hand_points = np.hstack([self.points[i][:, keypoint_list[i], :] for i in range(1, 3)])
        cx = np.mean(
            np.hstack(
                [*[
                        body_points[i][:, :, 0] for i in range(len(keypoint_list[0]))
                    ],
                        hand_points[:, :, 0]
                    
                ]), axis=1)
        cy = np.mean(
                np.hstack(
                [
                    *[
                        body_points[i][:, :, 1] for i in range(len(keypoint_list[0]))
                    ],
                        hand_points[:, :, 1]
                        
                ]), axis=1)
        Spoint = [p[:, 0, :] for p in body_points]

        dpoint = np.array([np.sqrt(np.power(cx-r[:, 0], 2) + np.power(cy-r[:, 1], 2)) for r in Spoint])

        bp = np.hstack(
            [
                np.dstack(
                    (np.divide(np.subtract(body_points[i][:, :, 0], np.expand_dims(cx, axis=1)), np.expand_dims(dpoint[i] + eps, axis=1)),
                    np.divide(np.subtract(body_points[i][:, :, 1], np.expand_dims(cy, axis=1)), np.expand_dims(dpoint[i] + eps, axis=1)))
                ) for i in range(len(body_points))
        ])
        
        max_hx = np.expand_dims(np.max(hand_points[:, :, 0], axis=1), axis=1)
        min_hx = np.expand_dims(np.min(hand_points[:, :, 0], axis=1), axis=1)
        max_hy = np.expand_dims(np.max(hand_points[:, :, 1], axis=1), axis=1)
        min_hy = np.expand_dims(np.min(hand_points[:, :, 1], axis=1), axis=1)

        hp = np.dstack(
            (
                np.divide(np.subtract(hand_points[:, :, 0], min_hx), max_hx-min_hx + eps) - 0.5
            ,   np.divide(np.subtract(hand_points[:, :, 1], min_hy), max_hy-min_hy + eps) - 0.5
            )
        )

        self.feature = np.hstack((bp, hp)).reshape((self.points[0].shape[0], -1))

    def makeDict(self):
        if len(self.feature) == 0:
            return
        d = {}
        d['label'] = self.label
        # feature = {}
        # # print(f'Interval: {interval}')
        # feature['body']  = self.feature[:, :divider[0]]
        # feature['face']  = self.feature[:, divider[0]:divider[1] ]
        # feature['left']  = self.feature[:, divider[1]:divider[2] ]
        # feature['right'] = self.feature[:, divider[2]: ]
        d['feature'] = self.feature
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
                print("return zero")
                return 
            with open(pkl_path, 'wb') as pkl:
                pickle.dump(c.makeDict(), pkl)
            return c.feature.shape[0]
        except Exception as err:
            print(f"{id}Error at {xml_path_dict[id]} with {err}")
            return

import time


print(f"ID SIZE: {len(id_list)}")


start = time.time()
id_len = len(id_list)

tot = 0
for i in range(32):
    print(i)
    with futures.ProcessPoolExecutor() as exec:
        res = exec.map(getDict, id_list[i*id_len//32:(i+1)*id_len//32])
        
        r = list(filter(None, list(res)))
        print(len(r))
        tot += sum(r)
end = time.time()
print(f"latency: {end - start}")
print(f"total data cnt: {tot}")


# end = time.time()
# with futures.ThreadPoolExecutor()
#  
