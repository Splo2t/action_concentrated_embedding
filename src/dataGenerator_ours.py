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
parser.add_argument('--fs'  ,           type=int    ,   default=30)
parser.add_argument('--nfft',           type=int    ,   default=40)
parser.add_argument('--overlap',        type=int    ,   default=30)

args = parser.parse_args()

projectPath = args.data_path # "../"
# video_path = projectPath + 'rawdata/원천데이터/1.mp4_TS001/1.mp4/1.Sangle/1.자연재난/COLDWAVE/1_1'
data_root_path = projectPath + '02_XML_TL/'
label_root_path = projectPath + '03_JSON_TrL/'

fs = args.fs # 30
n = args.nfft #12
overlap = args.overlap #9

output_path = os.path.join(args.output_path, f"{n}_{overlap}")


resolution = fs / n
max_idx = int(10 / resolution)

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

joint_link = [
    # body
    [[0, 0, 1], [0, 1, 2]],# 0
    [[0, 0, 1], [0, 1, 5]],
    [[0, 2, 1], [0, 1, 5]],
    [[0, 1, 2], [0, 2, 3]],
    [[0, 2, 3], [0, 3, 4]], 
    [[0, 1, 5], [0, 5, 6]],
    [[0, 5, 6], [0, 6, 7]], 
    [[0, 17, 18], [0, 0, 1]], # roll of face
    [[0, 6, 7], [2, 0, 9]], 
    [[0, 3, 4], [3, 0, 9]], #   9
    # face 
    # eye
    [[1, 41, 36], [1, 36, 37]],
    [[1, 40, 41], [1, 41, 36]],
    [[1, 47, 42], [1, 42, 43]],
    [[1, 46, 47], [1, 47, 42]],
    # utter mouse
    [[1, 59, 48], [1, 48, 49]],
    [[1, 58, 59], [1, 59, 48]],
    # inner mouse
    [[1, 67, 60], [1, 60, 61]],
    [[1, 66, 67], [1, 67, 61]],
]
body_range = [0, 10]
# face mapping
for i in [[37, 41], [43, 47], [49, 59], [61, 67], [18, 20], [23, 25]]:
    for j in range(i[0], i[1]):
        joint_link.append([[1, j-1, j], [1, j, j+1]])

face_range = [10, len(joint_link)]
# hand
# left
# hand mapping 
left_range = [len(joint_link)]
for h in [2]:
    for i in [1, 5, 9, 13, 17]:
        joint_link.append([[h, 0, i], [h, i, i+1]])
        for j in range(2):
            joint_link.append([[h, i+j, i+j+1], [h, i+j+1, i+j+2]])
left_range.append(len(joint_link))
# right
right_range = [len(joint_link)]
for h in [3]:
    for i in [1, 5, 9, 13, 17]:
        joint_link.append([[h, 0, i], [h, i, i+1]])
        for j in range(2):
            joint_link.append([[h, i+j, i+j+1], [h, i+j+1, i+j+2]])
right_range.append(len(joint_link))


class Chunk:
    def __init__(self, name):
        self.name = name
        self.points = self.getDataFromXml(xml_path_dict[name])
        self.label = pd.read_json(json_path_dict[name])['korean_text'].iloc[0]
        self.feature = []

    def getDataFromXml(self, path):
        f = open(path, 'r')
        s = f.read()
        
        # self.num_frame = pd.read_xml(s, xpath='./meta/task')['stop_frame'].values[0]

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
            raise Exception(f'Number of Frames is Not Same {frames}')
        if(frames[0] < n):
            raise Exception(f'Number of Frame is too small')
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

        # print(f"{self.name}\t{keypoints[0].shape}\t{keypoints[1].shape}\t{keypoints[2].shape}\t{keypoints[3].shape}")

        return keypoints # ( frame, joint_idx, xyc )

    def getAnglesForeachJoints(self, joints):
        anglesPerJoints = []
        idx = 0
        for pos_set in joint_link:
            angleOverTime = []
            for t in range(joints[0].shape[0]):
                vecs = []
                for j in pos_set:
                    # try:
                    vecs.append(joints[j[0]][t, j[1], :] - joints[j[0]][t, j[2], :])
                    # except:
                    #     print(f"Frame Not Matching at {self.name}")
                    #     # print(joints[0].shape)
                    #     return None

                a = np.dot(vecs[0], vecs[1].T)
                b = (np.linalg.norm(vecs[0])*np.linalg.norm(vecs[1]))
                if b == 0:
                    # print(f'WARN At {idx} set\t{t} frame\t{pos_set}')                
                    # for j in pos_set:
                    #     print(f'{joints[j[0]][t, j[1], :2]}\t-\t{joints[j[0]][t, j[2], :2]}')
                    angleOverTime.append(0)
                else:
                    angleOverTime.append(a/b)
            anglesPerJoints.append(angleOverTime)
            idx += 1
        return anglesPerJoints
    
    def getFeatures(self, angles):
        features = []
        for angle in angles:
            f, t, z = signal.stft(angle, fs=fs, nperseg=n, noverlap=overlap, boundary = None, padded=False) # 10 frame
            features.append(np.abs(z))
        return np.vstack(features)

    def process(self):
        angles = self.getAnglesForeachJoints(self.points)
        self.feature = self.getFeatures(angles).T

    def makeDict(self):
        if len(self.feature) == 0:
            return
        d = {}
        d['label'] = self.label
        feature = {}
        interval = self.feature.shape[1] // len(joint_link)
        # print(f'Interval: {interval}')
        feature['body']  = self.feature[:, body_range[0]*interval:body_range[1]*interval]
        feature['face']  = self.feature[:, face_range[0]*interval:face_range[1]*interval ]
        feature['left']  = self.feature[:, left_range[0]*interval:left_range[1]*interval ]
        feature['right'] = self.feature[:, right_range[0]*interval:right_range[1]*interval ]
        d['feature'] = feature
        return d
        
# import time
# start = time.time()

# print(body_range)
# print(face_range)
# print(left_range)
# print(right_range)



import time

# output_folder = f"our_processed_data_{n}_{overlap}_10hz"

print(f"ID SIZE: {len(id_list)}")
if not os.path.isdir(output_path):
    os.mkdir(path=output_path)

def getDict(id):
    pkl_path = os.path.join(output_path, f"{id}.pickle")
    if False and os.path.isfile(pkl_path):
        return
    else:
        try:
            c = Chunk(id)
            c.process()
            if len(c.feature) == 0:
                return 
            with open(pkl_path, 'wb') as pkl:
                pickle.dump(c.makeDict(), pkl)
            return c.feature.shape[0]
        except Exception as err:
            print(f"Error at {xml_path_dict[id]} with {err}")
            return

start = time.time()
id_len = len(id_list)
# with futures.ProcessPoolExecutor() as exec:
#     res = exec.map(getDict, id_list)
#     r = np.array(list(filter(None, list(res))))
#     print(f"{n}_{overlap}")
#     print(f"AVERAGE Token Length = {np.average(r)}")
#     print(f"STANDARD VARIANCE Token Length = {np.std(r)}")
#     print(f"MAXIMUM Token Length = {max(r)}")
#     print(f"Number of Files: {len(r)}")
tot = 0
for i in range(32):
    print(f"{i+1} / 32")
    with futures.ProcessPoolExecutor() as exec:
        res = exec.map(getDict, id_list[i*id_len//32:(i+1)*id_len//32])
        r = list(filter(None, list(res)))
        tot += sum(r)

end = time.time()
print(f"latency: {(end - start)/3}")

# end = time.time()
# with futures.ThreadPoolExecutor()
#  