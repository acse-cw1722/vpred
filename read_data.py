import os
import cv2 as cv
import numpy as np


def get_vid_path(root=os.sep.join([
        os.path.dirname(__file__),
        'VIDEOS'
        ])):
    files_path = []
    print(root)
    for root, _, files in os.walk(root):
        for file in files:
            files_path.append(os.sep.join([root, file]))
            print(files_path)
    return files_path


def build_data_set(paths):
    data = []
    cap = cv.VideoCapture()
    for p in paths:
        cap.open(p)
        vid = get_vid(cap)
        data.append(vid)
    cap.release()
    print(len(data))
    return np.stack(data, axis=0)


def get_vid(cap):
    vid = []
    ret = True
    while (ret):
        ret, img = cap.read()
        if (ret):
            vid.append(img)
    return np.stack(vid, axis=0)
