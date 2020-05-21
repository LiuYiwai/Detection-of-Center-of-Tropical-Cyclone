import base64
import json
import os

import numpy as np


def encode(mat: np.ndarray) -> dict:
    pixels = mat.flatten()
    return dict(data=base64.b64encode(pixels.astype(np.float).tobytes()).decode('utf-8'), shape=mat.shape)


def decode(rle: dict) -> np.ndarray:
    decode_item = np.frombuffer(base64.b64decode(rle['data']), np.float)
    return decode_item


def read(dir, filenames):
    # read json and filename return matrix and speed in filename
    mat_ret = []
    speed_ret = []
    for filename in filenames:
        path = os.path.join(dir, filename)
        with open(path, 'r') as f:
            mat = list(map(decode, json.load(f)))
        mat_ret.append(mat)

        speed = filename.split('-')[-2][:-3]
        speed = eval(speed)
        speed_ret.append(speed)

    return np.array(mat_ret), np.array(speed_ret)
