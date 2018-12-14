import PIL.Image
import numpy as np
import logging
import io
import os
import re
import tensorflow as tf

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0','1','2','3','4','5','6','7','8','9',
    '-',':','(',')','.',',','/'
    # Apostrophe only for specific cases (eg. : O'clock)
                            "'",
    " ",
    # "end of sentence" character for CTC algorithm
    '_'
]

def read_charset():
    charset = {}
    inv_charset = {}
    for i,v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset

chrset_index = {}

def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    charset,_ = read_charset()
    global chrset_index
    chrset_index = charset
    LOG.info("Init hooks")

def preprocess(inputs):
    LOG.info('inputs: {}'.format(inputs))
    image = inputs['images'][0]
    image = PIL.Image.open(io.BytesIO(image))
    resized_im = norm_image(image,32,320)
    return {
        'images': np.stack([resized_im], axis=0),
    }


def norm_image(im, infer_height, infer_width):
    w, h = im.size
    ration_w = max(w / infer_width, 1.0)
    ration_h = max(h / infer_height, 1.0)
    ratio = max(ration_h, ration_w)
    if ratio > 1:
        width = int(w / ratio)
        height = int(h / ratio)
        im = im.resize((width, height))
    im = np.asarray(im)
    im = im.astype(np.float32) / 127.5 - 1
    pw = max(0, infer_width - im.shape[1])
    ph = max(0, infer_height - im.shape[0])
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return im


def postprocess(outputs):
    LOG.info('outputs: {}'.format(outputs))
    predictions = outputs['output']
    line = []
    end_line = len(chrset_index)-1
    for i in predictions[0]:
        if i == end_line:
            break;
        t = chrset_index.get(i,-1)
        if t==-1:
            continue;
        line.append(t)
    return {'output': [''.join(line)]}
