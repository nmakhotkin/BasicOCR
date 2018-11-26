import PIL.Image
import numpy as np
import logging
import io
import os
import re
import tensorflow as tf

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

chrset_index = {}
def read_charset(filename):
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                logging.info('incorrect charset file. line #{}: {}'.format(i, line))
                continue
            code = int(m.group(1)) + 1
            char = m.group(2)
            if char == '<nul>':
                continue
            charset[code] = char
        inv_charset = {}
        for k, v in charset.items():
            inv_charset[v] = k
        return charset, inv_charset


def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    model_path = params['model_path']
    charset,_ = read_charset(os.path.join(model_path,'charset.txt'))
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
    end_line = len(chrset_index)+2
    for i in predictions[0]:
        if i == end_line:
            break;
        t = chrset_index.get(i,-1)
        if t==-1:
            continue;
        line.append(t)
    return {'output': [''.join(line)]}
