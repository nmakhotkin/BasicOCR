import cv2
import numpy as np
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def preprocess(inputs, **kwargs):
    LOG.info('KW: {}'.format(kwargs))
    LOG.info('inputs: {}'.format(inputs))
    image = inputs['image'][0]
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    resized_im = norm_image(image)
    return {
        'images': np.stack([resized_im], axis=0),
    }


def norm_image(im, infer_height, infer_width):
    h, w, _ = im.shape
    ration_w = max(w / infer_width, 1.0)
    ration_h = max(h / infer_height, 1.0)
    ratio = max(ration_h, ration_w)
    if ratio > 1:
        width = int(w / ratio)
        height = int(h / ratio)
        im = cv2.resize(im, (width, height))
    im = im.astype(np.float32) / 127.5 - 1
    pw = max(0, infer_width - im.shape[1])
    ph = max(0, infer_height - im.shape[0])
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return im


def postprocess(outputs):
    LOG.info('outputs: {}'.format(outputs))
    predictions = outputs['output']
    return {'output': predictions}
