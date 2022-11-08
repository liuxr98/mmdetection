import mmcv
import requests
from flask import Flask, request, Response, render_template, jsonify, flash, send_file
import os

from mmcv.runner import load_checkpoint
import cv2

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'some random string'

# Choose to use a config and initialize the detector
config = 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoint/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
score_thr = 0.3

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

@app.route('/status', methods= ['GET'])
def hello():
    return "The AI-Video-Detection service is running..."

import pdb

@app.route('/', methods= ['POST'])
def video_detection():
#     pdb.set_trace()
    ip, port = request.environ['werkzeug.socket'].getpeername()
    port = 10007
    url = f'http://{ip}:{port}/recvimg'
    # receive file
    video = request.files["video"]
    video_path = video.filename
    video.save(video_path)
    flash('Video successfully uploaded')
    # detection video
    score_thr = 0.3
    video_reader = mmcv.VideoReader(video_path)
    os.remove(video_path)
    session = requests.session()
    video_writer = cv2.VideoWriter(
        'result.mp4', 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        video_reader.fps, 
        (video_reader.width, video_reader.height))
    
    count = 0
    for frame in mmcv.track_iter_progress(video_reader):
        count = (count+1)%3
        if count!=1:
            continue
        result = inference_detector(model, frame)
        frame = model.show_result(frame, result, score_thr=score_thr)
        mmcv.imwrite(frame, 'tmp.jpg')
        video_writer.write(frame)
        files = {"image": open('tmp.jpg', 'rb')}
        response = session.post(url=url, files=files)
    
    return send_file('results.mp4')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

