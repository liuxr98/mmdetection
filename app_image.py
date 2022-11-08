import mmcv
from flask import Flask, request, Response, render_template, jsonify, flash, send_file
import os

from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

app = Flask(__name__)

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
    return "The AI-Graph-Detection service is running..."


@app.route('/', methods= ['POST'])
def image_detection():
    image = request.files["image"]
    image_name = image.filename
    image.save(image_name)
    image.close()
    raw_image = mmcv.imread(image_name)
    os.remove(image_name)
    result = inference_detector(model, raw_image)
    result_image = model.show_result(raw_image, result, score_thr=score_thr)
    mmcv.imwrite(result_image, 'result.jpg')
    return send_file("result.jpg", mimetype="image/jpg", download_name="result.jpg")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

