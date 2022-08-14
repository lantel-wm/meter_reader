# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from yolov5.detect_meter import run
import paddlex as pdx

def predict():

    
    deeplab_meter_model_dir = 'deeplabv3plus/trained_model/best_model/'
    segmenter_meter = pdx.load_model(deeplab_meter_model_dir)

    deeplab_water_model_dir = 'deeplabv3plus/trained_model_water/best_model/'
    segmenter_water = pdx.load_model(deeplab_water_model_dir)

    # debug信息的输出部分在 yolov5/utils/plots.py box_label() 函数
    run(
        weights='./yolov5/checkpoints/yolov5s_meter.pt',
        # weights='./yolov5/checkpoints/yolov5s_meter.engine',
        # weights='yolov5/runs/train/exp8/weights/best.pt',
        # weights='yolov5/yolov5x.pt',
        source='load_video/video/1.mp4',
        # source='../test_video/new_meter2.mp4',
        # source='datasets/meter_yolo/test/images',
        # source='result/exp19/crops/meter/',
        # source='./datasets/labelme_seg',
        # source='rtsp://admin:a120070001@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0',
        # source='https://192.168.1.2:8080/video',
        # source='https://www.youtube.com/watch?v=L2U9EkDTrBo',
        # source='0',
        data='./yolov5/data/VOC_meter.yaml',
        project='result/',
        # device='cpu',
        # name='meter_reading',
        # save_crop=True,
        # nosave=True,
        segmenters={'meter':segmenter_meter, 'water':segmenter_water},
        debug=True,
    )

if __name__ == '__main__':
    predict()