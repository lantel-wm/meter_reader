from yolov5 import train

'''
--weights：初始化的权重文件的路径地址
--cfg：模型yaml文件的路径地址
--data：数据yaml文件的路径地址
--hyp：超参数文件路径地址
--epochs：训练轮次
--batch-size：喂入批次文件的多少
--img-size：输入图片尺寸
--rect:是否采用矩形训练，默认False
--resume:接着打断训练上次的结果接着训练
--nosave:不保存模型，默认False
--notest:不进行test，默认False
--noautoanchor:不自动调整anchor，默认False
--evolve:是否进行超参数进化，默认False
--bucket:谷歌云盘bucket，一般不会用到
--cache-images:是否提前缓存图片到内存，以加快训练速度，默认False
--image-weights：使用加权图像选择进行训练
--device:训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
--multi-scale:是否进行多尺度训练，默认False
--single-cls:数据集是否只有一个类别，默认False
--adam:是否使用adam优化器
--sync-bn:是否使用跨卡同步BN,在DDP模式使用
--local_rank：DDP参数，请勿修改
--workers：最大工作核心数
--project:训练模型的保存位置
--name：模型保存的目录名称
--exist-ok：模型目录是否存在，不存在就创建
'''

train.run(
            data='./yolov5/data/VOC_meter.yaml', 
            cfg='./yolov5/models/yolov5s_meter.yaml',
            # weights='./yolov5/runs/train/exp/weights/best.pt', 
            weights='./yolov5/yolov5s.pt',
            epochs=600, 
            batch_size=50, 
            device=0
        )