# -*- coding: utf-8 -*-
from models.experimental import attempt_load
import cv2
import torch
from utils.general import check_img_size, non_max_suppression,scale_coords
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
import os
import warnings
from models import CircleDial as CDR, ArcGaugeRecognition as AGR
from models import FourtyfourLineFBTCircleDial as FFGR
from models import FourtyThreeLineCircleDial as FTGR
import time
from models import SevenSegCluster as SCL
warnings.filterwarnings('ignore', category=FutureWarning)

def get_next_filename(base_path):
    # 分離路徑和檔案名
    directory, filename = os.path.split(base_path)
    # 分離檔案名和副檔名
    name, extension = os.path.splitext(filename)
    filename = name.split('_')[0]
    count = 1
    new_filename = f"{filename}_{count}{extension}"
    final_path = os.path.join(directory, new_filename)
    # 開始編號
    while os.path.exists(final_path):
        # 組合新的檔案名稱
        count += 1
        new_filename = f"{filename}_{count}{extension}"
        final_path = os.path.join(directory, new_filename)
    return final_path
def detection_process(model, frame, device):
    device = select_device(device)  # half precision only supported on CUDA
    half = device.type != 'cpu'
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(640, s=stride)  # check img_size
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    imgs = [None]
    imgs[0] = frame
    img0 = imgs.copy()
    img = [letterbox(x, 640, stride=stride)[0] for x in img0]
    # Stack
    img = np.stack(img, 0)
    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, img0
def DialInference(model,img,shape):#第一階段錶頭種類辨識
    # Inference
    try:
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.7, 0.45)  # prediction, conf, iou
        for i, det in enumerate(pred):  # detections per image
            if len(det) == 1:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], shape).round()
                for d in det:# d = (x1, y1, x2, y2, conf, cls)
                    x1 = int(d[0].item())
                    y1 = int(d[1].item())
                    x2 = int(d[2].item())
                    y2 = int(d[3].item())
                    classidx = int(d[5].item())
                    DialRes = Dialnames[classidx].strip()
                    cropped_coord =[x1,y1,x2,y2]
                    return DialRes, cropped_coord
            else:
                return None,None
    except Exception:
        return None,None

def InferenceNum(DialResult,cropped_coord,model,img,img0,Oriimgshape,Segnames):#第二階段錶頭數字辨識
    # Inference
    # t1 = time_synchronized()
    frame = img0[0].copy()
    detection_result = []
    if cropped_coord == None:#錶頭切割範圍座標未取得直接回傳空
        return None,None,None
    x1, y1, x2, y2 = cropped_coord#取錶頭切割範圍座標
    if (DialResult == '7Seg'):
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.5, 0.45)  # prediction, conf, iou
        if(len(pred) > 0):
            clustered_detections = SCL.Getclusterdet(pred,img,Oriimgshape)
            if clustered_detections == None:#沒有找出數位表頭分群結果，回傳空
                return None,None,None
            else:
                frame,detection_result = SCL.GetSevenSegDetection(frame,detection_result,clustered_detections,DialResult,Segnames)
            return (img0[0], frame, detection_result)
    else:
        Value = -1
        if (DialResult == 'CircleGuage'):
            CropImg = frame[y1:y2, x1:x2]
            Value = round(CDR.get_41CircleDialvalue(CropImg), 2)
        elif (DialResult == 'ArcGauge'):
            CropImg = frame[y1:y2, x1:x2]
            Value = round(AGR.get_value(CropImg), 2)
        elif (DialResult == '44LineCircleGuageFBT'):
            CropImg = frame[y1:y2, x1:x2]
            Value = round(FFGR.get_44LineFBTCircleDialvalue(CropImg), 2)
        elif (DialResult == '43LineCircleGuage'):
            CropImg = frame[y1:y2, x1:x2]
            Value = round(FTGR.get_43LineCircleDialvalue(CropImg), 2)
        detection_result.append([x1, y1, x2, y2, 1.0, Value, DialResult])
        if Value == -1:#沒有辨識出類比表頭數值，回傳空
            return None,None, None
        else:
            return (img0[0], frame, detection_result)


# 使用範例
if __name__ == "__main__":
    path = os.getcwd()  # 取得目前路徑
    #print(path)  # 印出工作路徑
    image_path = 'Test\\42CircleDial\\CircleDialImage-1XX.png'  # 輸入你的圖片路徑
    Dial_model_path = 'Dial.pt'# 輸入錶頭模型路徑
    Seg_model_path = 'seg.pt'  # 輸入7段顯示器模型路徑
    Oriimg = cv2.imread(image_path)    # image 讀取 BGR format
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'#確認使用甚麼硬體設備，只需確認一次
    DialModel = attempt_load(Dial_model_path, map_location=device) #讀取模型，只需確認一次
    SevenSegModel = attempt_load(Seg_model_path, map_location=device)#讀取模型，只需確認一次
    Dialnames = DialModel.module.names if hasattr(DialModel, 'module') else DialModel.names #讀取模型種類，只需確認一次
    Segnames = SevenSegModel.module.names if hasattr(SevenSegModel, 'module') else SevenSegModel.names #讀取模型種類，只需確認一次
    start = time.time()
    imgtensor,imgshow = detection_process(DialModel,Oriimg,device)#影像轉換前處理，新圖像輸入每次都要做
    DialResult,CropCood = DialInference(DialModel,imgtensor,Oriimg.shape)#影像推論，回傳錶頭辨識結果，新圖像輸入每次都要做
    oriframe,afterframe,detection_result = InferenceNum(DialResult,CropCood,SevenSegModel,imgtensor,imgshow,Oriimg.shape,Segnames)#影像推論，回傳錶頭數字分群算法及結果，每次都要做
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    #newimage_path = image_path.replace('.png', '_Result.png')
    #cv2.imwrite(newimage_path,afterframe)
    #images = cv2.resize(afterframe, (512, 512), interpolation=cv2.INTER_AREA)
    #cv2.namedWindow("images", cv2.WINDOW_NORMAL)
    #cv2.imshow('test',images)
    #cv2.waitKey(0)  # 1 millisecond