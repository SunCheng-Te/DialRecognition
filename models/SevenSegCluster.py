import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.general import scale_coords
import torch
def Getclusterdet(pred,img,shape):
    try:
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], shape).round()
        # 假設 rects_coords 是包含矩形框位置的 tensor，形狀為 (n, 4)
        y1_coords = det[:, :4][:, 1]
        min_y1 = torch.min(y1_coords)
        max_y1 = torch.max(y1_coords)
        if (max_y1 - min_y1) > shape[1]*.2:#600:
            clustered_detections = Get_Clustered_detections(det)
        else:
            clustered_detections = [det]
        return clustered_detections
    except Exception:
        return None
def find_union_bbox(detections):
    # 初始化最大聯集框的初始範圍，讓其包含第一個偵測框。
    x1_min, y1_min, x2_max, y2_max = detections[0][:4]
    # 逐一遍歷剩餘的偵測框，更新最大聯集框的範圍以包含當前的偵測框。
    for detection in detections[0:]:
        x1, y1, x2, y2, _, _ = detection
        x1_min = int(min(x1_min, x1))
        y1_min = int(min(y1_min, y1))
        x2_max = int(max(x2_max, x2))
        y2_max = int(max(y2_max, y2))
    return x1_min, y1_min, x2_max, y2_max
def Get_Clustered_detections(det):
    # 提取 y 座標作為特徵進行 K-means 分群
    rects_coords = np.array(det[:, :4])
    y_coords = (rects_coords[:, 1] + rects_coords[:, 3]) / 2
    y_coords = y_coords.reshape(-1, 1)
    # 嘗試不同的分群數量
    max_clusters = int(len(det) / 2)
    best_score = -1
    best_num_clusters = 2
    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters)
        cluster_labels = kmeans.fit_predict(y_coords)
        # 只有在群集數大於1時才計算輪廓分數
        if len(np.unique(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(y_coords, cluster_labels)
            # 如果輪廓分數較高且成本函數有肘部，更新最佳分群數量
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_num_clusters = num_clusters
    # 使用最佳分群數量重新進行 K-means 分群
    kmeans = KMeans(n_clusters=best_num_clusters)
    cluster_labels = kmeans.fit_predict(y_coords)
    # 將原始偵測結果按照分群結果分開
    clustered_detections = [[] for _ in range(best_num_clusters)]
    for i, label in enumerate(cluster_labels):
        clustered_detections[label].append(det[i])
    return clustered_detections
def GetDecimal(Image):
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 74, 255, cv2.THRESH_BINARY)
    # 定义内核大小
    kernel_size = (5, 5)
    # 开操作
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size))
    # 闭操作
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size))
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(Image)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    #min_area = 0.0
    min_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bottom_coordinates = (x + w // 2, y + h)
        area = cv2.contourArea(contour)
        if bottom_coordinates[1] > Image.shape[0] * 0.8:#if area < min_area and bottom_coordinates[1] > Image.shape[0] * 0.5:
            perimeter = cv2.arcLength(contour, True)  # 计算轮廓周长
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)  # 计算圆度
                #print(f"Area: {area}, Circularity: {circularity}")
                # 如果圆度接近1，同时满足面积和位置条件
                if circularity > 0.6:
                    min_area = area
                    min_contour = contour
    if min_contour is not None:
        x, y, _, _ = cv2.boundingRect(min_contour)
        #cv2.drawContours(Image, [min_contour], -1, (0, 255, 0), 10)
        #cv2.imshow('Image with Min Area Contour', Image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return x, y
    else:
        return -1,-1

def GetSevenSegDetection(frame,detection_result,clustered_detections,DialResult,Segnames):
    if clustered_detections != None:
        # clustered_detections 中包含了根據 y 座標分群後的偵測結果
        Dot = True
        for i, group in enumerate(clustered_detections):
            value = ''
            sorted_segdet = sorted(group, key=lambda x: x[0])
            Ubx1, Uby1, Ubx2, Uby2 = find_union_bbox(sorted_segdet)
            cropped_image = frame[Uby1:Uby2, Ubx1:Ubx2]
            decimalx, decimaly = GetDecimal(cropped_image)
            if (decimalx == -1):
                break
            original_x = Ubx1 + decimalx
            for d in sorted_segdet:  # d = (x1, y1, x2, y2, conf, cls)
                x1 = int(d[0].item())
                y1 = int(d[1].item())
                x2 = int(d[2].item())
                y2 = int(d[3].item())
                conf = round(d[4].item(), 2)
                c = int(d[5].item())
                SegClass = f'{Segnames[c]}'.strip()
                midx = (x1 + x2) / 2
                if ((decimalx is not None and decimaly is not None) and midx > original_x) and Dot:
                    value = value + '.'
                    Dot = False
                value = value + str(SegClass)
                detection_result.append([x1, y1, x2, y2, conf, c, DialResult])
                cv2.putText(frame, f'{Segnames[c]} - {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)  # box
            print("Value: ", value)
            Dot = True
        return frame,detection_result