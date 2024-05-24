from math import atan2,sqrt,degrees
import numpy as np
import cv2
import traceback
def Get_Left_line(Leftline):
    one = tuple(Leftline[0].tolist()[0])
    if len(Leftline)>= 3:
        two = tuple(Leftline[1].tolist()[0])
        three = tuple(Leftline[2].tolist()[0])
        if(two[1] < one[1] and two[1] < three[1]):
            return two
        elif(two[1] > three[1] and three[0] - two[0] <20):
            return three
        else:
            return one
    else:
        return one
def Get_Right_line(Rightline):
    one = tuple(Rightline[0].tolist()[0])
    if len(Rightline)>= 3:
        two = tuple(Rightline[1].tolist()[0])
        three = tuple(Rightline[2].tolist()[0])
        if(two[0] < one[0] <30 and two[1] - three[1] > 30):
            return two
        elif(two[0] - three[0] < 30 and three[1] - two[1] < 30):
            return three
        else:
            return one
    else:
        return one
def point_to_line_intersection(point, line):
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    norm = sqrt(dx**2 + dy**2)
    if norm == 0:
        return None  # Line is just a point

    normalized_dx, normalized_dy = dx / norm, dy / norm
    x, y = point
    point_to_line_dx, point_to_line_dy = x - x1, y - y1
    dot_product = point_to_line_dx * normalized_dx + point_to_line_dy * normalized_dy
    intersection_x = x1 + dot_product * normalized_dx
    intersection_y = y1 + dot_product * normalized_dy

    return (int(intersection_x), int(intersection_y))
def calculate_angle(line1, line2):
    dx1 = line1[2] - line1[0]
    dy1 = line1[3] - line1[1]
    dx2 = line2[2] - line2[0]
    dy2 = line2[3] - line2[1]
    angle1, angle2 = np.arctan2(dy1, dx1), np.arctan2(dy2, dx2)
    angle_degrees = np.degrees(angle2 - angle1) % 360
    return angle_degrees
def find_final_union_point(final_lines,nearest_point):
    maxbounded = max(final_lines[2])
    p1 = get_intersection_point(final_lines[0], final_lines[2])
    p2 = get_intersection_point(final_lines[1], final_lines[2])
    closest_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    if(closest_point[0] > maxbounded or closest_point[0] < 0 or closest_point[1] > maxbounded or closest_point[1] < 0):
        return nearest_point
    else:
        if (point_on_the_line([final_lines[2]], nearest_point)):
            return nearest_point
        else:
            return point_to_line_intersection(nearest_point, final_lines[2])
def check_line_direction(check_line):
    line_segment = list(check_line)
    if line_segment[1] < line_segment[3]:
        line_segment[0], line_segment[1], line_segment[2], line_segment[3] = line_segment[2], line_segment[3], \
        line_segment[0], line_segment[1]

    return tuple(line_segment)
def point_on_the_line(line, most_common_intersection):
    x1, y1, x2, y2 = line[0]
    px, py = most_common_intersection
    dx,dy = x2 - x1,y2 - y1
    if dx == 0 and dy == 0:
        return False
    u = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    if u < 0:
        closest_x = x1
        closest_y = y1
    elif u > 1:
        closest_x = x2
        closest_y = y2
    else:
        closest_x = x1 + u * dx
        closest_y = y1 + u * dy

    distance = sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    result = True if distance < 1 else False
    return result

def is_line_near_point(line, point, min_length):
    x1, y1, x2, y2 = line
    px, py = point
    distance = abs((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)) / sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance <= min_length

def get_intersection_point(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    numerator1 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    if denominator != 0:
        t1 = numerator1 / denominator
        intersectionX = x1 + t1 * (x2 - x1)
        intersectionY = y1 + t1 * (y2 - y1)
        return (int(intersectionX), int(intersectionY))
    else:
        return (-1,-1)

def find_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return sqrt(dx * dx + dy * dy)

def find_point_on_line(merged_long_lines, extend_pointer_line, pointer_line):
    point_to_count = {}
    image_height = max(extend_pointer_line[1], extend_pointer_line[3])
    for i in range(len(merged_long_lines) - 1):
        intersection = get_intersection_point(extend_pointer_line, merged_long_lines[i])
        if intersection!= (-1,-1) and intersection[0] > 0 and intersection[1] > max(pointer_line[1], pointer_line[3]):
            if intersection not in point_to_count:
                point_to_count[intersection] = 0
            point_to_count[intersection] += 1
    nearest_point = ()
    max_count = 0
    for point, count in point_to_count.items():
        if count > max_count and point[1] < image_height and point[1] > image_height * 0.75 and point[1] < image_height * 0.95:
            max_count = count
            nearest_point = point
        elif count == max_count:
            midpoint = ((extend_pointer_line[0] + extend_pointer_line[2]) / 2,
                        (extend_pointer_line[1] + extend_pointer_line[3]) / 2)
            distance_to_midpoint1 = find_distance(midpoint, nearest_point)
            distance_to_midpoint2 = find_distance(midpoint, point)
            if distance_to_midpoint2 < distance_to_midpoint1 and midpoint[1] > image_height * 0.75 and midpoint[1] < image_height * 0.9:
                nearest_point = point
    if len(nearest_point) == 0:
        nearest_point = int(image_height * 0.5),int(image_height * 0.85)
        #print("no nearest point")
    return nearest_point

def extend_line(line, size):
    p1 = line[0]
    p2 = line[1]
    if p1[0] == p2[0]:  # 垂直線段
        if p1[1] < p2[1]:
            p2 = (p2[0], size[0])
            p1 = (p1[0], 0)
        else:
            p2 = (p2[0], 0)
            p1 = (p1[0], size[0])
    elif p1[1] == p2[1]:  # 水平線段
        if p1[0] < p2[0]:
            p2 = (size[1], p2[1])
            p1 = (0, p1[1])
        else:
            p2 = (0, p2[1])
            p1 = (size[1], p1[1])
    else:  # 斜線段
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - slope * p1[0]

        if p1[0] < p2[0]:
            p2 = (size[1], int(round(slope * size[1] + b)))
            if p2[1] > size[0]:
                p2 = (int((size[0] - b) / slope), size[0])
            elif p2[1] < 0:
                p2 = (int((0 - b) / slope), 0)
            p1 = (0, int(round(slope * 0 + b)))
            if p1[1] > size[0]:
                p1 = (int((size[0] - b) / slope), size[0])
            elif p1[1] < 0:
                p1 = (int((0 - b) / slope), 0)
        else:
            p2 = (0, int(round(slope * 0 + b)))
            if p2[1] > size[0]:
                p2 = (int((size[0] - b) / slope), size[0])
            elif p2[1] < 0:
                p2 = (int((0 - b) / slope), 0)
            p1 = (size[1], int(round(slope * size[1] + b)))
            if p1[1] > size[0]:
                p1 = (int((size[0] - b) / slope), size[0])
            elif p1[1] < 0:
                p1 = (int((0 - b) / slope), 0)
    return (p1[0],p1[1],p2[0],p2[1])

def get_length(thresh_image, arc_length_max, arc_length_min):
    eliminate_image = thresh_image.copy()
    contours, _ = cv2.findContours(eliminate_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 設定形狀檢測的閾值
    shape_threshold = 0.02  # 較小的閾值可能會檢測到更多的線段
    for contour in contours:
        arc_length = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 3, True)
        contour_points = polygon[:, 0, :]
        top_point = contour_points[np.argmin(contour_points[:, 1])]
        bottom_point = contour_points[np.argmax(contour_points[:, 1])]
        epsilon = shape_threshold * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        #print(len(approx_polygon))
        # 如果逼近後的多邊形是直線，則視為線段，將其保留
        #print(len(approx_polygon))
        #if len(approx_polygon) > 25 :
            #cv2.drawContours(eliminate_image, [contour], 0, (0, 0, 0), -1)
        if (top_point[0] < eliminate_image.shape[1] * 0.92 and bottom_point[0] > eliminate_image.shape[1] * 0.1 and
                top_point[1] < eliminate_image.shape[0] * 0.5 and bottom_point[1] > eliminate_image.shape[0] * 0.25 and
                arc_length > arc_length_min and arc_length < arc_length_max) and len(approx_polygon) < 10:
            cv2.drawContours(eliminate_image, [contour], 0, (0, 0, 0), -1)
    diff = cv2.absdiff(eliminate_image, thresh_image)
    return diff

def get_eliminate(thresh_image, threshold_max, threshold_min, reverse):
    eliminate_image = thresh_image.copy()
    contours, _ = cv2.findContours(eliminate_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if reverse:
        for contour in contours:
            area = cv2.contourArea(contour)
            #print(area)
            perimeter = cv2.arcLength(contour, True)
            # 計算面積和周長之比
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            if (area < threshold_max) and (area > threshold_min):
                cv2.drawContours(eliminate_image, [contour], 0, (0), -1)


        diff = cv2.absdiff(eliminate_image, thresh_image)
        return diff
    else:
        for contour in contours:
            area = cv2.contourArea(contour)
            # 計算面積和周長之比
            #if perimeter > 0:
                #circularity = 4 * np.pi * area / (perimeter * perimeter)
            if area < threshold_max and area > threshold_min and area < 1:
                eliminate_image = cv2.drawContours(eliminate_image, [contour], 0, (0), -1)
                #eliminate_image = cv2.bitwise_and(eliminate_image, [contour])
        return eliminate_image

def get_value(image):
    if image.shape[1] > 1024 or image.shape[0] > 1024:
        image  = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    elif image.shape[1] < 512 or image.shape[0] < 512:
        image  = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

    try:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_image = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 47, 11)
        thresh_image_Ori = cv2.bitwise_not(thresh_image)

        thresh_image = get_eliminate(thresh_image_Ori, 800, 0, True)
        thresh_image = cv2.erode(thresh_image, kernel, iterations=1)

        Line_image_Ori = cv2.erode(thresh_image_Ori, kernel, iterations=1)
        # 最長指針
        pointer_line = cv2.HoughLinesP(Line_image_Ori, 1, np.pi / 180, 30, 30, 5)
        pointer_line = sorted(pointer_line, key=lambda x: (
            (min(x[0][0], x[0][2]) > image.shape[1] * 0.3 and
             min(x[0][1], x[0][3]) > image.shape[0] * 0.3 and
             max(x[0][0], x[0][2]) < image.shape[1] * 0.7 and
             max(x[0][1], x[0][3]) < image.shape[0] * 0.5),
            np.linalg.norm(np.array(x[0][0:2]) - np.array(x[0][2:4]))
        ), reverse=True)[0]

        thresh_image = get_eliminate(thresh_image, 100, 10, True)

        thresh_image = get_length(thresh_image, 100,50)

        #cv2.imshow("thresh_image", thresh_image)
        #cv2.waitKey(0)
        # 找尋所有線段
        lines = cv2.HoughLinesP(thresh_image, 1, np.pi / 180, 15, 50)

        # 建立一個 List 來儲存符合條件的短線條
        merged_lines = []
        merged_long_lines = []
        merged_lines.append(lines[0])
        for i in range(1, len(lines)):#相鄰線段合併
            prev_line = merged_lines[-1]
            prev_angle = np.arctan2(prev_line[0][3] - prev_line[0][1], prev_line[0][2] - prev_line[0][0])
            merged_line_center = (
            (prev_line[0][0] + prev_line[0][2]) // 2, (prev_line[0][1] + prev_line[0][3]) // 2)
            current_line_center = ((lines[i][0][0] + lines[i][0][2]) // 2, (lines[i][0][1] + lines[i][0][3]) // 2)
            distance = np.sqrt((current_line_center[0] - merged_line_center[0]) ** 2 + (
                        current_line_center[1] - merged_line_center[1]) ** 2)
            if distance < 50:
                length = (np.linalg.norm(
                    np.array(prev_line[0][0:2]) - np.array(prev_line[0][2:4])) + np.linalg.norm(
                    np.array(lines[i][0][0:2]) - np.array(lines[i][0][2:4]))) // 2
                start_x = (prev_line[0][0] + lines[i][0][0]) // 2
                start_y = (prev_line[0][1] + lines[i][0][1]) // 2
                end_x = start_x + length * np.cos(prev_angle)
                end_y = start_y + length * np.sin(prev_angle)
                merged_lines[-1][0][0] = int(start_x)
                merged_lines[-1][0][1] = int(start_y)
                merged_lines[-1][0][2] = int(end_x)
                merged_lines[-1][0][3] = int(end_y)
            else:
                merged_lines.append(lines[i])
        merged_lines.append(pointer_line)
        linecount = 0
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            linep = (x1,y1),(x2,y2)
            long_line = extend_line(linep,image.shape)
            merged_long_lines.append(long_line)
            #x1, y1, x2, y2 = line[0]  # line[0]
            #cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            #if(linecount ==len(merged_lines) - 1):
                #cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            #linecount = linecount + 1
        #cv2.imshow("images0", image)
        #cv2.waitKey(0)
        extend_pointer_line = merged_long_lines[-1]
        x1, y1, x2, y2 =  pointer_line[0]
        pointer_line = (x1,y1,x2,y2)
        nearest_point = find_point_on_line(merged_long_lines,extend_pointer_line,pointer_line)

        filtered_lines = []
        filtered_short_lines = []
        for i in range(len(merged_long_lines)):
            filtered_lines.append(merged_long_lines[i])
            filtered_short_lines.append(merged_lines[i])
            x1, y1, x2, y2 = merged_lines[i][0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.imshow("images", image)
        cv2.waitKey(0)
        filtered_short_lines.pop()

        # 使用lambda表達式按照x1的大小進行排序
        Left_sorted_lines = sorted(filtered_short_lines, key=lambda x: x[0][0], reverse=False)
        leftTopLine = Get_Left_line(Left_sorted_lines)
        Right_sorted_lines = sorted(filtered_short_lines, key=lambda x: x[0][0], reverse=True)
        rightTopLine = Get_Right_line(Right_sorted_lines)

        PointerLineVector = check_line_direction(extend_pointer_line)
        StartShortLine = leftTopLine#filtered_short_lines[lefttopindex]
        EndShortLine = rightTopLine#filtered_short_lines[righttopindex]

        AngleLines = []
        AngleLines.append(leftTopLine)
        AngleLines.append(rightTopLine)
        AngleLines.append(PointerLineVector)

        FinalclosestPointF = find_final_union_point(AngleLines,nearest_point)
        AngleLines.clear()
        leftpt = ((StartShortLine[0] + StartShortLine[2]) // 2, (StartShortLine[1] + StartShortLine[3]) // 2)
        rightpt = ((EndShortLine[0] + EndShortLine[2]) // 2, (EndShortLine[1] + EndShortLine[3]) // 2)
        left_line = FinalclosestPointF[0],FinalclosestPointF[1],leftpt[0],leftpt[1]
        right_line = FinalclosestPointF[0],FinalclosestPointF[1], rightpt[0], rightpt[1]
        AngleLines.append(left_line)
        AngleLines.append(PointerLineVector)
        AngleLines.append(right_line)
        # count=0
        # for line in AngleLines:
        #     x1, y1,x2, y2 = line
        #     if count == 0:
        #         cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #     elif count == 1:
        #         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     else:
        #         cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #     count =count+1
        # cv2.circle(image, FinalclosestPointF, radius=10, color=(0, 255, 0), thickness=-1)
        # cv2.imshow("images", image)
        # cv2.waitKey(0)

        PointLineAngle = round(calculate_angle(left_line, PointerLineVector))
        LineAngle = round(calculate_angle(left_line, right_line))
        value = round(PointLineAngle/LineAngle * (1000))
        if(value) > 1000:
            return -1
        print(value)
        return value

    except Exception as ex:
        traceback.print_exc()
        return -1

# 讀取影像
#image = cv2.imread('../Test/ArcGauge_all/1704676875110.png')

# 呼叫 get_value 函式
#result = get_value(image)
