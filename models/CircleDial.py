from math import atan2,sqrt,degrees
import math
import numpy as np
import cv2
import traceback
def calculate_angle(line1, line2):
    dx1, dy1 = line1[2] - line1[0], line1[3] - line1[1]
    dx2, dy2 = line2[2] - line2[0], line2[3] - line2[1]
    angle1, angle2 = np.arctan2(dy1, dx1), np.arctan2(dy2, dx2)
    angle_degrees = np.degrees(angle2 - angle1) % 360
    return angle_degrees
def find_intersection_point(P, segment):
    xp, yp = P
    xa, ya, xb, yb = segment
    # 線段方向向量
    segment_vector = (xb - xa, yb - ya)
    # 線段長
    segment_length = np.sqrt((xb - xa)**2 + (yb - ya)**2)
    # 向量 AP 和 AB 的乘積
    dot_product_AP_AB = (xp - xa) * (xb - xa) + (yp - ya) * (yb - ya)
    # 算P到線的投影長度
    projection_length = dot_product_AP_AB / segment_length
    # 验证交点是否在线段上
    if projection_length < 0:
        return int(xa), int(ya)
    elif projection_length > segment_length:
        return int(xb), int(yb)
    else:
        # 计算交点坐标
        intersection_x = xa + (segment_vector[0] / segment_length) * projection_length
        intersection_y = ya + (segment_vector[1] / segment_length) * projection_length
        # 如果投影点在线段上，则返回交点，否则返回投影点
        if 0 <= projection_length <= segment_length:
            return (int(intersection_x), int(intersection_y))
        else:
            return (int(xa), int(ya))  # 投影点在线段外，返回线段的起点
def find_midpoint(segment):
    xa, ya, xb, yb = segment
    midpoint_x = (xa + xb) / 2
    midpoint_y = (ya + yb) / 2
    return int(midpoint_x), int(midpoint_y)
def find_final_union_point(final_lines,nearest_point):
    if(point_on_the_line([final_lines[2]],nearest_point)):
        return nearest_point
    else:
        point = find_intersection_point(nearest_point, final_lines[2])
        return point
def check_line_direction(check_line):
    line_segment = list(check_line)
    if line_segment[1] < line_segment[3]:
        line_segment[0], line_segment[1], line_segment[2], line_segment[3] = line_segment[2], line_segment[3], \
        line_segment[0], line_segment[1]
    return tuple(line_segment)
def point_on_the_line(line, most_common_intersection):
    x1, y1, x2, y2 = line[0]
    px, py = most_common_intersection
    dx = x2 - x1
    dy = y2 - y1
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
    result = True if distance < 5 else False
    return result
def is_line_near_point(line, point, min_length):
    x1, y1, x2, y2 = line
    if len(point) ==0:
        return False
    else:
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
        return 0
def find_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return sqrt(dx * dx + dy * dy)
def find_point_on_line(merged_long_lines, extend_pointer_line, pointer_line):
    point_to_count = {}
    image_height = max(extend_pointer_line[1], extend_pointer_line[3])
    for i in range(len(merged_long_lines) - 1):
        intersection = get_intersection_point(extend_pointer_line, merged_long_lines[i])
        if intersection != 0 and intersection[0] > 0 and intersection[1] > max(pointer_line[1], pointer_line[3]):
            if intersection not in point_to_count:
                point_to_count[intersection] = 0
            point_to_count[intersection] += 1
    nearest_point = ()
    max_count = 0
    for point, count in point_to_count.items():
        if count > max_count and point[1] < image_height * 0.35 and point[1] > image_height * 0.65 and \
                  point[0] > image_height * 0.35 and point[0] < image_height * 0.65:
            max_count = count
            nearest_point = point
        elif count == max_count:
            midpoint = ((extend_pointer_line[0] + extend_pointer_line[2]) / 2,
                        (extend_pointer_line[1] + extend_pointer_line[3]) / 2)
            distance_to_midpoint1 = find_distance(midpoint, nearest_point)
            distance_to_midpoint2 = find_distance(midpoint, point)
            if distance_to_midpoint2 < distance_to_midpoint1 :
                nearest_point = point
    if len(nearest_point) == 0:
        nearest_point = int(image_height * 0.5),int(image_height * 0.5)
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
    short_length = min(eliminate_image.shape[:2])
    cx = short_length // 2
    cy = short_length // 2
    min_r = int(short_length * 0.2)
    max_r = int(short_length * 0.5)
    contours, _ = cv2.findContours(eliminate_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, 3, True)
        contour_points = polygon.reshape(-1, 2)

        top_point = contour_points[np.argmin(contour_points[:, 1])]
        bottom_point = contour_points[np.argmax(contour_points[:, 1])]
        left_point = contour_points[np.argmin(contour_points[:, 0])]
        right_point = contour_points[np.argmax(contour_points[:, 0])]

        top_distance = np.sqrt((top_point[0] - cx) ** 2 + (top_point[1] - cy) ** 2)
        bottom_distance = np.sqrt((bottom_point[0] - cx) ** 2 + (bottom_point[1] - cy) ** 2)
        left_distance = np.sqrt((left_point[0] - cx) ** 2 + (left_point[1] - cy) ** 2)
        right_distance = np.sqrt((right_point[0] - cx) ** 2 + (right_point[1] - cy) ** 2)
        if top_distance > max_r and bottom_distance > max_r and left_distance > max_r and right_distance > max_r:
            cv2.drawContours(eliminate_image, [contour], -1, 0, thickness=cv2.FILLED)
        elif top_distance < min_r and bottom_distance < min_r and left_distance < min_r and right_distance < min_r:
            cv2.drawContours(eliminate_image, [contour], -1, 0, thickness=cv2.FILLED)
        elif top_point[1] < eliminate_image.shape[0] * 0.15 or top_point[1] > eliminate_image.shape[0] * 0.75:
            cv2.drawContours(eliminate_image, [contour], -1, 0, thickness=cv2.FILLED)
        elif top_point[0] < eliminate_image.shape[1] * 0.2 or top_point[0] > eliminate_image.shape[1] * 0.75:
            cv2.drawContours(eliminate_image, [contour], -1, 0, thickness=cv2.FILLED)
    return eliminate_image
def get_eliminate(thresh_image, threshold_max, threshold_min, reverse):
    eliminate_image = thresh_image.copy()
    contours, _ = cv2.findContours(eliminate_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < threshold_max and area > threshold_min:
            cv2.drawContours(eliminate_image, [contour], 0, (0), -1)
    if reverse:
        diff = cv2.absdiff(eliminate_image, thresh_image)
        return diff
    else:
        return eliminate_image
def get_41CircleDialvalue(image):
    #resizeimage = image.copy()
    if max(image.shape[1], image.shape[0]) > 1024:
        image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    elif min(image.shape[1], image.shape[0]) < 512:
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

    shortbounded = image.shape[1] // 2 if image.shape[1] >= image.shape[0] else image.shape[0] // 2

    # Convert image to grayscale
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(grayImg,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=1,
        param1=50,
        param2=50,
        minRadius=shortbounded // 32,
        maxRadius=shortbounded
    )

    if circles is not None:
        circles = np.uint32(np.around(circles))
        circles = sorted(circles[0, :], key=lambda c: (
            (image.shape[:2][1] > c[2] + c[0]) and
            (image.shape[:2][0] > c[2] + c[1]) and
            (int(c[0]) - int(c[2]) > 0) and
            (int(c[1]) - int(c[2]) > 0),
            c[2]
        ), reverse=True)

        # Find the largest circle
        bestCircle = circles[0]
        circleX = int(round(bestCircle[0]))
        circleY = int(round(bestCircle[1]))
        circleRadius = int(round(bestCircle[2]))
        bounding_rect  = (circleX - circleRadius, circleY - circleRadius, 2 * circleRadius, 2 * circleRadius)
        #cv2.circle(image, (circleX,circleY), circleRadius, (255, 0, 0), 3)
        #cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        #cv2.imshow("test", image)
        #cv2.waitKey(0)

        circle_roi_img = image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                         bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
        image = circle_roi_img.copy()

        try:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh_image = cv2.adaptiveThreshold(
                gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
            )
            thresh_image = cv2.bitwise_not(thresh_image)
            #thresh_image = cv2.erode(thresh_image, kernel, iterations=1)
            #cv2.imshow("test", thresh_image)
            #cv2.waitKey(0)
            # 最長指針
            pointer_lines = cv2.HoughLinesP(thresh_image, 1, np.pi / 180, 80, 50, 5)
            pointer_linesort = sorted(pointer_lines, key=lambda x: (
                (min(x[0][0], x[0][2]) > image.shape[1] * 0.15 and
                 min(x[0][1], x[0][3]) > image.shape[0] * 0.15 and
                 max(x[0][0], x[0][2]) < image.shape[1] * 0.8 and
                 max(x[0][1], x[0][3]) < image.shape[0] * 0.8),
                np.linalg.norm(np.array(x[0][0:2]) - np.array(x[0][2:4]))
            ), reverse=True)
            pointer_line = np.mean([pointer_linesort[0], pointer_linesort[1]], axis=0).astype(int)
            thresh_image = get_eliminate(thresh_image, 2000, 0, True)
            thresh_image = get_eliminate(thresh_image, 100, 50, False)
            thresh_image = get_length(thresh_image, 45, 10)
            #cv2.imshow("test", thresh_image)
            #cv2.waitKey(0)
            # 找尋所有線段
            lines = cv2.HoughLinesP(thresh_image, 1, np.pi / 180, 30, 30)

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
                if distance < 30:
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
            for line in merged_lines:
                x1, y1, x2, y2 = line[0]
                line = (x1,y1),(x2,y2)
                long_line = extend_line(line,image.shape)
                merged_long_lines.append(long_line)
                #cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            #cv2.imshow("lines", image)
            #cv2.waitKey(0)
            extend_pointer_line = merged_long_lines[-1]
            x1, y1, x2, y2 =  pointer_line[0]
            pointer_line =(x1,y1,x2,y2)
            nearest_point = find_point_on_line(merged_long_lines,extend_pointer_line,pointer_line)
            #image = cv2.circle(image, nearest_point, radius=10, color=(0, 255, 0), thickness=-1)

            filtered_lines = []
            filtered_short_lines = []
            for i in range(len(merged_long_lines)):
                if is_line_near_point(merged_long_lines[i], nearest_point,
                                      40) and point_on_the_line(merged_lines[i],nearest_point):
                    filtered_lines.append(merged_long_lines[i])
                    filtered_short_lines.append(merged_lines[i])
                    x1, y1, x2, y2 = merged_lines[i][0]
                    #cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #cv2.imshow("Detected", image)
            #cv2.waitKey(0)
            left_bottom_select_lines =[linel for linel in merged_lines if
                                       thresh_image.shape[0] * 0.1 < min(linel[0][0],linel[0][2]) < thresh_image.shape[0] * 0.3 and
                                        thresh_image.shape[1] * 0.65 < min(linel[0][1], linel[0][3]) < thresh_image.shape[1] * 0.8]

            right_bottom_select_lines = [line for line in merged_lines if
                                         thresh_image.shape[0] * 0.65 < max(line[0][0],line[0][2]) < thresh_image.shape[0] * 0.8 and
                                         thresh_image.shape[1] * 0.65 < max(line[0][1],line[0][3]) < thresh_image.shape[1] * 0.8]

            sorted_left_bottom_lines = sorted(left_bottom_select_lines, key=lambda linel: min(linel[0][0], linel[0][2]),
                                              reverse=True)
            sorted_right_bottom_lines = sorted(right_bottom_select_lines, key=lambda linel: min(linel[0][1], linel[0][3]),
                                              reverse=True)

            PointerLineVector = check_line_direction(extend_pointer_line)
            StartShortLine = sorted_left_bottom_lines[0]#left_bottom_select_lines[left_bottom_index]
            EndShortLine = sorted_right_bottom_lines[0]#right_bottom_select_lines[right_bottom_index]

            AngleLines = []
            AngleLines.append(sorted_left_bottom_lines[0])#(left_bottom_Line)
            AngleLines.append(sorted_right_bottom_lines[0])#(right_bottom_line)
            AngleLines.append(PointerLineVector)

            FinalclosestPointF = find_final_union_point(AngleLines,nearest_point)
            AngleLines.clear()
            leftpt = ((StartShortLine[0][0] + StartShortLine[0][2]) // 2, (StartShortLine[0][1] + StartShortLine[0][3]) // 2)
            rightpt = ((EndShortLine[0][0] + EndShortLine[0][2]) // 2, (EndShortLine[0][1] + EndShortLine[0][3]) // 2)
            left_line = FinalclosestPointF[0],FinalclosestPointF[1],leftpt[0],leftpt[1]
            right_line = FinalclosestPointF[0],FinalclosestPointF[1], rightpt[0], rightpt[1]
            AngleLines.append(left_line)
            AngleLines.append(PointerLineVector)
            AngleLines.append(right_line)
            #for line in AngleLines:
            #    x1, y1,x2, y2 = line
            #    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            PointLineAngle = round(calculate_angle(left_line, PointerLineVector))
            LineAngle = round(calculate_angle(left_line, right_line))
            value = round(PointLineAngle/LineAngle * (140))
            print(value)
            #image = cv2.circle(image, FinalclosestPointF, radius=3, color=(0, 255, 0), thickness=-1)

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
            # cv2.putText(image, f'Result: {str(value)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
            #             (0, 0, 255), 1, cv2.LINE_AA)
            # image_path = 'Test\\CircleDial\\CircleGuage_61.png'
            # newimage_path = image_path.replace('.png', '_Result.png')
            # cv2.imwrite(newimage_path,image)
            # cv2.namedWindow("images", cv2.WINDOW_NORMAL)
            # cv2.imshow("images", image)
            # cv2.waitKey(0)

            return value
        except Exception as ex:
            traceback.print_exc()
            print("Error")
            return -1

# 讀取影像
image = cv2.imread('../Test/CircleDial_all/1704677788006.png')

# 呼叫 get_value 函式
result = get_41CircleDialvalue(image)
