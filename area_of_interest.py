import cv2
import numpy as np
import sys
def pretreatmrnt():
    # 读取图片
    img = cv2.imread('D:/study/picture/HZ.JPG', 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转换到灰度空间
    blur = cv2.blur(gray, (3, 3))  # 图像平滑
    ret, thresh = cv2.threshold(blur, 127, 255, 0)  # 图像的二值阈值化
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)  # 创建一个空白的黑色图片
    color_contours = (0, 255, 0)  # 轮廓颜色
    cv2.drawContours(drawing, contours, -1, color_contours, 3, 8, hierarchy)  # 绘制轮廓
    cv2.namedWindow("photo", cv2.WINDOW_NORMAL)
    cv2.imshow("photo", drawing)
    cv2.waitKey(0)

    hull = []  # 为凸包点创建数组
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))  # 计算每个轮廓点
    for i in range(len(contours)):
        color = (255, 255, 255)  # 凸包的颜色
        cv2.drawContours(drawing, hull, i, color, 3, 8)  # 绘制凸包
    cv2.namedWindow("photo", cv2.WINDOW_NORMAL)
    cv2.imshow("photo", drawing)
    cv2.waitKey(0)
    

if __name__ == '__main__':
    pretreatmrnt() # 图片预处理
