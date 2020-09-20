import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


def YCbCr(path):  # 高斯似然需要调参
    img = cv2.imread(path, 1)
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (Y, Cr, Cb) = cv2.split(YCrCb)
    h, w = Y.shape
    res = np.zeros((h, w))  # 结果矩阵
    M = np.array([[124.2125], [132.9449]])  # 肤色均值
    C = np.array([[75.3881, 40.2587], [40.2587, 250.22942]])  # 肤色方差
    cbcr = np.array([[0], [0]])  # cb,cr分量矩阵
    for i in range(h):
        for j in range(w):
            cbcr[0, 0], cbcr[1, 0] = Cr[i, j], Cb[i, j]
            mid_cm = np.transpose(cbcr - M)
            C_ = np.linalg.inv(C)  # 求逆
            mid_ = cbcr - M
            mid = np.dot(mid_cm, C_)
            mid_t = np.dot(mid, mid_)
            mid_num = -0.5*mid_t
            num = np.exp(mid_num)
            res[i, j] = num
    cv2.namedWindow("photo", cv2.WINDOW_NORMAL)
    cv2.imshow("photo", res)
    cv2.waitKey(0)


def white_balance(path):
    start = time.time()

    img = cv2.imread(path, 1) # 打开彩色图片
    B, G, R = cv2.split(img) # 提取图片三通道值
    m, n, t = img.shape  #图片长宽深
    num = m*n

    avgB = np.sum(B) / num
    avgG = np.sum(G) / num
    avgR = np.sum(R) / num

    V = (avgB+avgG+avgR) / 3
    Vb = V / avgB
    Vg = V / avgG
    Vr = V / avgR

    for i in range(m):
        for j in range(n):
            B[i][j] = B[i][j]*Vb if B[i][j]*Vb < 255 else 255
            G[i][j] = G[i][j]*Vg if G[i][j]*Vg < 255 else 255
            R[i][j] = R[i][j]*Vr if R[i][j]*Vr < 255 else 255
    img_t = cv2.merge([B, G, R])

    end = time.time()
    print(end - start)
    cv2.namedWindow("photo", cv2.WINDOW_NORMAL)
    cv2.imshow("photo", img_t)
    cv2.waitKey(0)
    YCbCr(img_t) # 色彩空间转换
    pass


def main():
    path = "D:/study/picture/TLL.png"
    white_balance(path)


if __name__ == '__main__':
    main()
