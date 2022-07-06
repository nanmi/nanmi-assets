from re import X
import cv2
import numpy as np

image = np.ones([1080, 1920, 3], np.uint8) * 255
square_pixel = 120   # 棋盘格每个小个子宽高, 1080和1920最大公约数 = 120 pixels
x_nums = 14 # 棋盘格宽, 最大16, 左右预留边界一个格子
y_nums = 7  # 棋盘格高, 最大9, 左右预留边界一个格子

x0 = square_pixel
y0 = square_pixel

def DrawSquare():
    flag = -1
    for i in range(y_nums):
        flag = 0 - flag
        for j in range(x_nums):
            if flag > 0:
                color = [0,0,0]
            else:
                color = [255,255,255]
            cv2.rectangle(image, (x0 + j*square_pixel, y0 + i*square_pixel),
                                (x0 + j*square_pixel+square_pixel, y0 + i*square_pixel+square_pixel), color, -1)
            flag = 0 - flag
    cv2.imwrite(f'./chess_map_{x_nums}x{y_nums}.bmp',image)
 
if __name__ == '__main__':
    DrawSquare()

