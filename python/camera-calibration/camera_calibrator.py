import cv2
import numpy as np
import glob


#棋盘格模板规格
len = 18 # 黑白格实际长度, 单位mm
w = 6
h = 13

#criteria:角点精准化迭代过程的终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
world_point = np.zeros((w*h, 3), np.float32)#初始化一个6*13行3列的矩阵，类型为float

#把每一个世界坐标的xy赋值[:,:2]中，：表示列表的所有字列表全部，：2表示从1截止到第二个数字
world_point[:, :2] = np.mgrid[0:w*len:len, 0:h*len:len].T.reshape(-1, 2)
# 储存棋盘格角点的世界坐标和图像坐标对
world_points = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
j = 1

images = glob.glob('*.jpg')#读取所有jpg文件
for fname in images:
    img = cv2.imread(fname)
    #将图片缩小
    img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    #寻找角点，存入corners，ret是找到角点的flag（如果找到角点则为true）
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    if ret is True:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #加入到世界坐标系
        world_points.append(world_point)
        #角点加入到图像坐标系
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)    
        #保存渲染棋盘格角点图
        cv2.imwrite(f"find_corners_in_image_{j}.png", img)
        print(str(j) + "完成\n")
        j += 1
    else:
        print('错误')
        j += 1
# cv2.destroyAllWindows()

#求解摄像机的内在参数和外在参数。mtx 内参数矩阵，dist 畸变系数，rvecs 旋转向量，tvecs 平移向量 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, imgpoints, gray.shape[::-1], None, None)

# 找棋盘格角点
# 去畸变
img2 = cv2.imread('1.jpg') #原有畸变图像
img2 = cv2.resize(img2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
# cv2.imshow("畸变图像", img2)
# cv2.waitKey(100)
# cv2.destroyAllWindows()
h, w = img2.shape[:2]
#内参数矩阵
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数
#校正后图像
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
cv2.imwrite('calib_result.png', dst)

with open('内参数矩阵.txt', 'w+') as f:
    f.write('内参数矩阵:\n'+str(newcameramtx)+'\n')
print('内参数输出完毕')

with open('外参数矩阵.txt', 'w+') as f:
    #得到旋转矩阵与平移
    for t in range(0, j-1):#向量变为矩阵
        newrvecs = cv2.Rodrigues(rvecs[t].ravel())
        newtvecs = cv2.Rodrigues(tvecs[t].ravel())
        f.write('第'+str(t+1)+'张图像的外参数矩阵:\n'+'(1)旋转矩阵：\n'+str(newrvecs[0])+'\n'+'(2)平移矩阵：\n'+str(newtvecs[0])+'\n')
        t += 1
print('外参数输出完毕')

with open('畸变系数.txt', 'w+') as f:
    f.write('畸变系数:\n'+str(dist)+'\n')
print('结果输出完毕')

np.savez('data.npz', mtx= mtx, dist= dist, rvecs=rvecs, tvecs=tvecs)
print('*.npz文件输出完毕')
