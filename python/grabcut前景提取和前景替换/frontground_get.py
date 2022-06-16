import cv2
import numpy as np

src = cv2.imread("sample.jpg")
src = cv2.resize(src, (0,0), fx=0.5, fy=0.5)

# 手动画ROI然后enter buttom
r = cv2.selectROI('input', src, False)  # 返回 (x_min, y_min, w, h)
print("ROI region: ", r)
# roi区域
roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# 原图mask
mask = np.zeros(src.shape[:2], dtype=np.uint8)

# 矩形roi
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) # 包括前景的矩形，格式为(x,y,w,h)

bgdmodel = np.zeros((1, 65),np.float64) # bg模型的临时数组
fgdmodel = np.zeros((1, 65),np.float64) # fg模型的临时数组

cv2.grabCut(src, mask, rect, bgdmodel, fgdmodel, 11, mode=cv2.GC_INIT_WITH_RECT)

# 提取前景和可能的前景区域
mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')

cv2.imshow('mask2', mask2)

result = cv2.bitwise_and(src, src, mask=mask2)

cv2.imwrite('result.jpg', result)
cv2.imwrite('roi.jpg', roi)

cv2.imshow('roi', roi)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
