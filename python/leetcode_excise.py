import pyautogui as pg
import time
import cv2

import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils  # 画线函数
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
drawing_spec1 = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(255, 255, 255))

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
'''
Hands是一个类，他有四个初始化参数，
static_image_mode：是静态图片还是视频帧
max_num_hands：最多检测几只手
min_detection_confidence：置信度阈值
min_tracking_confidence：追踪阈值
'''
cap = cv2.VideoCapture(0)  # 获取视频对象，0是摄像头，也可以输入视频路径
time.sleep(2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 再将图片转化为BGR

    if results.multi_hand_landmarks:  # 该变量非空，表示检测到了手，并且存放了检测到的手的个数
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = image.shape
            
            point8 = hand_landmarks.landmark[8]
            pg.moveTo(int(point8.x*1920), int(point8.y*1080))

            point6 = hand_landmarks.landmark[6]
            point7 = hand_landmarks.landmark[7]

            point10 = hand_landmarks.landmark[10]
            point11 = hand_landmarks.landmark[11]
            point12 = hand_landmarks.landmark[12]

            


            '''
            如果有两只手，则每一次遍历获得的是每只手经过处理的21个关节点坐标的信息，也可以这么写：hand_0 = 
            results.multi_hand_landmarks[0] 
            '''
            
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec1)  # 画线
    cv2.imshow('MediaPipe Hands', image)  # 将图片展示出来
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
hands.close()
cap.release()







