# -*- coding:utf8 -*-
import cv2
import os
import shutil
import numpy as np


class GetFrame():

    def __init__(self, filepath, video_class=".mp4"):
        self.filepath = filepath
        self.interval = 1 # 默认抽帧间隔为0，即全部保存
        self.video_class = video_class

    # 从单个视频中抽帧保存图片
    def get_frame_from_video(self, video_name, multifolder=True):
        print("save frame from {}".format(video_name))
        if multifolder:
            # 保存图片的路径, 按视频名称保存
            save_path = video_name.split(self.video_class)[0]
            save_path = os.path.join(self.filepath, save_path)
        else:
            # 统一保存在一个路径
            save_path = os.path.join(self.filepath, "img")
        print("save path is ", save_path)

        is_exists = os.path.exists(save_path)
        if not is_exists:
            os.makedirs(save_path)
            print('path of %s is build' % save_path)    
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
            print('path of %s already exist and rebuild' % save_path)
    
        # 开始读视频
        video_path = os.path.join(self.filepath, video_name)
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        self.interval = 3
        # self.interval = int(fps)
        print("the video fps is {} and interval is {}".format(int(fps), self.interval))

        i = 0
        j = 0

        while True:
            success, frame = video_capture.read()            
            if not success:
                print("video of {} read failed".format(video_path))
                break
            else:
                i += 1
                if i % self.interval == 0:
                    # 保存图片
                    j += 1

                    save_name = save_path + '/' + video_name.split(self.video_class)[0] + \
                        '_' + str(j).zfill(zero_fill_length) + '_' + str(i).zfill(zero_fill_length) + '.jpg'

                    # frame = np.rot90(frame, -1)
                    cv2.imwrite(save_name, frame)
                    print('image of %s is saved' % save_name)

    
    # 遍历文件夹读取视频
    def read_file_by_path(self):
        filelist = os.listdir(self.filepath)
        for item in filelist:
            if item.endswith(self.video_class):
                print(item)
                self.get_frame_from_video(item, multifolder=False)


if __name__ == '__main__':
    test = GetFrame("./srcvideo", video_class=".dav")
    test.read_file_by_path()
