import cv2
import time
import os
import configparser

class RecordImage(object):
    def __init__(self) -> None:
        self.n_camera = 0
        self.rtsp_list = {}
        self.interval = 0
    
    def time_stamp(self):
        time_stamp = time.strftime(r"%Y_%m_%d_%H_%M_%S", time.gmtime())
        return time_stamp

    def camera_num(self):
        count = 0
        with open('./cameras.cfg', 'r', encoding='utf-8') as f:
            line = 'a'
            while line != '':
                line = f.readline()
                if line != '' and line[:3] == 'URI':
                    count += 1
        self.n_camera = count

    def read_cfg(self):
        self.camera_num()
        cfg_path = './cameras.cfg'
        cp = configparser.RawConfigParser()
        cp.read(cfg_path, encoding='utf-8')
        self.interval = int(cp.get('sample', 'time'))
        for i in range(self.n_camera):
            loca = cp.get(f'rtsp{i}', 'LOCATION')
            uri = cp.get(f'rtsp{i}', 'URI')
            self.rtsp_list[f'rtsp{i}'] = [str(loca), str(uri)]

    def caputure_(self, save_path:str, i:int, rtsp : str) -> None:
        try:
            # cap = cv2.VideoCapture(self.rtsp_list[rtsp][1])
            time_out = 250 # micro second
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"timeout;{time_out}"
            cap=cv2.VideoCapture(self.rtsp_list[rtsp][1], cv2.CAP_FFMPEG)
        except Exception as e:
            # print(e)
            return
        
        if not cap.isOpened():
            print(f'{rtsp} is not opened or not online')
            return
        
        ret, frame = cap.read()
        if ret:
            time_s = self.time_stamp()
            loca = self.rtsp_list[rtsp][0]
            print(f'>>> 相机采集-{loca}-{time_s} - 第 {i} 张图像')
            save_img_path = os.path.join(save_path, f'{loca}_{time_s}_{str(i).zfill(8)}.jpg')
            # cv2.imwrite(save_img_path, frame)
            cv2.imencode('.jpg', frame)[1].tofile(save_img_path)
        else:
            print('read frame failed')
            cap.release()
        return

    def makedir_(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def runtime(self):
        self.read_cfg()
        i = 0
        base = './record_imgs'
        if not os.path.exists(base):
            os.makedirs(base)
        while True:
            for c_i in range(self.n_camera):
                rtsp = f'rtsp{c_i}'
                path = os.path.join(base, self.rtsp_list[rtsp][0])
                self.makedir_(path)
                self.caputure_(path, i, rtsp)
            i += 1
            time.sleep(self.interval)

if __name__ == '__main__':
    record = RecordImage()
    record.runtime()


    