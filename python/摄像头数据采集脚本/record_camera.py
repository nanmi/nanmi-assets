import cv2
import time
import os
import json
import logging


def set_log():
    logger = logging.getLogger()
    fh = logging.FileHandler("record.log", encoding="utf-8", mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    return logger

class RecordImage(object):
    def __init__(self) -> None:
        self.interval = 0
        self.bridge_list = []
        self.logger = set_log()
    
    def time_stamp(self):
        time_stamp = time.strftime(r"%Y_%m_%d_%H_%M_%S", time.gmtime())
        return time_stamp
    
    def read_json(self):
        # self.camera_num()
        json_path = './cameras.json'
        json_dict = None
        with open(json_path, 'r', encoding='utf-8') as fp:
            json_dict = json.load(fp)
        self.interval = int(json_dict['time_inteval'])
        
        self.bridge_list = list(json_dict.keys())
        return json_dict

    def caputure_(self, save_path:str, i:int, camera_name:str,  camera_url : str) -> None:
        try:
            cap = cv2.VideoCapture(camera_url)
            # time_out = 250 # micro second
            # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"timeout;{time_out}"
            # cap=cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        except Exception as e:
            print(e)
            return
        # cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            print(f'{camera_url} is not opened or not online')
            return
        
        ret, frame = cap.read()
        
        if ret:
            time_s = self.time_stamp()
            print(f'>>> 相机采集-第 {i} 轮采集 - {camera_name} -{time_s} - 第 {i} 张图像')
            self.logger.info(f'>>> 相机采集-第 {i} 轮采集 - {camera_name} -{time_s} - 第 {i} 张图像')
            save_img_path = os.path.join(save_path, f'{camera_name}_{time_s}_{str(i).zfill(8)}.jpg')
            # cv2.imwrite(save_img_path, frame)
            cv2.imencode('.jpg', frame)[1].tofile(save_img_path)
        else:
            print('read frame failed')
            cap.release()
        
        cap.release()

    def makedir_(self, path):
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def runtime(self):
        json_dict = self.read_json()
        i = 0
        base = './record_imgs'
        self.makedir_(base)
        while True:
            for bridge_name in self.bridge_list[1:]:
                bridge_path = os.path.join(base, bridge_name)
                self.makedir_(bridge_path)

                bridge_cameras_lsit = json_dict[bridge_name]
                for camera_json in bridge_cameras_lsit:
                    camera_name = camera_json['camera_name']
                    camera_url = camera_json['url']
                    camera_path = os.path.join(bridge_path, camera_name)
                    self.makedir_(camera_path)
                    if camera_url != "":
                        self.caputure_(camera_path, i, camera_name, camera_url)
            i += 1
            time.sleep(self.interval)

if __name__ == '__main__':
    record = RecordImage()
    record.runtime()


    