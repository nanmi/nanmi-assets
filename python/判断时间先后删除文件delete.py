import os
from datetime import datetime


target_time = '2023-02-16 10:57:18'

path = './水位-桥墩'

format_pattern = '%Y-%m-%d %H:%M:%S'
def judge_time(image_time) -> bool:
    difference = (datetime.strptime(image_time , format_pattern) - datetime.strptime(target_time, format_pattern))
    if difference.days < 0:
        return True
    else:
        return False


def main():
    imgs_list = os.listdir(path)
    length = len(imgs_list)
    for i, img_p in enumerate(imgs_list):
        print(f'\t {i+1:5}/{length} - {img_p}')
        time_stamp = img_p.split('_')
        date_ = time_stamp[1:4]
        time_ = time_stamp[4:7]
        image_time = f"{'-'.join(date_)} {':'.join(time_)}"
        if judge_time(image_time):
            os.remove(os.path.join(path, img_p))

if __name__ == '__main__':
    main()














