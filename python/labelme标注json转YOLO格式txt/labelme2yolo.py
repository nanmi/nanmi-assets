import json
from typing import Tuple
import os
import re


labels = ["egg"]


def coord_normal_convert(size : Tuple, x, y, w, h): # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    '''
    size: (原图w, 原图h)
    x: bbox在图中的中心点x坐标
    y: bbox在图中的中心点y坐标
    w: bbox实际像素宽度
    h: bbox实际像素高度
    '''
    if size[0] != 0 or size[1] != 0:
        dw = 1./size[0]     # 1/w
        dh = 1./size[1]     # 1/h
    else:
        dw = 0
        dh = 0

    x = x * dw    # 物体中心点x的坐标比(相当于 x/原图w)
    w = w * dw    # 物体宽度的宽度比(相当于 w/原图w)
    y = y * dh    # 物体中心点y的坐标比(相当于 y/原图h)
    h = h * dh    # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]

def labelme2yolo(json_file, output_dir):
    with open(json_file, "r") as f:
        json_content = json.load(f)

    img_w = json_content["imageWidth"]
    img_h = json_content["imageHeight"]
    bboxs = json_content["shapes"]
    print("\t img W:{} H:{} json:{}".format(img_w, img_h, json_file))
    with open(os.path.join(output_dir, re.findall("([\w-]+).json", json_file)[0]+".txt"), "a+") as f:
        for bbox in bboxs:
            label = bbox["label"]
            coord = bbox["points"]
            x = (coord[0][0] + coord[1][0]) / 2
            y = (coord[0][1] + coord[1][1]) / 2
            w = abs(coord[0][0] - coord[1][0])
            h = abs(coord[0][1] - coord[1][1])
            xn, yn, wn, hn = coord_normal_convert((img_w, img_h), x, y, w, h)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(labels.index(label), xn, yn, wn, hn))

def process_each_json(json_dir, output_dir):
    json_lists = os.listdir(json_dir)
    total_json_files = len(json_lists)
    for i, json_file in enumerate(json_lists):
        print("\t --- [{:4d}/{}] {} \t ".format(i+1, total_json_files, json_file), end="")
        labelme2yolo(os.path.join(json_dir, json_file), output_dir)

if __name__ == "__main__":
    json_dir = "./labels"
    output_dir = "./eggs_data/labels/train2017"
    process_each_json(json_dir, output_dir)