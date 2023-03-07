import os
import platform
from PIL import Image

class Img2video:
    """
    imgs_folder: 系列图片路径
    output_folder：输出视频的路径
    """
    def __init__(self, imgs_folder, output_folder='.', format='.jpg') -> None:
        self.imgs_folder = imgs_folder
        self.output_folder = output_folder
        self.files = os.listdir(input_folder)
        self.length = len(self.files)
        self.fill_zeros = len(str(self.length))+1
        self.format_ = format

    def rename(self):
        self.files.sort()
        for i, img in enumerate(self.files):
            print(f'\t {i+1:3}/{self.length}   {img}')
            if self.fill_zeros:
                os.rename(os.path.join(self.imgs_folder, img), 
                            os.path.join(self.imgs_folder, f'ac-1_{str(i).zfill(self.fill_zeros)}{self.format_}'))
            else:
                print("Fill zeros is 0")
                break

    def imgs2video_command(self, fps=8, output_video_name='test.mp4'):
        example_img = self.files[1]
        img0 = Image.open(self.imgs_folder + '/' + example_img)
        if not example_img.startswith('ac-1_'):
            self.rename()
        print('images has format with ac-1_*')
        exe_file = ''
        if platform.system().lower() == 'windows':
            exe_file = 'ffmpeg.exe'
        elif platform.system().lower() == 'linux':
            exe_file = 'ffmpeg'
        else:
            print('system platform is not detected')
        os.system(f'{exe_file} -y -r {fps} -i ./{self.imgs_folder}/ac-1_%0{self.fill_zeros}d{self.format_} -s {img0.width}x{img0.height} -vcodec h264_nvenc {output_video_name}')



if __name__ ==  '__main__':
    input_folder = './水位-桥墩'
    output_folder = '.'
    ins = Img2video(input_folder, output_folder)
    ins.imgs2video_command(fps=6)