import os

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

    def imgs2video_command(self, fps=8):
        self.rename()
        os.system(f'ffmpeg -y -r {fps} -i ./{self.imgs_folder}/ac-1_%0{self.fill_zeros}d{self.format_} -vcodec h264_nvenc test.mp4')



if __name__ ==  '__main__':
    input_folder = './tt'
    output_folder = '.'
    ins = Img2video(input_folder, output_folder)
    ins.imgs2video_command()