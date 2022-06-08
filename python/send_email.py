#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/25 17:43
# @Author  : nanmi
# @Email   : yueshangChang@gmail.com
# @File    : mail.py

import cv2
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from mimetypes import guess_type
import time

class MyEmail():

    def __init__(self, sender, receivers, subject):
        self.sender = sender
        self.receivers = receivers
        self.subject = subject
    def send_alarm(self, img, info):

        msgRoot = MIMEMultipart('related')

        msgRoot['From'] = Header(self.sender, 'utf-8')
        # 支持多人接收
        msgRoot['To'] = ','.join(self.receivers)

        msgRoot['Subject'] = Header(self.subject, 'utf-8')

        msgAlternative = MIMEMultipart('alternative')
        msgRoot.attach(msgAlternative)

        # mail_msg = """
        # <p>type:  </p>
        # <p>图片演示：</p>
        # <p><img height=480 width=640 src="cid:image1"></p>
        # """
        mail_msg = '<p>报警类型:%s</p><p>图片演示：</p><p><img height=480 width=640 src="cid:image1"></p>'%(info)

        msgAlternative.attach(MIMEText(mail_msg, 'html', 'utf-8'))

        # 指定图片为当前目录

        # 以文件形式读取的图片
        # (mimetype, encoding) = guess_type(img)
        # (maintype, subtype) = mimetype.split('/')
        # fp = open(img, 'rb')
        # msgImage = MIMEImage(fp.read(), **{'_subtype': subtype})
        # fp.close()

        # 发送opencv读取的图片
        # image = cv2.imread(img)
        img_encode = cv2.imencode('.jpg', img)[1].tobytes()
        msgImage = MIMEImage(img_encode, **{'_subtype': 'jpeg'})

        # 定义图片 ID，在 HTML 文本中引用
        msgImage.add_header('Content-ID', '<image1>')
        msgRoot.attach(msgImage)

        try:
            smtpObj = smtplib.SMTP_SSL('xxxxxxxxx', 465)
            smtpObj.login(self.sender, 'xxxxxxxxx')
            smtpObj.sendmail(self.sender, self.receivers, msgRoot.as_string())
            print('邮件发送成功')
        except smtplib.SMTPException as ex:
            print('Error: 无法发送邮件')
            print(ex)


if __name__ == "__main__":
    sender = 'xxx@xxx.com'
    receivers = ['nanmi<xxx@xxx.com>']
    subject = '值班室检测'
    start = time.time()
    email_test = MyEmail(sender, receivers, subject)
    end = time.time()
    print(end - start)
    email_test.send_alarm(sys.argv[0], sys.argv[1])