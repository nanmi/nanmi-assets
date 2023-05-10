import ftplib

ftp = ftplib.FTP()
ftp.connect('localhost', 21)
ftp.login('username', 'password')

ftp.set_pasv(False) # 主动模式

ftp.mkd('dirname')

# 上传文件
with open('filename', 'rb') as f:
    ftp.storbinary('STOR %s' % 'filename', f, 1024)

# 下载文件
with open('filename', 'wb') as f:
    ftp.retrbinary('RETR %s' % 'filename', f.write)

# 关闭FTP连接
ftp.quit()

