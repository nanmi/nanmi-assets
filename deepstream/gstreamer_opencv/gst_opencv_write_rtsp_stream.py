#!/usr/bin/env python
#link:https://blog.csdn.net/zong596568821xp/article/details/108492308

import cv2
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
 
 
def main():
    
    out_send = cv2.VideoWriter('appsrc is-live=true ! videoconvert ! \
                                omxh264enc bitrate=12000000 ! video/x-h264, \
                                stream-format=byte-stream ! rtph264pay pt=96 ! \
                                udpsink host=127.0.0.1 port=5400 async=false',
                                cv2.CAP_GSTREAMER, 0, 30, (1920,1080), True)
 
 
    if not out_send.isOpened():
        print('VideoWriter not opened')
        exit(0)
 
    rtsp_port_num = 8554 
 
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch("(udpsrc name=pay0 port=5400 buffer-size=524288 \
                        caps=\"application/x-rtp, media=video, clock-rate=90000, \
                        encoding-name=(string)H264, payload=96 \")")
                        
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)
 
    # 输出rtsp码流信息
    print("\n *** Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)    
 
    cap = cv2.VideoCapture(0)
 
    while True:
        _, mat = cap.read()
        out_send.write(mat)
        cv2.waitKey(30) 
        
if __name__ == '__main__':
    main()
