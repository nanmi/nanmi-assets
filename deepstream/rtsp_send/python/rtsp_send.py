#!/usr/bin/env python
import cv2
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
 
 
def main():
    
    # out_send = cv2.VideoWriter('appsrc is-live=true ! videoconvert !  video/x-raw format=I420 !\
    #                             x264enc bitrate=12000000 ! video/x-h264, \
    #                             stream-format=byte-stream ! rtph264pay pt=96 ! \
    #                             udpsink host=127.0.0.1 port=5400 async=false',
    #                             cv2.CAP_GSTREAMER, 0, 25, (1280,720), True)

    # out_send = cv2.VideoWriter('appsrc ! videoconvert ! x264enc !  rtph264pay name=pay0 py=96 ! udpsink host=127.0.0.1 port=5400', cv2.CAP_GSTREAMER, 0, 25, (1280,720), True)
    out_send = cv2.VideoWriter('appsrc is-live=true  ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! \
        nvv4l2h264enc ! h264parse ! rtph264pay pt=96 ! udpsink host=127.0.0.1 port=5400', \
            cv2.CAP_GSTREAMER, 0, 25, (1280,720), True)


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
 
    cap = cv2.VideoCapture("rtsp://xxx.xxx.xxx.xxx")
 
    while True:
        ret, mat = cap.read()
        if ret:
            # print("1111")
            out_send.write(mat)
            cv2.waitKey(30) 
        
if __name__ == '__main__':
    main()