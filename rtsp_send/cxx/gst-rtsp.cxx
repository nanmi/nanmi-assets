#include<opencv2/opencv.hpp>
#include<gst/gst.h>
#include<gst/rtsp-server/rtsp-server.h>


int main(int argc, char *argv[])
{
    /* Declarations */
    GMainLoop *loop;
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;
    GError *error = NULL;

    const char* port = "8554";

    cv::VideoWriter out_send("appsrc is-live=true ! videoconvert ! nvvideoconvert ! video/x-raw(memory:NVMM) ! nvv4l2h264enc ! h264parse ! rtph264pay pt=96 ! udpsink host=127.0.0.1 port=5400", \
        cv::CAP_GSTREAMER, 0, (double)25, cv::Size(1280,720), true);
    if(!out_send.isOpened()) 
    { 
        printf("Error: VideoWriter not opened \n");
        exit(-1); 
    } 


    /* create a server instance */
    server = gst_rtsp_server_new();
    g_object_set(server, "service", port, NULL);

    /* get the mount points for this server, every server has a default object
    * that be used to map uri mount points to media factories */
    mounts = gst_rtsp_server_get_mount_points (server);

    /* make a media factory for a test stream. The default media factory can use
   * gst-launch syntax to create pipelines.
   * any launch line works as long as it contains elements named pay%d. Each
   * element with pay%d names will be a stream */
    factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, "(udpsrc name=pay0 port=5400 buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H264, payload=96 \")");
    gst_rtsp_media_factory_set_shared(factory, true);

    /* attach the test factory to the /test url */
    gst_rtsp_mount_points_add_factory(mounts, "/ds-test", factory);

    /* don't need the ref to the mapper anymore */
    g_object_unref(mounts);

    /* attach the server to the default maincontext */
    gst_rtsp_server_attach(server, NULL);

    /* start serving */
    g_print("stream ready at rtsp://127.0.0.1:%s/test\n", port);


    /* write frame and push stream */
    auto cap = cv::VideoCapture("rtsp://xxx.xxx.xxx.xxx");
    // auto cap = cv::VideoCapture("./sample_720p.mp4");
    if(!cap.isOpened()) 
    { 
        printf("Error: VideoCapture not opened \n");
        exit(-1); 
    } 
    cv::Mat frame;
    while (true)
    {
        cap >> frame; //read last frame
        out_send << frame;
        cv::waitKey(1);
    }

    cap.release();
    out_send.release();

    return 0;
}