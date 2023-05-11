/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>

#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"

#include "gstnvdsinfer.h" // for NvDsInferTensorMeta
#include "nvdsinfer.h" // for NvDsInferLayerInfo
#include "cuda_runtime.h" // for cudaMemcpyDeviceToHost
#include <opencv2/opencv.hpp>

int MUXER_BATCH_TIMEOUT_USEC = 40000;
int MUXER_OUTPUT_WIDTH = 960;
int MUXER_OUTPUT_HEIGHT = 640;
int BATCH_SIZE = 1;

void save_imgs(NvDsFrameMeta *frame_meta)
{
  for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list;
        l_user != NULL; l_user = l_user->next)
  {
    NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
    if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
      continue;

    NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;

    for (unsigned int i = 0; i < meta->num_output_layers; i++)
    {
      static int ii = 0;
      NvDsInferLayerInfo *info = &meta->output_layers_info[i];
      info->buffer = meta->out_buf_ptrs_host[i];
      if (meta->out_buf_ptrs_dev[i])
      {
        cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                    info->inferDims.numElements * 1, cudaMemcpyDeviceToHost);
      }
      NvDsInferDimsCHW LayerDims;
      NvDsInferLayerInfo *outputLayersInfo = meta->output_layers_info;
      getDimsCHWFromDims(LayerDims, outputLayersInfo[i].inferDims);
      // Prepare CVCORE input layers
      if (strcmp(outputLayersInfo[i].layerName, "output") == 0)
      {
        // The layer of enhanced_image contains 1x640x960 float
        float *heatmap_data = (float *)meta->out_buf_ptrs_host[i];
        int heatmap_w = LayerDims.w; // 960
        int heatmap_h = LayerDims.h; // 640
        int heatmap_c = LayerDims.c; // 1

        if (heatmap_data != nullptr)
        {
          cv::Mat image = cv::Mat(heatmap_h, heatmap_w, CV_32FC3, heatmap_data);
          printf(">>>>>>>>>>>>>\n");
          cv::imwrite("../imgs/" + std::to_string(ii) + ".jpg", image);
          ii++;
        }
      }
    }
  }
}


/* tiler_sink_pad_buffer_probe  will extract metadata received on segmentation
 *  src pad */
static GstPadProbeReturn
tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsMetaList * l_frame = nullptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != nullptr;
      l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
      if (frame_meta == NULL)
        continue;
      // NvDsDisplayMeta* display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
      save_imgs(frame_meta);
    }
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      // Add the delay to show the result
      usleep(2000000);
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

//gst-launch-1.0 filesrc location=xxx.jpg ! jpegparse ! nvv4l2decoder ! ...
static GstElement *
create_jpeg_source_bin (gchar * uri)
{
  GstElement *bin = nullptr;

  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new ("source-bin");

  GstElement *source, *jpegparser, *decoder;

  source = gst_element_factory_make ("filesrc", "source");

  jpegparser = gst_element_factory_make ("jpegparse", "jpeg-parser");

  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  if (!source || !jpegparser || !decoder)
  {
    g_printerr ("One element could not be created. Exiting.\n");
    return nullptr;
  }
  g_object_set (G_OBJECT (source), "location", uri, nullptr);

  gst_bin_add_many (GST_BIN (bin), source, jpegparser, decoder, nullptr);

  gst_element_link_many (source, jpegparser, decoder, nullptr);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return nullptr;
  }

  GstPad *srcpad = gst_element_get_static_pad (decoder, "src");
  if (!srcpad) {
    g_printerr ("Failed to get src pad of source bin. Exiting.\n");
    return nullptr;
  }
  GstPad *bin_ghost_pad = gst_element_get_static_pad (bin, "src");
  if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
        srcpad)) {
    g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
  }

  return bin;
}

//gst-launch-0.10 filesrc location=xxxx.mp4 ! qtdemux name=demux0 demux0.video_0或者demux0.audio_0 ! h264parse ! nvv4l2decoder ! ...
static GstElement *
create_mp4_source_bin (gchar * uri)
{
  GstElement *bin = nullptr;

  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new ("source-bin");

  GstElement *source, *qtdemuxer, *h264parser, *decoder;

  source = gst_element_factory_make ("filesrc", "source");

  qtdemuxer = gst_element_factory_make ("qtdemux", "qtdemux-er");

  h264parser = gst_element_factory_make ("h264parse", "h264pars-er");

  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  if (!source || !qtdemuxer || !h264parser || !decoder)
  {
    g_printerr ("One element could not be created. Exiting.\n");
    return nullptr;
  }
  g_object_set (G_OBJECT (source), "location", uri, nullptr);

  gst_bin_add_many (GST_BIN (bin), source, qtdemuxer, h264parser, decoder, nullptr);

  // source -> qtdemuxer
  gst_element_link_many (source, qtdemuxer, nullptr);

  // qtdemuxer -> h264parser
  GstPad* qtdemuxer_srcpad = gst_element_get_request_pad (qtdemuxer, "video_0");
  if (!qtdemuxer_srcpad) {
    g_printerr ("qtdemuxer request src pad failed. Exiting.\n");
    return -1;
  }

  GstPad *h264parser_sinkpad = gst_element_get_static_pad (h264parser, "sink");
  if (!h264parser_sinkpad) {
    g_printerr ("Failed to get sink pad of h264parser. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (qtdemuxer_srcpad, h264parser_sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link qtdemuxer to h264parser. Exiting.\n");
    return -1;
  }

  // h264parser -> decoder
  gst_element_link_many (h264parser, decoder, nullptr);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return nullptr;
  }

  GstPad *srcpad = gst_element_get_static_pad (decoder, "src");
  if (!srcpad) {
    g_printerr ("Failed to get src pad of source bin. Exiting.\n");
    return nullptr;
  }
  GstPad *bin_ghost_pad = gst_element_get_static_pad (bin, "src");
  if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
        srcpad)) {
    g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
  }

  return bin;
}

//gst-launch-0.10 rtspsrc name=src0 location=rtsp://... src0.stream_0 ! rtph264depay ! h264parse ! nvv4l2decoder ! ...
static GstElement *
create_rtsp_source_bin (gchar * uri)
{
  GstElement *bin = nullptr;

  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new ("source-bin");

  GstElement *source, *rtph264depayer, *h264parser, *decoder;

  source = gst_element_factory_make ("rtspsrc", "source_input");

  rtph264depayer = gst_element_factory_make ("rtph264depay", "qtdemux-er");

  h264parser = gst_element_factory_make ("h264parse", "h264pars-er");

  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  if (!source || !rtph264depayer || !h264parser || !decoder)
  {
    g_printerr ("One element could not be created. Exiting.\n");
    return nullptr;
  }
  g_object_set (G_OBJECT (source), "location", uri, nullptr);

  gst_bin_add_many (GST_BIN (bin), source, rtph264depayer, h264parser, decoder, nullptr);

  // source -> rtph264depayer
  GstPad* source_srcpad = gst_element_get_request_pad (source, "stream_0");
  if (!source_srcpad) {
    g_printerr ("rtspsrc source request src pad failed. Exiting.\n");
    return -1;
  }

  GstPad *rtph264depayer_sinkpad = gst_element_get_static_pad (rtph264depayer, "sink");
  if (!rtph264depayer_sinkpad) {
    g_printerr ("Failed to get sink pad of h264parser. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (source_srcpad, rtph264depayer_sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link source to rtph264depay. Exiting.\n");
    return -1;
  }

  // rtph264depay -> h264parse -> decoder
  gst_element_link_many (rtph264depayer, h264parser, decoder, nullptr);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return nullptr;
  }

  GstPad *srcpad = gst_element_get_static_pad (decoder, "src");
  if (!srcpad) {
    g_printerr ("Failed to get src pad of source bin. Exiting.\n");
    return nullptr;
  }
  GstPad *bin_ghost_pad = gst_element_get_static_pad (bin, "src");
  if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
        srcpad)) {
    g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
  }

  return bin;
}



static void printUsage(const char* cmd) {
    g_printerr ("\tUsage: %s -c dstest_segmentation_config_industrial.txt -b tatch-size -h img_out_height -w img_out_width -d image_dir\n", cmd);
    g_printerr ("-h: \n\timage output height \n");
    g_printerr ("-i: \n\timage output width \n");
    g_printerr ("-c: \n\tseg config file, e.g. dstest_segmentation_config_industrial.txt  \n");
    g_printerr ("-b: \n\tbatch size, this will override the value of \"baitch-size\" in config file  \n");
    g_printerr ("-d: \n\tThe image directory  \n");
}

int main(int argc, char *argv[])
{
//define the GstElement pointer
  GMainLoop *loop = nullptr;
  GstElement *pipeline = nullptr, *streammux = nullptr, *sink = nullptr, *seg = nullptr,
      *nvsegvisual = nullptr, *tiler = nullptr, *nvvidconv = nullptr,
      *parser = nullptr, *parser1 = nullptr, *source = nullptr, *encoder = nullptr,
      *nvvidconv1 = nullptr, *decoder = nullptr, *tee = nullptr, *nvdsosd = nullptr;


#ifdef PLATFORM_TEGRA
  GstElement *transform = nullptr;
#endif
  GstBus *bus = nullptr;
  guint bus_watch_id;
  GstPad *seg_src_pad = nullptr;
  guint pgie_batch_size;

  /* Check input arguments */
  if (argc < 2) {
      printUsage(argv[0]);
     return -1;
  }

    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (nullptr, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("dstest-image-decode-pipeline");

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
      g_printerr ("One element could not be created. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), streammux);

    GstPad *sinkpad, *srcpad;
    GstElement *source_bin = create_mp4_source_bin ((gchar*)(argv[1]));

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    sinkpad = gst_element_get_request_pad (streammux, "sink_0");
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);

    /* Use nvinfer to infer on batched frame. */
    seg = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

    nvsegvisual = gst_element_factory_make ("nvsegvisual", "nvsegvisual");

    // /* Use nvtiler to composite the batched frames into a 2D tiled array based
    //   * on the source of the frames. */
    // tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "transform");
#endif

    sink = gst_element_factory_make ("filesink", "file-sink");
    //sink = gst_element_factory_make ("fakesink", "fake-renderer");
    //sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    //following is for later with the option: display on screee ro goes to a file
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
    nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");

    /* Create OSD to draw on the converted RGBA buffer */
    nvdsosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
    parser = gst_element_factory_make ("jpegparse", "jpeg-parser");
    parser1 = gst_element_factory_make ("jpegparse", "jpeg-parser1");
    encoder = gst_element_factory_make ("jpegenc", "jpeg-encoder");

    //for the output file
    g_object_set (G_OBJECT (sink), "location", "./out.jpg", nullptr);

    decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");
    tee = gst_element_factory_make("tee", "tee");

    if (!seg || !nvsegvisual || !sink || !encoder || !parser ||
          !nvdsosd || !decoder || !tee || !parser1
                                || !nvvidconv || !nvvidconv1) {
      g_printerr ("Here one element could not be created. Exiting.\n");
      return -1;
    }


#ifdef PLATFORM_TEGRA
    if(!transform) {
      g_printerr ("One tegra element could not be created. Exiting.\n");
      return -1;
    }
#endif

    g_object_set (G_OBJECT (streammux), "batch-size", BATCH_SIZE, nullptr);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
        MUXER_OUTPUT_HEIGHT,
        "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, nullptr);

    /* Configure the nvinfer element using the nvinfer config file. */
    g_object_set (G_OBJECT (seg), "config-file-path", "../segmentation-infer-config.txt", nullptr);

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get (G_OBJECT (seg), "batch-size", &pgie_batch_size, nullptr);

    g_object_set (G_OBJECT (nvsegvisual), "batch-size", BATCH_SIZE, nullptr);
    g_object_set (G_OBJECT (nvsegvisual), "width", MUXER_OUTPUT_WIDTH, nullptr);
    g_object_set (G_OBJECT (nvsegvisual), "height", MUXER_OUTPUT_HEIGHT, nullptr);

    g_object_set(G_OBJECT(sink), "async", FALSE, nullptr);

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline */
    /* Add all elements into the pipeline */
#ifdef PLATFORM_TEGRA

    //no need to put the transform in since it output to a file
    gst_bin_add_many (GST_BIN (pipeline), seg, nvsegvisual, tiler,
                  nvvidconv, nvdsosd, nvvidconv1, encoder, parser1, sink, nullptr);
    /* we link the elements together
      * nvstreammux -> nvinfer -> nvsegvidsual -> nvtiler -> filesink */
    if (!gst_element_link_many (streammux, seg, nvsegvisual, tiler,
                nvvidconv, nvdsosd, nvvidconv1, encoder, parser1, sink, nullptr))
    {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

#else

    gst_bin_add_many (GST_BIN (pipeline), seg, nvsegvisual,
            nvvidconv, nvdsosd, nvvidconv1, encoder, parser1, sink, nullptr);

    /* Link the elements together
      * nvstreammux -> nvinfer -> nvsegvisual -> nvtiler -> video-renderer */

    if (!gst_element_link_many (streammux, seg, nvsegvisual, nvvidconv, nvdsosd, nvvidconv1, encoder, parser1, sink, nullptr)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

#endif

    /* Lets add probe to get informed of the meta data generated, we add probe to
      * the src pad of the nvseg element, since by that time, the buffer would have
      * had got all the segmentation metadata. */
    seg_src_pad = gst_element_get_static_pad (seg, "src");
    if (!seg_src_pad)
      g_print ("Unable to get src pad\n");
    else
      gst_pad_add_probe (seg_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
          tiler_src_pad_buffer_probe, nullptr, nullptr);
    gst_object_unref (seg_src_pad);

    /* Set the pipeline to "playing" state */
    g_print ("\nNow playing: %s", (gchar*)(argv[1]));
    g_print ("\n");
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");

    //start the main loop and perform the profile check
    // profile_start();
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pifile_outpeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);

  return 0;
}