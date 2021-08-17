################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################


APP:= gstreamer-rtsp-app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

BASE_DEEPSTREAM_DIR?=/opt/nvidia/deepstream/deepstream/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif
CC:= g++

SRCS:= $(wildcard *.cxx)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.cxx=.o)

CFLAGS+= -g -I$(BASE_DEEPSTREAM_DIR)sources/includes -I /usr/local/cuda/include

CFLAGS+= `pkg-config --cflags $(PKGS)`

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
	   -L/usr/local/cuda/lib64/ -lcudart \
	   -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform \
       -Wl,-rpath,$(LIB_INSTALL_DIR)


ifeq ($(TARGET_DEVICE),aarch64)
	PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv4
else
	PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv gstreamer-rtsp-1.0 gstreamer-rtsp-server-1.0 
endif

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(APP)

%.o: %.cxx $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)

