/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include <map>
#include "yololayer.h"


float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Yolo::Detection& a, Yolo::Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {

        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

template<typename T>
T clamp(T x, T min, T max)
{
	return std::max(std::min(x, max), min);
}

/* This is a sample bounding box parsing function for the sample YoloV7 detector model */
static bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    std::vector<Yolo::Detection> res;
    float kCONF_THRESH = 0.5;
    float kNMS_THRESH = 0.45;

    nms(res, (float*)(outputLayersInfo[0].buffer), kCONF_THRESH, kNMS_THRESH);
    
    for(auto& r : res) {
	    NvDsInferParseObjectInfo oinfo;        
        
	    oinfo.classId = r.class_id;

        float netW = 1920.f;
        float netH = 1080.f;
        float x0 = r.bbox[0]-r.bbox[2]*0.5f;
        float y0 = r.bbox[1]-r.bbox[3]*0.5f;
        float width = r.bbox[2];
        float height = r.bbox[3];
	    oinfo.left    = clamp(x0, 0.f, netW);
	    oinfo.top     = clamp(y0, 0.f, netH);
	    oinfo.width   = clamp(width, 0.f, netW);
	    oinfo.height  = clamp(height, 0.f, netH);
        oinfo.height /= 2;
	    oinfo.detectionConfidence = r.conf;
	    objectList.push_back(oinfo);
    }
    
    return true;
}

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseYoloV5(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);



static std::vector < std::vector< std:: string > > labels { { "green", "other", "red"} };

static bool NvDsInferParseMobilenetv3(
    std::vector< NvDsInferLayerInfo > const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector< NvDsInferAttribute > &attrList,
    std::string &descString)
{
    /* Get the number of attributes supported by the classifier. */
    unsigned int numAttributes = outputLayersInfo.size();
    
    /* Iterate through all the output coverage layers of the classifier.
    */
    for (unsigned int l = 0; l < numAttributes; l++)
    {
        /* outputCoverageBuffer for classifiers is usually a softmax layer.
         * The layer is an array of probabilities of the object belonging
         * to each class with each probability being in the range [0,1] and
         * sum all probabilities will be 1.
         */
        NvDsInferDimsCHW dims;
        // NvDsInferDimsCHW
        getDimsCHWFromDims(dims, outputLayersInfo[l].inferDims);
        
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer = (float *)outputLayersInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;
        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++)
        {
            float probability = outputCoverageBuffer[c];
            if (probability > classifierThreshold
                    && probability > maxProbability)
            {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
                
            }
        }
        if (attrFound)
        {
            if (labels.size() > attr.attributeIndex &&
                    attr.attributeValue < labels[attr.attributeIndex].size())
                attr.attributeLabel = strdup(const_cast<char*>(labels[attr.attributeIndex][attr.attributeValue].c_str()));
            else
                attr.attributeLabel = nullptr;
            
            attrList.push_back(attr);
            if (attr.attributeLabel)
            {
                descString.append(attr.attributeLabel).append(" ");
            }
            std::cout << ">>>>> " << std::string(attr.attributeLabel) << std::endl;
        }
    }

    return true;
}

extern "C" bool NvDsInferClassiferParseCustomMobilenetv3(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString)
{
    return NvDsInferParseMobilenetv3(
        outputLayersInfo, networkInfo, classifierThreshold, attrList, descString);
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseCustomMobilenetv3);
