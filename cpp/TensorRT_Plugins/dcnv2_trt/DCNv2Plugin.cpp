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

#include "DCNv2Plugin.h"
#include "NvInfer.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cudnn.h>
#include <sstream>
#include <cuda.h>
#include <cassert>



namespace nvinfer1
{

namespace
{
static const char* DCNV2_NAME{"DCNv2_TRT"};
static const char* DCNV2_VERSION{"1"};

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer)
{
    T val{};
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}
}


// Static class fields initialization
PluginFieldCollection DCNv2PluginDynamicCreator::mFC{};
std::vector<PluginField> DCNv2PluginDynamicCreator::mPluginAttributes;

DCNv2PluginDynamic::DCNv2PluginDynamic(const void* data, size_t length, 
	const std::string& name)
	: mLayerName(name)
{
	const char* d = reinterpret_cast<const char*>(data), *a = d;

	for (int i = 0 ; i < DILATION_DIM ; i++)
		mParam.dilation.push_back(read<int>(d));
	for (int i = 0 ; i < PADDING_DIM ; i++)
		mParam.padding.push_back(read<int>(d));
	for (int i = 0 ; i < STRIDE_DIM ; i++)
		mParam.stride.push_back(read<int>(d));
	mParam.deformable_groups = read<int>(d);
}

DCNv2PluginDynamic::DCNv2PluginDynamic(DCNv2Parameters param, const std::string& name)
	: mParam{param}, mLayerName(name)
{
}

DCNv2PluginDynamic::~DCNv2PluginDynamic(){}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* DCNv2PluginDynamic::clone() const TRT_NOEXCEPT
{
	auto p = new DCNv2PluginDynamic(mParam, mLayerName);
	p->setPluginNamespace(mNamespace.c_str());

	return p;
}

DimsExprs DCNv2PluginDynamic::getOutputDimensions(int outputIndex, const DimsExprs* inputs,
	int nbInputs, IExprBuilder& exprBuilder) TRT_NOEXCEPT
{
	// Validate input arguments
	assert(nbInputs == 4);

	DimsExprs ret;
	ret.nbDims = 4;
	ret.d[0] = inputs[0].d[0];
	ret.d[1] = inputs[2].d[0];
	ret.d[2] = inputs[0].d[2];
	ret.d[3] = inputs[0].d[3];
	return ret;
}

bool DCNv2PluginDynamic::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs,
	int nbOutputs) TRT_NOEXCEPT
{
	// Validate input arguments
	assert(nbInputs == 4);
	assert(nbOutputs == 1);

	const PluginTensorDesc& desc = inOut[pos];
		// std::cout << "<<<<<<<<<" << (inOut[1].type == DataType::kFLOAT)<< "|" << (inOut[1].type == DataType::kHALF)<< "|"
		// << (inOut[1].type == DataType::kINT8)<< "|" << (inOut[1].type == DataType::kINT32)<<   std::endl;
	// return ( (desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) 
	// 	&& desc.format == TensorFormat::kLINEAR
	// 	&& inOut[0].type == inOut[1].type
	// 	&& inOut[1].type == inOut[2].type
	// 	&& inOut[2].type == inOut[3].type
	// 	&& inOut[3].type == inOut[4].type);


	return ((desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) && desc.format == TensorFormat::kLINEAR);

}

void DCNv2PluginDynamic::configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
	const DynamicPluginTensorDesc* outputs, int nbOutputs) TRT_NOEXCEPT
{
	// Validate input arguments
	assert(nbInputs == 4);
	assert(nbOutputs == 1);

	mType = inputs[0].desc.type;
	input1_shape = inputs[0].desc.dims;
	input2_shape = inputs[1].desc.dims;
	weights_shape = inputs[2].desc.dims;
	output_shape = outputs[0].desc.dims;
}

size_t DCNv2PluginDynamic::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
	const PluginTensorDesc* outputs, int nbOutputs) const TRT_NOEXCEPT
{
	int kernel_size = weights_shape.d[3];
	int deformable_groups = 1;

	size_t maskSize = static_cast<size_t>(input1_shape.d[H_DIM] * input1_shape.d[W_DIM] *
		kernel_size * kernel_size * deformable_groups);
	size_t im2colSize = static_cast<size_t>(input1_shape.d[C_DIM] * kernel_size *
		kernel_size * output_shape.d[H_DIM] * output_shape.d[W_DIM]);

	int maxBatchSize = 100;
	return (im2colSize + maskSize) * maxBatchSize * (mType == DataType::kFLOAT ? 4 : 2);
}

int DCNv2PluginDynamic::enqueue(const PluginTensorDesc* inputDesc,
	const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs,
	void* workspace, cudaStream_t stream) TRT_NOEXCEPT
{
	enqueue_call(inputs, outputs, workspace, stream, mType, input1_shape, input2_shape,
		weights_shape, output_shape, mType, cublasHandle_, mParam);
	return 0;
}

// IPluginV2Ext Methods
DataType DCNv2PluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
{
	assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
	return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin
// the access to some context resource
void DCNv2PluginDynamic::attachToContext(cudnnContext* cudnnContext,
	cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
{
	cublasHandle_ = cublasContext;
	cudnnHandle_ = cudnnContext;
}

// Detach the plugin object from its execution context
void DCNv2PluginDynamic::detachFromContext() TRT_NOEXCEPT {}

// IPluginV2 Methods
const char* DCNv2PluginDynamic::getPluginType() const TRT_NOEXCEPT
{
	return DCNV2_NAME;
}

const char* DCNv2PluginDynamic::getPluginVersion() const TRT_NOEXCEPT
{
	return DCNV2_VERSION;
}

int DCNv2PluginDynamic::getNbOutputs() const TRT_NOEXCEPT
{
	return 1;
}

int DCNv2PluginDynamic::initialize() TRT_NOEXCEPT
{
	return 0;
}

void DCNv2PluginDynamic::terminate() TRT_NOEXCEPT {}

size_t DCNv2PluginDynamic::getSerializationSize() const TRT_NOEXCEPT
{
	return DILATION_DIM * sizeof(int)	// dilation
		+ PADDING_DIM * sizeof(int)	// padding
		+ STRIDE_DIM * sizeof(int)	// stride
		+ 1 * sizeof(int);				// deformable
}

void DCNv2PluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT
{
	char* d = reinterpret_cast<char*>(buffer), *a = d;
	for (int i = 0 ; i < DILATION_DIM ; i++)
		write(d, mParam.dilation[i]);
	for (int i = 0 ; i < PADDING_DIM ; i++)
		write(d, mParam.padding[i]);
	for (int i = 0 ; i < STRIDE_DIM ; i++)
		write(d, mParam.stride[i]);
	write(d, mParam.deformable_groups);
}

void DCNv2PluginDynamic::destroy() TRT_NOEXCEPT
{
	delete this;
}

void DCNv2PluginDynamic::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
{
	mNamespace = pluginNamespace;
}

const char* DCNv2PluginDynamic::getPluginNamespace() const TRT_NOEXCEPT
{
	return mNamespace.c_str();
}



DCNv2PluginDynamicCreator::DCNv2PluginDynamicCreator()
{
	mPluginAttributes.emplace_back(PluginField("dilation", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("padding", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("stride", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("deformable_groups", nullptr,
		PluginFieldType::kINT32, 1));

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

const char* DCNv2PluginDynamicCreator::getPluginName() const TRT_NOEXCEPT
{
	return DCNV2_NAME;
}

const char* DCNv2PluginDynamicCreator::getPluginVersion() const TRT_NOEXCEPT
{
	return DCNV2_VERSION;
}

const PluginFieldCollection* DCNv2PluginDynamicCreator::getFieldNames() TRT_NOEXCEPT
{
	return &mFC;
}

IPluginV2* DCNv2PluginDynamicCreator::createPlugin(const char* name,
	const PluginFieldCollection* fc) TRT_NOEXCEPT
{
	std::vector<int> dilation;
	std::vector<int> padding;
	std::vector<int> stride;
	int deformable_groups = 1;
	const PluginField* fields = fc->fields;

	for (int i = 0 ; i < fc->nbFields ; i++)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "dilation"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			dilation.reserve(size);
			const auto* d = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				dilation.push_back(*d);
				d++;
			}
		}
		else if (!strcmp(attrName, "padding"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			padding.reserve(size);
			const auto* p = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				padding.push_back(*p);
				p++;
			}
		}
		else if (!strcmp(attrName, "stride"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			stride.reserve(size);
			const auto* s = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				stride.push_back(*s);
				s++;
			}
		}
		else if (!strcmp(attrName, "deformable_groups"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			deformable_groups = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
	}

	DCNv2Parameters dcnv2Params;
	dcnv2Params.dilation = dilation;
	dcnv2Params.padding = padding;
	dcnv2Params.stride = stride;
	dcnv2Params.deformable_groups = deformable_groups;

	DCNv2PluginDynamic* p = new DCNv2PluginDynamic(dcnv2Params, name);
	return p;
}

IPluginV2* DCNv2PluginDynamicCreator::deserializePlugin(const char* name,
	const void* serialData, size_t serialLength) TRT_NOEXCEPT
{
	// This object will be deleted when the network is destroyed, which will
	// call DCNv2PluginDynamic::destroy()
	return new DCNv2PluginDynamic(serialData, serialLength, name);
}

void DCNv2PluginDynamicCreator::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
{
	mNamespace = pluginNamespace;
}

const char* DCNv2PluginDynamicCreator::getPluginNamespace() const TRT_NOEXCEPT
{
	return mNamespace.c_str();
}

inline unsigned int getElementSize(DataType t)
{
	switch (t)
	{
	case DataType::kFLOAT: return 4;
	case DataType::kHALF: return 2;
	default:break;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;	
}

} // namespace nvinfer1
