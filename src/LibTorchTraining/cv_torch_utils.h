#pragma once

#include "TorchHeader.h"
#include <opencv2/imgproc/imgproc.hpp>

inline torch::Tensor CVMatToTorchTensor(cv::Mat img, const bool perm = true, const int normalize = 0)
{
	if (!img.isContinuous())
		img = img.clone();
	auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
	if (perm)
		tensor_image = tensor_image.permute({ 2,0,1 });			//hwc -> chw
	tensor_image.unsqueeze_(0);
	tensor_image = tensor_image.toType(c10::kFloat);
	if (normalize != 0)
		tensor_image = (tensor_image.div(255) - (normalize == 2 ? 0.5f : 0.f)) / (normalize == 2 ? 0.5f : 1.f);
	return tensor_image;		//tensor_image.clone();
}

inline cv::Mat TorchTensorToCVMat(const torch::Tensor tensor_image, const bool perm = true, const int normalize = 0)
{
	auto t = tensor_image.detach().squeeze(0).cpu();
	if (perm)
		t = t.permute({ 1, 2, 0 });			//chw -> hwc
	if (normalize != 0)
		t = (t * (normalize == 2 ? 0.5f : 1.f) + (normalize == 2 ? 0.5f : 0.f)).mul(255).clamp(0, 255);
	t = t.to(torch::kU8);
	cv::Mat result_img;
	t = t.contiguous();
	cv::Mat(static_cast<int>(t.size(0)), static_cast<int>(t.size(1)), CV_MAKETYPE(CV_8U, t.sizes().size() >= 3 ? static_cast<int>(t.size(2)) : 1), t.data_ptr()).copyTo(result_img);
	return result_img;
}

