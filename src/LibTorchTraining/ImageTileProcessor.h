#pragma once

#include "TorchHeader.h"
#include "cv_torch_utils.h"

#include <opencv2/imgproc/imgproc.hpp>

///
class ImageTileProcessor {
protected:
	///Split the image into tiles of the given size; returns an empty vector if the image is smaller than a tile
	std::vector <cv::Rect> GetROI(const cv::Size &img_size, const cv::Size &sz, const float &overlap);

	virtual void ModelEval() = 0;
	//{
	//	bool model_training = Model->is_training();
	//	if (model_training)
	//	Model->eval();
	//}

	virtual void ModelTrain() = 0;
	//{
	//	if (model_training)
	//		Model->train();
	//}

	virtual torch::Tensor EncodeBatch(torch::Tensor batch) = 0;
	virtual torch::Tensor DecodeBatch(torch::Tensor batch) = 0;
public:
	virtual ~ImageTileProcessor() {}

	///Получить результат применения EncodeBatch к изображению
	torch::Tensor Encode(
		cv::Mat in,
		const int batch_size,
		const cv::Size &input_size,
		const int input_channels,
		const cv::Size &output_size,
		const int output_channels,
		const torch::Device &device,
		const bool perm,
		const int norm,
		const bool need_argmax
	);

	///Получить результат применения DecodeBatch к тензору 
	cv::Mat Decode(
		torch::Tensor in,
		const int batch_size,
		const cv::Size &input_size,
		const int input_channels,
		const cv::Size &output_size,
		const int output_channels,
		const torch::Device &device,
		const bool perm,
		const int norm,
		const bool need_argmax
	);
};				///ImageTileProcessor