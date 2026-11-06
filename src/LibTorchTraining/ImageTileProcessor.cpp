#include "ImageTileProcessor.h"

///поделить входное изображение на тайлы ориентируясь на размеры входа сети
std::vector <cv::Rect> ImageTileProcessor :: GetROI(const cv::Size &img_size, const cv::Size &net_input_size, const float &overlap)
{
	std::vector <cv::Rect> result;
	//Tiles do not fit into an image smaller than the network input - return empty ROI list
	if (img_size.width < net_input_size.width || img_size.height < net_input_size.height)
		return result;
	cv::Rect rect(0, 0, net_input_size.width, net_input_size.height);
	int nw = static_cast<int>((img_size.width - net_input_size.width * overlap) / (net_input_size.width * (1. - overlap)));
	int nh = static_cast<int>((img_size.height - net_input_size.height * overlap) / (net_input_size.height * (1. - overlap)));

	int hgap = static_cast<int>(img_size.width - net_input_size.width * overlap - nw * net_input_size.width * (1. - overlap));
	int vgap = static_cast<int>(img_size.height - net_input_size.height * overlap - nh * net_input_size.height * (1. - overlap));

	for (int i = 0; i < nw + (hgap > 0 ? 1 : 0); i++)
		for (int j = 0; j < nh + (vgap > 0 ? 1 : 0); j++)
		{
			if (i != nw)
				rect.x = static_cast<int>(i * net_input_size.width * (1. - overlap));
			else
				rect.x = static_cast<int>(img_size.width - net_input_size.width);

			if (j != nh)
				rect.y = static_cast<int>(j * net_input_size.height * (1. - overlap));
			else
				rect.y = static_cast<int>(img_size.height - net_input_size.height);

			result.push_back(rect);
		}
	return result;
}			//GetROI


///Получить результат применения EncodeBatch к изображению
torch::Tensor ImageTileProcessor :: Encode(
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
) {
	ModelEval();
	float overlap = 0;
	// argmax collapses the class dimension into a single channel
	const int out_ch = need_argmax ? 1 : output_channels;
	int nw = static_cast<int>((in.cols - input_size.width * overlap) / (input_size.width * (1. - overlap)));
	int nh = static_cast<int>((in.rows - input_size.height * overlap) / (input_size.height * (1. - overlap)));
	// GetROI adds shifted edge tiles when the image is not an exact multiple of the tile size,
	// so the result grid must cover them too (otherwise index_put_ writes out of bounds)
	int hgap = static_cast<int>(in.cols - input_size.width * overlap - nw * input_size.width * (1. - overlap));
	int vgap = static_cast<int>(in.rows - input_size.height * overlap - nh * input_size.height * (1. - overlap));
	int ncols = nw + (hgap > 0 ? 1 : 0);
	int nrows = nh + (vgap > 0 ? 1 : 0);
	cv::Size out(output_size.width * ncols, output_size.height * nrows);
	torch::Tensor result = torch::zeros({ 1, out_ch, out.height, out.width }, torch::dtype(torch::kFloat32).device(device));
	//1. поделить входное изображение на тайлы ориентируясь на размеры входа сети
	//2. упаковать тайлы в батчи
	//3. прогнать батчи через EncodeBatch
	//4. собрать выходной тензор
	std::vector <cv::Rect> roi = GetROI(in.size(), input_size, 0);
	// Image smaller than the network input - nothing to process (see GetROI)
	if (roi.empty()) { ModelTrain(); return result; }
	torch::Tensor batch = torch::zeros({ batch_size, input_channels, input_size.height, input_size.width }, torch::dtype(torch::kFloat32).device(device));

	int ib = 0;
	for (int i = 0; i < roi.size(); i++)
	{
		ib = i % batch_size;
		if (ib == 0 && i != 0)
		{
			//батч готов - отправляем EncodeBatch, а результат помещаем по roi в результирующее изображение циклом по батчу
			torch::Tensor prediction = EncodeBatch(batch);		//prediction: [bs, ncl, h, w] CUDAFloatType
			if (need_argmax)
				prediction = torch::argmax(prediction, 1, true); // [bs, 1, h, w]
			//добавить все кусочки из батча на соответствующие им позиции в результирующем тензоре
			for (int jb = i - batch_size; jb < i; jb++)
			{
				//Вычисляем соответствующий регион в выходном тензоре
				int out_y = float(roi[jb].y) * output_size.height / input_size.height;
				int out_x = float(roi[jb].x) * output_size.width / input_size.width;
				auto tile_pred = prediction[jb - i + batch_size].slice(1, 0, output_size.height).slice(2, 0, output_size.width);
				result.index_put_({ 0, torch::indexing::Slice(),
					torch::indexing::Slice(out_y, out_y + output_size.height),
					torch::indexing::Slice(out_x, out_x + output_size.width) },
					tile_pred
				);
			}
		}
		//добавить очередной i-й тайл на ib-ое место в батч
		batch = batch.index_put_({ ib, torch::indexing::Slice() }, CVMatToTorchTensor(in(roi[i]), perm, norm)/*.squeeze(0)*/.to(device));
	}		//for i
	//последний батч неполный - padding или если всего с изображения не набралось даже на один батч
	torch::Tensor prediction = EncodeBatch(batch);		//prediction: [bs, ncl, h, w] CUDAFloatType
	if (need_argmax)
		prediction = torch::argmax(prediction, 1, true); // [bs, 1, h, w]
	//добавить все кусочки из батча на соответствующие им позиции в результирующем тензоре
	for (int jb = static_cast<int>(roi.size()) - ib - 1; jb < static_cast<int>(roi.size()); jb++)
	{
		int out_y = float(roi[jb].y) * output_size.height / input_size.height;
		int out_x = float(roi[jb].x) * output_size.width / input_size.width;
		auto tile_pred = prediction[jb - static_cast<int>(roi.size()) + ib + 1].slice(1, 0, output_size.height).slice(2, 0, output_size.width);
		result.index_put_({ 0, torch::indexing::Slice(),
			torch::indexing::Slice(out_y, out_y + output_size.height),
			torch::indexing::Slice(out_x, out_x + output_size.width) },
			tile_pred
		);
	}
	
	//Восстанавливаем исходное состояние модели
	ModelTrain();
	return result;
}			//Encode


///Получить результат применения DecodeBatch к тензору 
cv::Mat ImageTileProcessor :: Decode(
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
) {
	ModelEval();
	float overlap = 0;
	int nw = static_cast<int>((in.sizes()[3] - input_size.width * overlap) / (input_size.width * (1. - overlap)));
	int nh = static_cast<int>((in.sizes()[2] - input_size.height * overlap) / (input_size.height * (1. - overlap)));
	const int out_ch = need_argmax ? 1 : output_channels;
	cv::Size out(output_size.width * nw, output_size.height * nh);
	cv::Mat result(out, CV_MAKETYPE(CV_8U, out_ch));
	//1. поделить входное изображение на тайлы ориентируясь на размеры входа сети
	//2. упаковать тайлы в батчи
	//3. прогнать батчи через DecodeBatch
	//4. собрать выходной тензор
	std::vector <cv::Rect> roi = GetROI(result.size(), output_size, 0);
	if (roi.empty()) { ModelTrain(); return result; }
	torch::Tensor batch = torch::zeros({ batch_size, input_channels, input_size.height, input_size.width }, torch::dtype(torch::kFloat32).device(device));

	int ib = 0;
	for (int i = 0; i < roi.size(); i++)
	{
		ib = i % batch_size;
		if (ib == 0 && i != 0)
		{
			//батч готов - отправляем DecodeBatch, а результат помещаем по roi в результирующее изображение циклом по батчу
			torch::Tensor prediction = DecodeBatch(batch);		//prediction: [bs, ncl, h, w] CUDAFloatType
			if (need_argmax)
				prediction = torch::argmax(prediction, 1, true); // [bs, 1, h, w]
			//добавить все кусочки из батча на соответствующие им позиции в результирующем тензоре
			//#pragma omp parallel for
			for (int jb = i - batch_size; jb < i; jb++)
				TorchTensorToCVMat(prediction[jb - i + batch_size]/*.unsqueeze(0)*//*.to(torch::kInt)*/, perm, norm).copyTo(result(roi[jb]));
		}
		//добавить очередной i-й тайл на ib-ое место в батч
		//Вычисляем соответствующий регион во входном тензоре
		int out_y = float(roi[i].y) * input_size.height / output_size.height;
		int out_x = float(roi[i].x) * input_size.width / output_size.width;
		batch = batch.index_put_({ ib, torch::indexing::Slice() },
			in.index({ 0, torch::indexing::Slice(),
				torch::indexing::Slice(out_y, out_y + input_size.height),
				torch::indexing::Slice(out_x, out_x + input_size.width) }
			).to(device)		//input tensor may live on another device than the batch buffer
		);
	}		//for i
	//последний батч неполный - padding или если всего с изображения не набралось даже на один батч
	torch::Tensor prediction = DecodeBatch(batch);		//prediction: [bs, ncl, h, w] CUDAFloatType
	if (need_argmax)
		prediction = torch::argmax(prediction, 1, true); // [bs, 1, h, w]
	//добавить все кусочки из батча на соответствующие им позиции в результирующем тензоре
	//#pragma omp parallel for
	for (int jb = static_cast<int>(roi.size()) - ib - 1; jb < static_cast<int>(roi.size()); jb++)
		TorchTensorToCVMat(prediction[jb - static_cast<int>(roi.size()) + ib + 1]/*.unsqueeze(0)*//*.to(torch::kInt)*/, perm, norm).copyTo(result(roi[jb]));

	//Восстанавливаем исходное состояние модели
	ModelTrain();
	return result;
}			//Decode

