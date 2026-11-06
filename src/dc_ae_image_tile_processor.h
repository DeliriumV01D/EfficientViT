#pragma once

#include "ImageTileProcessor.h"
#include "dc_ae.h"

#include <iostream>
#include <string>

///
class DCAEImageTileProcessor : public ImageTileProcessor {
protected:
	torch::Device Device{ torch::kCUDA };
	DCAEConfig Config;
	DCAE dcae{nullptr};
	bool ModelTraining{ false };

	virtual void ModelEval() override
	{
		if (!dcae) {
			ModelTraining = false;
			return;
		}
		ModelTraining = dcae->is_training();
		dcae->eval();
	}

	virtual void ModelTrain() override
	{
		if (dcae && ModelTraining)
			dcae->train();
	}

	virtual torch::Tensor EncodeBatch(torch::Tensor batch) override
	{
		return dcae->encode(batch).to(Device) /* config.scaling_factor.value()*/;
	}

	virtual torch::Tensor DecodeBatch(torch::Tensor batch) override
	{
		return dcae->decode(batch)/* / Config.scaling_factor.value()*/;
	}

public:
	DCAEImageTileProcessor(
		const std::string name = "dc-ae-f32c32-in-1.0",
		const std::string pretrained_path = "../DCAE/dc-ae-f32c32-in-1.0model.safetensors",
		torch::Device device = torch::kCUDA
	) {
		Device = device;
		register_activation_functions();
		register_norms();
		Config = dc_ae_f32c32(name, pretrained_path);
		//Вывод конфигурации
		std::cout << "Encoder Latent Channels: " << Config.encoder.latent_channels << std::endl;
		std::cout << "Decoder Latent Channels: " << Config.decoder.latent_channels << std::endl;
		std::cout << "Pretrained Path: " << Config.pretrained_path << std::endl;
		dcae = DCAE(Config);
		dcae->eval();
		dcae->to(device);
	}
	virtual ~DCAEImageTileProcessor() {};
};