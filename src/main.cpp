#include "TorchHeader.h"

#include "act.h"
#include "norm.h"
#include "opt.h"
#include "dc_ae.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>


void test()
{
	register_activation_functions();

	auto relu_act = REGISTERED_ACT_DICT["relu"];
	if (relu_act) 
		std::cout << "ReLU activation function created." << std::endl;

	auto gelu_act = build_act("gelu");
	if (gelu_act) 
		std::cout << "GELU activation function created." << std::endl;

	auto unknown_act = build_act("unknown");
	if (!unknown_act) 
		std::cout << "Unknown activation function." << std::endl;

	register_norms();

	auto norm = build_norm("bn2d", 3);
	if (norm) 
		std::cout << "bn2d norm function created." << std::endl;

	auto norm2 = build_norm("trms2d", 10);
	if (norm2 != nullptr) {
		torch::Tensor x = torch::rand({ 3, 10, 2, 3 });
		//auto y = norm2->forward(x);
		auto y = forward_norm(norm2, x, "trms2d");
		std::cout << "Forward pass trms2d output: " << y << std::endl;
	} else {
		std::cout << "Failed to build trms2d norm" << std::endl;
	}

	auto conv = ConvLayer(3, 3, 3, 1, 1, 1, false, 0, "bn2d", "relu");
	if (conv) 
		std::cout << "conv layer created." << std::endl;

	auto up_sample = 	UpSampleLayer(torch::kBicubic,  c10::nullopt, 2., false); 
	if (up_sample) 
		std::cout << "up_sample layer created." << std::endl;

	ConvPixelUnshuffleDownSampleLayer down_sample(3, 12, 3, 2);
	if (down_sample) 
		std::cout << "down_sample layer created." << std::endl;
	down_sample->forward(torch::randn({1, 3, 28, 28})/*.to(torch::kCUDA)*/);

	PixelUnshuffleChannelAveragingDownSampleLayer down_sample2(16, 4, 2);
	if (down_sample2) 
		std::cout << "down_sample2 layer created." << std::endl;
	torch::Tensor output = down_sample2->forward(torch::randn({1, 16, 8, 8}));

	ConvPixelShuffleUpSampleLayer up_sample2(16, 4, 3, 2);
	if (up_sample2) 
		std::cout << "up_sample2 layer created." << std::endl;
	torch::Tensor output2 = up_sample2->forward(torch::randn({1, 16, 8, 8}));

	InterpolateConvUpSampleLayer up_sample3(16, 4, 3, 2, torch::kNearest);
	if (up_sample3) 
		std::cout << "up_sample3 layer created." << std::endl;
	torch::Tensor output3 = up_sample3->forward(torch::randn({1, 16, 8, 8}));

	ChannelDuplicatingPixelUnshuffleUpSampleLayer up_sample4(16, 64, 2);
	if (up_sample4) 
		std::cout << "up_sample4 layer created." << std::endl;
	torch::Tensor output4 = up_sample4->forward(torch::randn({1, 16, 8, 8}));

	LinearLayer linear_layer(10, 5, true, 0.2, "", "");
	if (linear_layer) 
		std::cout << "linear_layer created." << std::endl;
	torch::Tensor output5 = linear_layer->forward(torch::randn({32, 10}, torch::device(torch::kCPU)));

	DSConv ds_conv(3, 16, 3, 2, false);//{"bn2d", "bn2d"}, {"relu6", ""});
	if (ds_conv) 
		std::cout << "ds_conv created." << std::endl;
	torch::Tensor output6 = ds_conv->forward(torch::randn({2, 3, 24, 24}));

	//std::tuple<std::string, std::string, std::string> norm2 = {"bn2d", "bn2d", "bn2d"};
	//std::tuple<std::string, std::string, std::string> act_func2 = {"relu6", "relu6", ""};
	MBConv mb_conv(3, 16, 3, 2, c10::nullopt, 6., std::make_tuple( true, true, false ));
	if (mb_conv) 
		std::cout << "mb_conv created." << std::endl;
	torch::Tensor output7 = mb_conv->forward(torch::randn({4, 3, 24, 24}));

	//std::pair<std::string, std::string> norm3 = {"bn2d", "bn2d"};
	//std::pair<std::string, std::string> act_func3 = {"relu6", ""};
	FusedMBConv fused_mb_conv(3, 16, 3, 2, c10::nullopt, 6., 1, false);
	if (fused_mb_conv) 
		std::cout << "fused_mb_conv created." << std::endl;
	torch::Tensor output8 = fused_mb_conv->forward(torch::randn({4, 3, 24, 24}));
		
	GLUMBConv glumb_conv(3, 16, 3, 2, c10::nullopt, 6., std::make_tuple(true, true, false));
	if (glumb_conv) 
		std::cout << "glumb_conv created." << std::endl;
	torch::Tensor output9 = glumb_conv->forward(torch::randn({4, 3, 24, 24}));

	ResBlock res_block(3, 16, 3, 2, c10::nullopt, 1., std::make_pair(false, false));
	if (res_block) 
		std::cout << "res_block created." << std::endl;
	torch::Tensor output10 = res_block->forward(torch::randn({4, 3, 24, 24}));

	LiteMLA lite_mla(3, 16, 2);
	if (lite_mla) 
		std::cout << "lite_mla created." << std::endl;
	///!!!НЕТ ИНИЦИАЛИЗАЦИИ ВЕСОВ ПОЭТОМУ МУСОР на выходе
	auto output11 = lite_mla->forward(torch::randn({4, 3, 24, 24}));

	ResidualBlock<torch::nn::Conv2d, torch::nn::Conv2d> residual_block(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 1)),
			std::string("relu6"),
			std::string("bn2d"),
			std::dynamic_pointer_cast<torch::nn::Module>(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions({3})).ptr())
		);
	if (residual_block) 
		std::cout << "residual_block created." << std::endl;
	auto output14 = residual_block->forward(torch::randn({4/*bs*/, 3, 24, 24}/*, device*/));

	EfficientViTBlock efficient_vit(32,1.0f,32,4.0f, std::vector<int>{ 5 },"bn2d","relu","LiteMLA","MBConv");
	if (efficient_vit)
		std::cout << "efficient_vit created." << std::endl;
	auto output12 = efficient_vit->forward(torch::randn({ 16, 32, 24, 24 }/*, device*/));

	StackSequential stage_main_seq;
	add_to_seq_stage_main(
		stage_main_seq,
		16,
		2,
		"ResBlock", //const std::variant<std::string, std::vector<std::string>> &block_type,
		"",//const std::string & norm,
		"",//const std::string & act,
		16//int input_width
	);
	if (!stage_main_seq.is_empty())
		std::cout << "stage_main created." << std::endl;

	StackSequential downsample_block_seq;
	add_to_seq_downsample_block(downsample_block_seq, "ConvPixelUnshuffle", 16, 16, "averaging");
	if (downsample_block_seq)
		std::cout << "downsample_block created." << std::endl;

	StackSequential upsample_block_seq;
	add_to_seq_upsample_block(upsample_block_seq, "ConvPixelShuffle", 16, 16, "duplicating");
	if (upsample_block_seq)
		std::cout << "upsample_block created." << std::endl;


	StackSequential encoder_project_in_block = build_encoder_project_in_block(16, 16, 1, "ConvPixelUnshuffle");
	if (encoder_project_in_block)
		std::cout << "encoder_project_in_block created." << std::endl;

	StackSequential encoder_project_out_block = build_encoder_project_out_block(16, 16, "bn2d", "relu", "averaging");
	if (encoder_project_out_block)
		std::cout << "encoder_project_out_block created." << std::endl;

	StackSequential decoder_project_out_block = build_decoder_project_out_block(16, 16, 2, "InterpolateConv", "", "");
	if (decoder_project_out_block)
		std::cout << "decoder_project_out_block created." << std::endl;

	StackSequential block_seq;
	add_to_seq_block(block_seq, "EViT_GLU", 32, 32, "bn2d", "relu");
	if (!block_seq->is_empty())
		std::cout << "block created." << std::endl;
	auto output17 = block_seq->forward(torch::randn({ 16, 32, 24, 24 }/*, device*/));

	StackSequential decoder_project_in_block = build_decoder_project_in_block(16, 16, "duplicating");
	if (decoder_project_in_block)
		std::cout << "decoder_project_in_block created." << std::endl;
	//auto output17 = decoder_project_in_block->as<EfficientViTBlockImpl>()->forward(torch::randn({ 16, 32, 24, 24 }/*, device*/));

	//Пример использования Encoder
	EncoderConfig econfig;
	econfig.in_channels = 3;
	econfig.latent_channels = 64;
	econfig.width_list = { 64, 128, 256 };
	econfig.depth_list = { 0, 2, 2 };
	econfig.block_type = std::string("EViT_GLU");
	econfig.norm = "trms2d";
	econfig.act = "relu";
	econfig.downsample_block_type = "ConvPixelUnshuffle";
	econfig.downsample_match_channel = true;
	econfig.downsample_shortcut = "averaging";
	econfig.out_norm = ""/* = {}*/;
	econfig.out_act = ""/*= {}*/;
	econfig.out_shortcut = "averaging";
	econfig.double_latent = false;
	Encoder encoder(econfig);
	torch::Tensor input = torch::randn({ 1, 3, 256, 256 });
	torch::Tensor output18 = encoder->forward(input);
	std::cout << "Encoder output:\n" << output18.sizes() << std::endl;

	//Пример использования Decoder
	DecoderConfig dconfig;
	dconfig.in_channels = 3;
	dconfig.latent_channels = 64;
	dconfig.in_shortcut = "duplicating";
	dconfig.width_list = { 256, 128, 64 };
	dconfig.depth_list = { 0, 2, 2 };
	dconfig.block_type = "EViT_GLU";
	dconfig.norm = "trms2d";
	dconfig.act = "relu";
	dconfig.upsample_block_type = "ConvPixelShuffle";
	dconfig.upsample_match_channel = true;
	dconfig.upsample_shortcut = "duplicating";
	dconfig.out_norm = "bn2d";
	dconfig.out_act = "silu";
	Decoder decoder(dconfig);
	torch::Tensor input19 = torch::randn({ 1, 64, 8, 8 });
	torch::Tensor output19 = decoder->forward(input19);
	std::cout << "Decoder output:\n" << output19.sizes() << " " << output19.mean() << " " << output19.std() << std::endl;

	//Пример использования DCAE
	DCAEConfig dcae_config;
	dcae_config.encoder = econfig;
	dcae_config.decoder = dconfig;
	DCAE dcae(dcae_config);
	torch::Tensor input20 = torch::randn({ 1, 3, 64, 64 });
	auto output20 = dcae->forward(input20, 0);
	std::cout << "Output:\n" << std::get<0>(output20).sizes() << std::endl;
	//std::cout << std::get<0>(output20) << std::endl;
}

inline torch::Tensor CVMatToTorchTensor(cv::Mat img, const bool perm = true)
{
	if (!img.isContinuous())
		img = img.clone();
	auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
	if (perm)
		tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image.unsqueeze_(0);
	tensor_image = (tensor_image.toType(c10::kFloat).div(255) - 0.5) / 0.5;
	return tensor_image;		//tensor_image.clone();
}

inline cv::Mat TorchTensorToCVMat(const torch::Tensor tensor_image, const bool perm = true)
{
	auto t = tensor_image.detach().squeeze(0).cpu();
	if (perm)
		t = t.permute({ 1, 2, 0 });
	t = (t.mul(0.5) + 0.5).mul(255).clamp(0, 255).to(torch::kU8);
	cv::Mat result_img;
	t = t.contiguous();
	cv::Mat(static_cast<int>(t.size(0)), static_cast<int>(t.size(1)), CV_MAKETYPE(CV_8U, t.sizes().size() >= 3 ? static_cast<int>(t.size(2)) : 1), t.data_ptr()).copyTo(result_img);
	return result_img;
}

int main(int argc, const char* argv[])
{
	torch::manual_seed(42);

	//test();

	try {
		register_activation_functions();
		register_norms();
		torch::Device device = torch::kCUDA;
		std::string name = "dc-ae-f32c32-in-1.0";
		std::string pretrained_path = "D:/Delirium/PROJECTS/EfficientViT/DCAE/dc-ae-f32c32-in-1.0model.safetensors";
		DCAEConfig config = dc_ae_f32c32(name, pretrained_path);
		//Вывод конфигурации
		std::cout << "Encoder Latent Channels: " << config.encoder.latent_channels << std::endl;
		std::cout << "Decoder Latent Channels: " << config.decoder.latent_channels << std::endl;
		std::cout << "Pretrained Path: " << config.pretrained_path << std::endl;
		DCAE dcae(config);
		dcae->eval();
		dcae->to(device);


		cv::Mat img = cv::imread("../data/1.jpg", cv::IMREAD_COLOR);
		cv::resize(img, img, cv::Size(512, 512));
		torch::Tensor input = CVMatToTorchTensor(img);
		//torch::Tensor input = torch::randn({ 1, 3, 128, 128 });
		std::cout << "input: " << input.sizes() << " " << input.mean() << " " << input.std() << std::endl;
		auto latent = dcae->encode((input* config.scaling_factor.value()).to(device)) /* config.scaling_factor.value()*/;
		std::cout << /*latent[0] <<*/ "latent: " << latent.sizes() << " " << latent.mean() << " " << latent.std() << std::endl;
		auto out = dcae->decode(latent)/ config.scaling_factor.value();
		std::cout << "out: " << out.sizes() << " " << out.mean() << " " << out.std() << std::endl;
		auto img_out = TorchTensorToCVMat(out);
		cv::imshow("img", img);
		cv::imshow("img_out", img_out);
		cv::imwrite("img_out.jpg", img_out);
		cv::imwrite("input_out.jpg", TorchTensorToCVMat(input));
		cv::waitKey(0);

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}
	
	return 0;
}