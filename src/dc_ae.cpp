#include "dc_ae.h"

void add_to_seq_block(
	StackSequential seq,
	const std::string &block_type,
	int in_channels,
	int out_channels,
	const std::optional<std::string> &norm,
	const std::optional<std::string> &act
) {
	if (block_type == "ResBlock")
	{
		if (in_channels != out_channels)
			throw std::invalid_argument("build_block error: For ResBlock, in_channels must be equal to out_channels.");

		auto main_block = /*std::make_shared<*/ResBlock/*>*/(
			in_channels,
			out_channels,
			3/*kernel_size*/,
			1/*stride*/,
			c10::nullopt/*mid_channels*/,
			1.0/*expand_ratio*/,
			std::make_pair(true, false),
			std::make_pair("", norm.value_or("")),
			std::make_pair(act.value_or(""), "")
		);

		auto block = ResidualBlock<ResBlock, IdentityLayer>(main_block, IdentityLayer());
		seq->push_back(block);
	} else if (block_type == "EViT_GLU") {
		if (in_channels != out_channels)
			throw std::invalid_argument("build_block error: For EViT_GLU, in_channels must be equal to out_channels.");

		auto block = EfficientViTBlock(in_channels, 1.0 /*heads_ratio*/, 32/*dim*/, 4.0f/*expand_ratio*/, std::vector<int>{}/*scales*/, norm.value_or(""), act.value_or(""), std::string("LiteMLA")/*context_module*/, std::string("GLUMBConv")/*local_module*/);
		seq->push_back(block);
	} else if (block_type == "EViTS5_GLU") {
		if (in_channels != out_channels)
			throw std::invalid_argument("build_block error: For EViTS5_GLU, in_channels must be equal to out_channels.");

		auto block = EfficientViTBlock(in_channels, 1.0 /*heads_ratio*/, 32/*dim*/, 4.0f/*expand_ratio*/, std::vector<int>{5}/*scales*/, norm.value_or(""), act.value_or(""), std::string("LiteMLA")/*context_module*/, std::string("GLUMBConv")/*local_module*/);
		return seq->push_back(block);
	} else {
		throw std::invalid_argument("build_block error: block_type " + block_type + " is not supported");
	}
}



///
void add_to_seq_stage_main(
	StackSequential seq,
	int width,
	int depth,
	const std::variant<std::string, std::vector<std::string>> &block_type,
	const std::string &norm,
	const std::string &act,
	int input_width
) {

	if (std::holds_alternative<std::string>(block_type))		//Если block_type является строкой
	{
		for (int d = 0; d < depth; d++)
		{
			auto current_block_type = std::get<std::string>(block_type);
			add_to_seq_block(
				seq,
				current_block_type,
				d > 0 ? width : input_width,
				width,
				norm,
				act
			);
		}
	} else if (std::holds_alternative<std::vector<std::string>>(block_type)) {			//Если block_type является вектором строк
		const auto& block_types = std::get<std::vector<std::string>>(block_type);
		if (depth != static_cast<int>(block_types.size()))
			throw std::invalid_argument("add_to_seq_stage_main error: If block_type is a list, its length must be equal to depth.");

		for (int d = 0; d < depth; d++)
		{
			auto current_block_type = block_types[d];
			add_to_seq_block(
				seq,
				current_block_type,
				d > 0 ? width : input_width,
				width,
				norm,
				act
			);
		}
	} else {
		throw std::invalid_argument("add_to_seq_stage_main error: block_type must be a string or a list of strings.");
	}
}


///
void add_to_seq_downsample_block(
	StackSequential seq,
	const std::string &block_type,
	int in_channels,
	int out_channels,
	const std::optional<std::string> &shortcut)
{
	if (block_type == "Conv")
	{
		auto block_module = ConvLayer(
			in_channels,
			out_channels,
			3, // kernel_size
			2, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			true /*use_bias*/,
			0 /*dropuot*/,
			"",	/*norm*/
			""	/*act*/
		);

		if (shortcut.has_value())
		{
			if (shortcut.value() == "averaging")
			{
				auto shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, 2);
				seq->push_back(ResidualBlock<ConvLayer, PixelUnshuffleChannelAveragingDownSampleLayer>(
					block_module,
					shortcut_block
				));
			} else {
				throw std::invalid_argument("build_downsample_block error: shortcut " + *shortcut + " is not supported for downsample");
			}
		} else {
			seq->push_back(block_module);
		}
	} else if (block_type == "ConvPixelUnshuffle") {
		auto block_module = ConvPixelUnshuffleDownSampleLayer(in_channels, out_channels, 3, 2);

		if (shortcut.has_value())
		{
			if (shortcut.value() == "averaging")
			{
				auto shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, 2);
				seq->push_back(
					ResidualBlock<ConvPixelUnshuffleDownSampleLayer, PixelUnshuffleChannelAveragingDownSampleLayer>(
						block_module,
						shortcut_block
					));
			} else {
				throw std::invalid_argument("build_downsample_block error: shortcut " + *shortcut + " is not supported for downsample");
			}
		} else {
			seq->push_back(block_module);
		}
	} else {
		throw std::invalid_argument("build_downsample_block error: block_type " + block_type + " is not supported for downsampling");
	}
}


///
void add_to_seq_upsample_block(
	StackSequential seq,
	const std::string &block_type,
	int in_channels,
	int out_channels,
	const std::optional<std::string> &shortcut
) {

	if (block_type == "ConvPixelShuffle")
	{
		auto cpsus_layer = ConvPixelShuffleUpSampleLayer(in_channels, out_channels, 3, 2);

		if (shortcut.has_value())
		{
			if (shortcut.value() == "duplicating")
			{
				auto shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, 2);
				seq->push_back(
					ResidualBlock<ConvPixelShuffleUpSampleLayer, ChannelDuplicatingPixelUnshuffleUpSampleLayer>(
						cpsus_layer,
						shortcut_block
					)/*.ptr()*/
				);
			} else {
				throw std::invalid_argument("add_to_seq_upsample_block error: shortcut " + *shortcut + " is not supported for upsample");
			}
		} else {
			seq->push_back(cpsus_layer);
		}
	} else if (block_type == "InterpolateConv") {
		auto icus_layer = InterpolateConvUpSampleLayer(in_channels, out_channels, 3, 2);

		if (shortcut.has_value())
		{
			if (shortcut.value() == "duplicating")
			{
				auto shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, 2);
				seq->push_back(
					ResidualBlock<InterpolateConvUpSampleLayer, ChannelDuplicatingPixelUnshuffleUpSampleLayer>(
						icus_layer,
						shortcut_block
					)/*.ptr()*/
				);
			} else {
				throw std::invalid_argument("add_to_seq_upsample_block error: shortcut " + *shortcut + " is not supported for upsample");
			}
		} else {
			seq->push_back(icus_layer);
		}
	} else {
		throw std::invalid_argument("add_to_seq_upsample_block error: block_type " + block_type + " is not supported for upsampling");
	}
}


///
StackSequential build_encoder_project_in_block(
	int in_channels,
	int out_channels,
	int factor,
	const std::string &downsample_block_type
) {
	StackSequential seq;

	if (factor == 1)
	{
		seq->push_back(ConvLayer(
			in_channels,
			out_channels,
			3, /*kernel_size*/
			1, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			true, /*use_bias*/
			0, /*dropout*/
			"",
			""
		));
	} else if (factor == 2) {
		add_to_seq_downsample_block(seq, downsample_block_type, in_channels, out_channels, std::nullopt);
	} else {
		throw std::invalid_argument("build_encoder_project_in_block error: downsample factor " + std::to_string(factor) + " is not supported for encoder project in");
	}

	return seq;
}


///
StackSequential build_encoder_project_out_block(
	const int in_channels,
	const int out_channels,
	const std::string &norm,
	const std::string &act,
	const std::optional<std::string> &shortcut
) {
	StackSequential seq;
	std::shared_ptr<torch::nn::Module> norm_module = build_norm(norm, /*?*/in_channels/*?*/);
	if (norm_module != nullptr)
		add_to_seq_norm(seq, norm_module, norm);		//seq->push_back(norm_module.ptr());

	std::shared_ptr<torch::nn::Module> act_module = build_act(act);
	if (act_module != nullptr)
		add_to_seq_act(seq, act_module, act);		//seq->push_back(act_module);

	seq->push_back(
		ConvLayer(
			in_channels,
			out_channels,
			3, /*kernel_size*/
			1, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			true, /*use_bias*/
			0, /*dropout*/
			"",
			""
		)
	);

	StackSequential result;
	if (shortcut)
	{
		if (*shortcut == "averaging")
		{
			auto shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, 1);
			result->push_back(ResidualBlock<StackSequential, PixelUnshuffleChannelAveragingDownSampleLayer>(
				seq,
				shortcut_block
			));
		} else {
			throw std::invalid_argument("build_downsample_block error: shortcut " + *shortcut + " is not supported for downsample");
		}
	} else {
		result = seq;
	}

	return result;
}


///
StackSequential build_decoder_project_in_block(
	int in_channels,
	int out_channels,
	const std::optional<std::string> &shortcut
) {
	StackSequential seq;
	auto block_module = ConvLayer(
		in_channels,
		out_channels,
		3, /*kernel_size*/
		1, /*stride*/
		1, /*dilation*/
		1, /*groups*/
		true, /*use_bias*/
		0, /*dropout*/
		"",
		""
	);

	if (shortcut.has_value())
	{
		if (shortcut.value() == "duplicating")
		{
			auto shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, 1);
			seq->push_back(ResidualBlock<ConvLayer, ChannelDuplicatingPixelUnshuffleUpSampleLayer>(
				block_module,
				shortcut_block
			));
		} else {
			throw std::invalid_argument("build_decoder_project_in_block error: shortcut " + *shortcut + " is not supported for decoder project in");
		}
	} else {
		seq->push_back(block_module);
	}

	return seq;
}


///
StackSequential build_decoder_project_out_block(
	const int in_channels,
	const int out_channels,
	const int factor,
	const std::string &upsample_block_type,
	const std::string &norm,
	const std::string &act
) {
	StackSequential seq;
	std::shared_ptr<torch::nn::Module> norm_module = build_norm(norm, in_channels);
	if (norm_module != nullptr)
		add_to_seq_norm(seq, norm_module, norm);		//seq->push_back(norm_module);

	std::shared_ptr<torch::nn::Module> act_module = build_act(act);
	if (act_module != nullptr)
		add_to_seq_act(seq, act_module, act);		//seq->push_back(act_module);

	if (factor == 1)
	{
		seq->push_back(
			ConvLayer(
				in_channels,
				out_channels,
				3, /*kernel_size*/
				1, /*stride*/
				1, /*dilation*/
				1, /*groups*/
				true, /*use_bias*/
				0, /*dropout*/
				"",
				""
			)
		);
	} else if (factor == 2) {
		add_to_seq_upsample_block(seq, upsample_block_type, in_channels, out_channels, std::nullopt);		//seq->push_back(build_upsample_block(upsample_block_type, in_channels, out_channels, std::nullopt));
	} else {
		throw std::invalid_argument("build_decoder_project_out_block error: upsample factor " + std::to_string(factor) + " is not supported for decoder project out");
	}

	return seq;
}




///
DCAEConfig dc_ae_f32c32(const std::string &name, const std::string &pretrained_path)
{
	DCAEConfig cfg;

	if (name == "dc-ae-f32c32-in-1.0" || name == "dc-ae-f32c32-mix-1.0")
	{
		cfg.encoder.latent_channels = 32;
		cfg.encoder.block_type = std::vector<std::string>{ "ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU" };
		cfg.encoder.width_list = { 128, 256, 512, 512, 1024, 1024 };
		cfg.encoder.depth_list = { 0, 4, 8, 2, 2, 2 };
		cfg.decoder.block_type = std::vector<std::string>{ "ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU" };
		cfg.decoder.width_list = { 128, 256, 512, 512, 1024, 1024 };
		cfg.decoder.depth_list = { 0, 5, 10, 2, 2, 2 };
		cfg.decoder.norm = std::vector<std::string>{ "bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d" };
		cfg.decoder.act = std::vector<std::string>{ "relu", "relu", "relu", "silu", "silu", "silu" };
		cfg.scaling_factor = 0.3189;
	} else if (name == "dc-ae-f32c32-sana-1.0") {
		cfg.encoder.latent_channels = 32;
		cfg.encoder.block_type = std::vector<std::string>{ "ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU" };
		cfg.encoder.width_list = { 128, 256, 512, 512, 1024, 1024 };
		cfg.encoder.depth_list = { 2, 2, 2, 3, 3, 3 };

		cfg.decoder.latent_channels = 32;
		cfg.decoder.block_type = std::vector<std::string>{ "ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU" };
		cfg.decoder.width_list = { 128, 256, 512, 512, 1024, 1024 };
		cfg.decoder.depth_list = { 3, 3, 3, 3, 3, 3 };
		cfg.decoder.norm = { "trms2d" };
		cfg.decoder.act = { "silu" };
		cfg.scaling_factor = 0.41407;
		cfg.decoder.upsample_block_type = "InterpolateConv";
	} else {
		throw std::runtime_error("NotImplementedError");
	}

	cfg.pretrained_path = pretrained_path;
	return cfg;
}