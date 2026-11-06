#pragma once

#include "TorchHeader.h"

#include <unordered_map>
#include <memory>


///Example:
///LayerNorm2d model(LayerNormOptions({2, 2}).elementwise_affine(false).eps(2e-5));
class LayerNorm2dImpl : public torch::nn::LayerNormImpl {
private:
public:
	LayerNorm2dImpl(std::vector<int64_t> normalized_shape) : LayerNorm2dImpl(torch::nn::LayerNormOptions(normalized_shape)) {}
	explicit LayerNorm2dImpl(torch::nn::LayerNormOptions options) : torch::nn::LayerNormImpl(options){};
	virtual ~LayerNorm2dImpl() {}

	torch::Tensor forward(torch::Tensor x) 
	{
		auto out = x - x.mean({1}, true);
		out = out / torch::sqrt(torch::square(out).mean({1}, true) + options.eps());
		if (options.elementwise_affine())
			out = out * weight.view({1, -1, 1, 1}) + bias.view({1, -1, 1, 1});
		return out;
	}
};		//LayerNorm2dImpl
TORCH_MODULE(LayerNorm2d);


///
class TritonRMSNorm2dImpl : public torch::nn::Module {
private:
	torch::Tensor Weight;
	torch::Tensor Bias;
	float Eps;
	bool ElementwiseAffine;
	bool bias_;
public:
	explicit TritonRMSNorm2dImpl(
		int64_t num_channels,
		float eps = 1e-5,
		bool elementwise_affine = true,
		bool bias = true
	) : Eps(eps), ElementwiseAffine(elementwise_affine), bias_(bias)
	{
		if (elementwise_affine)
		{
			Weight = register_parameter("weight", torch::ones({ num_channels }));
			if (bias) {
				this->Bias = register_parameter("bias", torch::zeros({ num_channels }));
			} else {
				this->Bias = torch::Tensor();
			}
		}
		else {
			this->Weight = torch::Tensor();
			this->Bias = torch::Tensor();
		}
	}

	torch::Tensor forward(torch::Tensor x)
	{
		//Input shape: [N, C, H, W]
		auto input_dtype = x.dtype();

		//RMS over channels per spatial position
		auto variance = x.to(torch::kFloat32).pow(2).mean({ 1 }, /*keepdim=*/true); // [N,1,H,W]

		//Normalize
		x = x * torch::rsqrt(variance + Eps);

		if (ElementwiseAffine && Weight.defined())
		{
			//Convert to weight's dtype if needed
			if (Weight.dtype() == torch::kFloat16 || Weight.dtype() == torch::kBFloat16)
			{
				x = x.to(Weight.dtype());
			}

			//Reshape weight and bias for broadcasting
			auto weight = Weight.view({ 1, -1, 1, 1 });

			//Apply weight
			x = x * weight;

			//Apply bias if defined
			if (bias_ && Bias.defined())
			{
				auto bias_ = Bias.view({ 1, -1, 1, 1 });
				x = x + bias_;
			}
		} else {
			// Restore original dtype if no affine transformation
			x = x.to(input_dtype);
		}

		return x;
	}
};
TORCH_MODULE(TritonRMSNorm2d);

extern std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> REGISTERED_NORM_DICT;

//Initialize registered norms
inline void register_norms()
{
	REGISTERED_NORM_DICT["bn2d"] = nullptr;	//std::make_shared<torch::nn::AnyModule>(torch::nn::BatchNorm2d());
	REGISTERED_NORM_DICT["ln"] = nullptr;		//std::make_shared<torch::nn::AnyModule>(torch::nn::LayerNorm());
	REGISTERED_NORM_DICT["ln2d"] = nullptr;	// std::make_shared<torch::nn::AnyModule>(LayerNorm2d());
	REGISTERED_NORM_DICT["trms2d"] = nullptr;	//TritonRMSNorm2d().ptr();			!!!
}

//Function to build a norm function
inline std::shared_ptr<torch::nn::Module> build_norm(
	const std::string &name,
	const int64_t &num_features
	//const std::unordered_map<std::string, double> &kwargs
){
	std::shared_ptr<torch::nn::Module> result;

	if (REGISTERED_NORM_DICT.find(name) != REGISTERED_NORM_DICT.end())
	{
		//if (name == "ln" || name == "ln2d" || name == "trms2d") kwargs["normalized_shape"] = num_features else kwargs["num_features"] = num_features;
		if (name == "bn2d")
			result = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions({num_features})/*.eps(1e-5).momentum(0.1).affine(true).track_running_stats(true)*/).ptr();
		if (name == "ln")
			result = torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_features})/*.elementwise_affine(elementwise_affine).eps(eps)*/).ptr();
		if (name == "ln2d")
			result = LayerNorm2d(torch::nn::LayerNormOptions({num_features})/*.elementwise_affine(elementwise_affine).eps(eps)*/).ptr();
		if (name == "trms2d")
			result = TritonRMSNorm2d(num_features).ptr();
		REGISTERED_NORM_DICT[name] = result;
	} else {
		std::cout<<"build_norm: unregistered norm "<<name<<"!"<<std::endl;
		return nullptr;
	}

	return result;
}


///!!!Придумать как избавиться от этого
inline torch::Tensor forward_norm(std::shared_ptr<torch::nn::Module> norm, torch::Tensor x, const std::string &name)
{
	if (name == "bn2d")
		return norm->as<torch::nn::BatchNorm2d>()->forward(x);
	if (name == "ln")
		return norm->as<torch::nn::LayerNorm>()->forward(x);
	if (name == "ln2d")
		return norm->as<LayerNorm2d>()->forward(x);
	if (name == "trms2d")
		return norm->as<TritonRMSNorm2d>()->forward(x);

	if (!(name == "bn2d" || name == "ln" || name == "ln2d" || name == "trms2d"))
		throw std::runtime_error("forward_norm error: unknown norm name " + name);

	return torch::Tensor();
}


///!!!Придумать как избавиться от этого
template <typename TSequential>
inline void add_to_seq_norm(TSequential seq, std::shared_ptr<torch::nn::Module> norm, const std::string &name)
{
	if (name == "bn2d")
		seq->push_back(/*std::make_shared */*norm->as<torch::nn::BatchNorm2d>());
	if (name == "ln")
		seq->push_back(/*std::make_shared */*norm->as<torch::nn::LayerNorm>());
	if (name == "ln2d")
		seq->push_back(/*std::make_shared */*norm->as<LayerNorm2d>());
	if (name == "trms2d")
		seq->push_back(/*std::make_shared */*norm->as<TritonRMSNorm2d>());

	if (!(name == "bn2d" || name == "ln" || name == "ln2d" || name == "trms2d"))
		throw std::runtime_error("add_to_seq_norm error: unknown norm name " + name);
}
