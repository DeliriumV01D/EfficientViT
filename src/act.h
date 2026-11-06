#pragma once

#include "TorchHeader.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <optional>

extern std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> REGISTERED_ACT_DICT;

//Initialize registered activation functions
inline void register_activation_functions()
{
	REGISTERED_ACT_DICT["relu"] = torch::nn::ReLU().ptr();
	REGISTERED_ACT_DICT["relu6"] = torch::nn::ReLU6().ptr();
	//REGISTERED_ACT_DICT["hswish"] = torch::nn::Hardswish().ptr(); !!!Есть в более позднем torch
	//REGISTERED_ACT_DICT["hswish"] = torch::nn::SiLU().ptr();			!!!Можно попробовать заменить
	REGISTERED_ACT_DICT["silu"] = torch::nn::SiLU().ptr();
	REGISTERED_ACT_DICT["gelu"] = torch::nn::GELU(torch::nn::GELUOptions().approximate("tanh")).ptr();
}

//Function to build an activation function
inline std::shared_ptr<torch::nn::Module> build_act(const std::string &name)
{
	if (REGISTERED_ACT_DICT.find(name) != REGISTERED_ACT_DICT.end())
	{
		return REGISTERED_ACT_DICT[name];
	} else {
		std::cout << "build_act: unregistered activation " << name << "!" << std::endl;
		return nullptr;
	}
}


///!!!Придумать как избавиться от этого
inline torch::Tensor forward_act(std::shared_ptr<torch::nn::Module> act, torch::Tensor x, const std::string &name)
{
	if (name == "relu")
		return act->as<torch::nn::ReLU>()->forward(x);
	if (name == "relu6")
		return act->as<torch::nn::ReLU6>()->forward(x);
	if (name == "silu")
		return act->as<torch::nn::SiLU>()->forward(x);
	if (name == "gelu")
		return act->as<torch::nn::GELU>()->forward(x);
	//if (name == "hswish")
	//	return act->as<torch::nn::Hardswish>()->forward(x);

	if (!(name == "relu" || name == "relu6" || name == "silu" || name == "gelu"/*|| name == "hswish"*/))
		throw std::runtime_error("forward_act error: unknown activation name " + name);

	return torch::Tensor();
}


///!!!Придумать как избавиться от этого
template <typename TSequential>
inline void add_to_seq_act(TSequential seq, std::shared_ptr<torch::nn::Module> act, const std::string &name)
{
	if (name == "relu")
		seq->push_back(/*std::make_shared */*act->as<torch::nn::ReLU>());
	if (name == "relu6")
		seq->push_back(/*std::make_shared */*act->as<torch::nn::ReLU6>());
	if (name == "silu")
		seq->push_back(/*std::make_shared */*act->as<torch::nn::SiLU>());
	if (name == "gelu")
		seq->push_back(/*std::make_shared */*act->as<torch::nn::GELU>());
	//if (name == "hswish")
	//	seq->push_back(/*std::make_shared */*act->as<torch::nn::Hardswish>());

	if (!(name == "relu" || name == "relu6" || name == "silu" || name == "gelu"/*|| name == "hswish"*/))
		throw std::runtime_error("add_to_seq_act error: unknown activation name " + name);
}