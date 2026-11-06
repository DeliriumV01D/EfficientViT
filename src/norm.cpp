#include "norm.h"

std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> REGISTERED_NORM_DICT;