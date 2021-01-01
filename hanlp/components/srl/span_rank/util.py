# Adopted from https://github.com/KiroSummer/A_Syntax-aware_MTL_Framework_for_Chinese_SRL
import torch


def block_orth_normal_initializer(input_size, output_size):
    weight = []
    for o in output_size:
        for i in input_size:
            param = torch.FloatTensor(o, i)
            torch.nn.init.orthogonal_(param)
            weight.append(param)
    return torch.cat(weight)
