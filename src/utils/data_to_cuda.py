import torch
from src.sparse_torch.csx_matrix import CSRMatrix3d, CSCMatrix3d
import torch_geometric as pyg

def data_to_cuda(inputs,device):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
    if type(inputs) is list:
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x, device=device)
    elif type(inputs) is tuple:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x, device=device)
    elif type(inputs) is dict:
        for key in inputs:
            inputs[key] = data_to_cuda(inputs[key], device=device)
    elif type(inputs) in [str, int, float]:
        inputs = inputs
    elif type(inputs) in [torch.Tensor, CSRMatrix3d, CSCMatrix3d]:
        inputs = inputs.to(device)
    elif type(inputs) in [pyg.data.Data, pyg.data.Batch]:
        inputs = inputs.to(device)
    else:
        raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs
