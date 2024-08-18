import torch
import torch.nn.functional as F
import torch.nn as nn

class FakeTensorQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, amax, bit_list, mode, activation = False):
        ctx.save_for_backward(inputs, amax)
        if not activation:
            if mode == 'linear':
                outputs = _tensor_quant(inputs.transpose(0,1), amax.cuda(), bit_list).transpose(0,1)
            elif mode == 'Conv2d':
                outputs = _tensor_quant(inputs.view(inputs.shape[0],-1).transpose(0,1), amax.cuda(), bit_list).transpose(0,1).view(inputs.shape)
            else:
                raise Exception('mode error')
        else:
            if mode == 'linear':
                outputs = _tensor_quant(inputs, amax.cuda(), bit_list, True)
            elif mode == 'Conv2d':
                outputs = _tensor_quant(torch.swapaxes(inputs, 0, 1).reshape(inputs.shape[1],-1).transpose(0,1), amax.cuda(), bit_list, True)
                outputs = outputs.transpose(0,1).reshape([inputs.shape[1],inputs.shape[0],inputs.shape[2],inputs.shape[3]])
                outputs = torch.swapaxes(outputs, 0, 1)
            else:
                raise Exception('mode error')
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None

def _tensor_quant(inputs, amax, bit_list, unsigned = False):
    input_dtype = inputs.dtype
    if inputs.dtype == torch.half:
        inputs = inputs.float()
    if amax.dtype == torch.half:
        amax = amax.float()
    if amax < 0:
        raise ValueError("Negative values in amax")

    epsilon = torch.tensor(1. / (1<<24), device=inputs.device)
    bit_tensor = torch.tensor(bit_list, device=amax.device)
    if not unsigned:
        xn = torch.clamp((inputs + amax)/(2*amax), 0, 1)
        bit_n = torch.pow(2, bit_tensor) - 1
        xq = torch.round(xn*bit_n) / bit_n
        outputs = xq*(2*amax) - amax
        outputs = outputs * (bit_tensor>0)
        outputs = torch.nan_to_num(outputs)
    else:
        xn = torch.clamp(inputs/amax, 0, amax)
        bit_n = torch.pow(2, bit_tensor) - 1
        xq = torch.round(xn*bit_n) / bit_n
        outputs = xq*amax
        outputs = torch.nan_to_num(outputs)

    return outputs



def quantizer_weight(weight, bits_list, type, amax):
    quant_weight = torch.zeros_like(weight)

    assert len(bits_list) == weight.shape[0]
    quant_weight = FakeTensorQuantFunction.apply(weight, amax, bits_list, type)

    return quant_weight

def quantizer_act(act, bits_list, type, amax):
    quant_act = torch.zeros_like(act)

    assert len(bits_list) == act.shape[1]
    quant_act = FakeTensorQuantFunction.apply(act, amax, bits_list, type, True)

    return quant_act