import torch
import os
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization import MinMaxObserver
from torch.ao.quantization.qconfig import QConfig
import torch
def size_of_model(model):
    name_file = "temp.pt"
    torch.save(model.state_dict(), name_file)
    size =  os.path.getsize(name_file)/1e6
    os.remove(name_file)
    return size


def get_eager_quantization(
    weight_quantize:bool  = True,
    weight_observer_type:str = "moving_average_minmax",
    weight_quantization_min:int = 0,
    weight_quantization_max:int = 255,
    weight_quantization_dtype:torch.dtype = torch.quint8,
    weight_quantization_qscheme:torch.qscheme = torch.per_tensor_symmetric,
    weight_reduce_range = True,
    activation_quantize:bool = True,
    activation_observer_type:str = "moving_average_minmax",
    activation_quantization_min:int = -128,
    activation_quantization_max:int = 127,
    activation_quantization_dtype:torch.dtype = torch.quint8,
    activation_quantization_qscheme:torch.qscheme = torch.per_tensor_symmetric,
    activation_reduce_range:bool = False
) -> QConfig:
    assert isinstance( weight_quantization_dtype , torch.dtype)
    assert isinstance( activation_quantization_dtype , torch.dtype)
    assert isinstance( weight_quantization_qscheme , torch.qscheme)
    assert isinstance( activation_quantization_qscheme , torch.qscheme)
    ## all quantization  in eager mode are unifrom quantization 
    weight_quantization_fake_quantize = torch.nn.Identity
    if weight_quantize:
        weight_quantization_fake_quantize = FakeQuantize.with_args(
                    observer =  MinMaxObserver.with_args(
                        dtype = weight_quantization_dtype,
                        qscheme = weight_quantization_qscheme,
                        reduce_range = False , 
                        quant_min= weight_quantization_min,
                        quant_max = weight_quantization_max,
                    ))
    activation_quantization_fake_quantize = torch.nn.Identity
    if activation_quantize:
            activation_quantization_fake_quantize = FakeQuantize.with_args(
                    observer =   MinMaxObserver.with_args( 
                        quant_min = activation_quantization_min,
                        quant_max = activation_quantization_max,
                        dtype = activation_quantization_dtype,
                        qscheme = activation_quantization_qscheme,
                        reduce_range = False
                    ))
    quantization_config = QConfig(
        weight = weight_quantization_fake_quantize,
        activation = activation_quantization_fake_quantize
    )
    return quantization_config
    