import torch
import os
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization import MinMaxObserver , MovingAverageMinMaxObserver
from torch.ao.quantization.qconfig import QConfig
from dorefa import DoReFaFakeQuantize 
import torch
QAT_ALGORITHM = ["fake_quantize"]
OBSERVER_TYPE = ["min_max_observer", "moving_average_min_max"]
def size_of_model(model):
    name_file = "temp.pt"
    torch.save(model.state_dict(), name_file)
    size =  os.path.getsize(name_file)/1e6
    os.remove(name_file)
    return size
def get_qat_algorithm( name_of_qat:str = "fake_quantize"):
    if name_of_qat == "fake_quantize":
        return FakeQuantize
    elif name_of_qat == "dorefa":
        return DoReFaFakeQuantize
    else:
        raise ValueError("name_of_qat should be one of the following: {}".format(QAT_ALGORITHM))
def get_observer( observer_type:str = "min_max_observer"):
    if observer_type == "min_max_observer":
        return MinMaxObserver
    elif observer_type == "moving_average_min_max":
        return MovingAverageMinMaxObserver
    else:
        raise ValueError("observer_type should be one of the following: {}".format(OBSERVER_TYPE))


def get_eager_quantization(
    weight_quantize:bool  = True,
    weight_quantization_min:int = -128,
    weight_quantization_max:int = 127,
    weight_quantization_dtype:torch.dtype = torch.qint8,
    weight_quantization_qscheme:torch.qscheme = torch.per_tensor_symmetric,
    weight_reduce_range = True,
    w_observer:str = "moving_average_min_max",
    w_fakequantize:str = "fake_quantize",
    activation_quantize:bool = True,
    activation_quantization_min:int = 0,
    activation_quantization_max:int = 255,
    activation_quantization_dtype:torch.dtype = torch.quint8,
    activation_quantization_qscheme:torch.qscheme = torch.per_tensor_affine,
    activation_reduce_range:bool = False , 
    a_observer:str = "moving_average_min_max",
    a_fakequantize:str = "fake_quantize",
) -> QConfig:
    assert isinstance( weight_quantization_dtype , torch.dtype)
    assert isinstance( activation_quantization_dtype , torch.dtype)
    assert isinstance( weight_quantization_qscheme , torch.qscheme)
    assert isinstance( activation_quantization_qscheme , torch.qscheme)
    if activation_quantization_dtype == torch.quint8 and activation_quantization_min != 0:
        raise ValueError("activation_quantization_min should be 0 for activation_quantization_dtype == torch.quint8")
    if weight_quantization_min == 0 and weight_quantization_dtype == torch.quint8:
        raise ValueError("weight_quantization_min should be 0 for weight_quantization_dtype == torch.quint8")
    ## all quantization  in eager mode are unifrom quantization 
    ##https://pytorch.org/blog/quantization-in-practice/
    weight_quantization_fake_quantize = torch.nn.Identity
    if weight_quantize:
        weight_quantization_fake_quantize = get_qat_algorithm(w_fakequantize).with_args(
                    observer =  get_observer(w_observer).with_args(
                        dtype = weight_quantization_dtype,
                        qscheme = weight_quantization_qscheme,
                        reduce_range = weight_reduce_range , 
                        quant_min= weight_quantization_min,
                        quant_max = weight_quantization_max,
                    ))
    activation_quantization_fake_quantize = torch.nn.Identity
    if activation_quantize:
            activation_quantization_fake_quantize = get_qat_algorithm(a_fakequantize).with_args(
                    observer =   get_observer(a_observer).with_args( 
                        quant_min = activation_quantization_min,
                        quant_max = activation_quantization_max,
                        dtype = activation_quantization_dtype,
                        qscheme = activation_quantization_qscheme,
                        reduce_range = activation_reduce_range
                    ))
    quantization_config = QConfig(
        weight = weight_quantization_fake_quantize,
        activation = activation_quantization_fake_quantize
    )
    return quantization_config
    