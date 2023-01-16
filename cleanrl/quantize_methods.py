import torch
import os
def size_of_model(model):
    name_file = "temp.pt"
    torch.save(model.state_dict(), name_file)
    size =  os.path.getsize(name_file)/1e6
    os.remove(name_file)
    return size
def get_quantization_obverver( observer_type:str = "minmax", 
                              ):
    if observer_type == "minmax":
        return torch.ao.quantization.MovingAverageMinMaxObserver
    elif observer_type == "histogram":
        return torch.ao.quantization.HistogramObserver
    elif observer_type == "moving_average_minmax":
        return torch.ao.quantization.MovingAverageMinMaxObserver
    else:
        raise ValueError("observer_type must be one of minmax, histogram, moving_average")
    
'''
https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fake_quantize.py#L114
Torch Quantization Scheme has two value for tesnor quantization
1. per_tensor_symmetric
2. per_tensor_affine ( Asymmetric Quantization )
'''
def get_eager_quantization(
    weight_quantize:bool  = True,
    weight_observer_type:str = "moving_average_minmax",
    weight_quantization_min:int = 0,
    weight_quantization_max:int = 255,
    weight_quantization_dtype:torch.dtype = None,
    weight_quantization_qscheme:torch.qscheme = torch.per_tensor_symmetric,
    weight_reduce_range = True,
    activation_quantize:bool = True,
    activation_observer_type:str = "moving_average_minmax",
    activation_quantization_min:int = -128,
    activation_quantization_max:int = 127,
    activation_quantization_dtype:torch.dtype = None,
    activation_quantization_qscheme:torch.qscheme = torch.per_tensor_symmetric,
    
    
):
    ## all quantization  in eager mode are unifrom quantization 
    weight_quantization_fake_quantize = None
    if weight_quantize:
        weight_quantization_fake_quantize = torch.ao.quantization.FakeQuantize.with_args(
                    observer =  get_quantization_obverver( weight_observer_type) , 
                    quant_min= weight_quantization_min,
                    quant_max = weight_quantization_max,
                    dtype = weight_quantization_dtype,
                    qscheme =  weight_quantization_qscheme,
                    reduce_range = weight_reduce_range
                    ),
    activation_quantization_fake_quantize = None
    if activation_quantize:
            activation_quantization_fake_quantize = torch.ao.quantization.FakeQuantize.with_args(
                observer =   get_quantization_obverver( activation_observer_type) ,
                                                quant_min = activation_quantization_min,
                                                quant_max = activation_quantization_max,
                                                dtype = activation_quantization_dtype,
                                                qscheme = activation_quantization_qscheme,
                                                reduce_range = False
                                                )
    return torch.ao.quantization.QConfig(
        weight = weight_quantization_fake_quantize,
        activation = activation_quantization_fake_quantize
    )