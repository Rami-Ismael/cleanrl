import torch
def get_datatype(args):
    if args.quantize_activation_quantize_dtype is not None:
        if args.quantize_activation_quantize_dtype == "quint8":
            args.quantize_activation_quantize_dtype = torch.quint8
        elif args.quantize_activation_quantize_dtype == "qint8":
            args.quantize_activation_quantize_dtype = torch.qint8
        else:
            raise ValueError(f"{args.quantize_activation_quantize_dtype} is not supported for quantization")
    return args