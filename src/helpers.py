import torch


def show_summary(model, input_size):
    from torchinfo import summary
    from rich import print
    summary_str = summary(model, input_size=input_size,
                          col_names=['input_size', 'output_size'], verbose=0)
    print(summary_str)


def conv_shape(in_shape: any,
               kernel_size:  int,
               stride: int = 1,
               padding: int = 0) -> any:
    """ 
    Helper function that returns the shape of an image after 2D convolution 
    Parameters:
        in_shape    (int): shape of input
        kernel_size (int): size of convolution kernel
        stride      (int): stride of convolution operation
        padding     (int): padding of convolution operation
    Output:
        out_shape   (int): shape of input after 2D convolution
    """

    # Initialize shape
    if not isinstance(in_shape, torch.Tensor):
        out_shape = torch.tensor(in_shape)
    else:
        out_shape = in_shape

    out_shape[-1] = torch.div(out_shape[-1] - kernel_size + 2 * padding,
                              stride, rounding_mode='floor') + 1
    out_shape[-2] = torch.div(out_shape[-2] - kernel_size + 2 * padding,
                              stride, rounding_mode='floor') + 1
    return out_shape


def deconv_shape(in_shape: any,
                 kernel_size:  int,
                 stride: int = 1,
                 padding: int = 0) -> any:
    """ 
    Helper function that returns the shape of an image after 2D deconvolution 
    Parameters:
        in_shape    (int): shape of input
        kernel_size (int): size of deconvolution kernel
        stride      (int): stride of deconvolution operation
        padding     (int): padding of deconvolution operation
    Output:
        out_shape   (int): shape of input after 2D deconvolution
    """

    # Initialize shape
    if not isinstance(in_shape, torch.Tensor):
        out_shape = torch.tensor(in_shape)
    else:
        out_shape = in_shape

    out_shape[-1] = (stride * (out_shape[-1] - 1) +
                     kernel_size - 2 * padding)
    out_shape[-2] = (stride * (out_shape[-2] - 1) +
                     kernel_size - 2 * padding)
    return out_shape


def flat_size(in_shape: any, start_dim: int = 1):
    """ 
    Helper function that returns the shape of an input after a flattening layer
    Parameters:
        in_shape  (int): shape of input
    Output:
        out_size (int): size of input after flattening layer
    """
    if not isinstance(in_shape, torch.Tensor):
        out_shape = torch.tensor(in_shape)
    else:
        out_shape = in_shape
    out_size = out_shape[start_dim:].prod().item()
    return out_size


def flat_shape(in_shape: any, start_dim: int = 1):
    """ 
    Helper function that returns the shape of an input after a flattening layer
    Parameters:
        in_shape  (int): shape of input
    Output:
        out_shape (int): shape of input after flattening layer
    """
    if not isinstance(in_shape, torch.Tensor):
        out_shape = torch.tensor(in_shape)
    else:
        out_shape = in_shape
    init_shape = out_shape[:start_dim]
    prod_shape = out_shape[start_dim:].prod()
    out_shape = torch.tensor([init_shape, prod_shape])
    return out_shape


if __name__ == "__main__":

    from rich import print

    kernel_size, stride, padding = (5, 1, 0)
    conv_layer = torch.nn.Conv2d(3, 3, kernel_size, stride)
    deconv_layer = torch.nn.ConvTranspose2d(3, 3, kernel_size, stride)

    shape = torch.Size([5, 3, 64, 64])
    x = torch.rand(shape)
    y = conv_layer(x)

    print('conv_shape():')
    print((f'{shape} -> kernel_size: {kernel_size} stride: {stride} '
           f'-> {conv_shape(shape, kernel_size, stride, padding)}'))
    print((f'{shape} -> kernel_size: {kernel_size} stride: {stride} '
           f'-> {y.shape}'))

    shape = torch.Size(conv_shape(shape, kernel_size, stride, padding))
    print('deconv_shape():')
    print((f'{shape} -> kernel_size: {kernel_size} stride: {stride} '
           f'-> {deconv_shape(shape, kernel_size, stride, padding)}'))
    print((f'{shape} -> kernel_size: {kernel_size} stride: {stride} '
           f'-> {deconv_layer(y).shape}'))

    print((f'{shape} -> nn.Flatten() -> {flat_shape(shape)}'))
