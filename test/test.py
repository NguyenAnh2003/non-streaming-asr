import torch

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)
  

if __name__ == "__main__":
  x = torch.tensor([203, 270, 216, 256, 652, 786, 710, 631])
  output_lengths = calc_length(x, all_paddings=2, 
                               kernel_size=3, 
                               stride=2, 
                               ceil_mode=False,
                               repeat_num=2)
  print(output_lengths)