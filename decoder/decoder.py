import torch
import torch.nn as nn
from collections import OrderedDict
from residual import AdaptiveResidual



class Decoder(nn.Module):
  def __init__(self,
               num_ada_res_block=3,
               hidden_channels_in_block=[3,3,3],
               depth_per_block = [1,1,1],
               conv_param=None,
               scale_param=None,
               act_func=[nn.ReLU()],
               do_normalization=False,
               dropout_ps=[0.5],
               do_skip_connect=[False],
               do_scale_per_block=[True],
               ):
    """
      Parameters:
      ----------
      num_ada_res_block: int - num residual blocks
      hidden_channels_in_block: list - hidden_channels_in_block
      depth_per_block: list - depth res blocks
      conv_param: conv parameters
      scale_param: scale parameters
      fc_out_size: fully conection tensor out dimension
      
      read about other parameters in residual
      ----------
    """
    super().__init__()

    def islist(arg, size):
      if isinstance(arg, list):

        if len(arg) < size:
          arg += [arg[-1]]*(size - len(arg))
        if len(arg) > size:
          arg = arg[:size]
        return arg
      else:
        return islist([arg], size)


    hidden_channels_per_block = islist(hidden_channels_in_block,
                                            num_ada_res_block + 1)

    depth_per_block = islist(depth_per_block, num_ada_res_block)
    conv_param = islist(conv_param, num_ada_res_block)
    scale_param = islist(scale_param, num_ada_res_block)
    act_func = islist(act_func, num_ada_res_block)
    dropout_ps = islist(dropout_ps, num_ada_res_block)
    do_skip_connect = islist(do_skip_connect, num_ada_res_block)
    do_scale_per_block = islist(do_scale_per_block,num_ada_res_block)
    do_normalization = islist(do_normalization, num_ada_res_block)

    self.decoder = nn.Sequential(OrderedDict(

      [(f'Residual Block {i}',AdaptiveResidual(cin=hidden_channels_per_block[i],
                                               cout=hidden_channels_per_block[i+1],
                                               reverse=True,
                                               scale_param=scale_param[i],
                                               conv_param=conv_param[i],
                                               num_blocks=depth_per_block[i],
                                               do_skip_connect=do_skip_connect[i],
                                               dropout_prob=dropout_ps[i],
                                               do_scale=do_scale_per_block[i],
                                               do_normalization=do_normalization[i],
                                               act_func=act_func[i])) for i in range(num_ada_res_block)] ))


  def forward(self, x):

    return self.decoder(x)