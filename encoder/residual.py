import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial


class AdaptiveResidual(nn.Module):
  def __init__(self,
               cin=3,
               cout=3,
               reverse=False,
               scale_param={'kernel_size':2,'stride':2,'padding':1},
               conv_param={'kernel_size':3,'stride':1,'padding':1,"bias":False},
               num_blocks=3,
               do_skip_connect=True,
               dropout_prob=0.5,
               do_scale=True,
               do_normalization=True,
               act_func=nn.ReLU(),
               ):
      super().__init__()
      """
      Parameters:
      ----------
      cin: int - input channels
      cout: int - output channels
      reverse: bool - if True use Conv2dTranspose,else Conv2d


      scale_param: dict - parameters for scale function
      conv_param:dict - parameters for each convolution function
      dropout_prob:float or int - parameter for Dropout2D

      num_blocks: int - num convolutional operations
      do_skip_connect: bool - if true - use scip connection, else dont use
      do_scale: bool - if true use MaxPool2d or Conv2dTranspose after all blocks
      do_normalization: bool - if true, use Bathc2DNormalization, else - not
      act_func: Callable - your activation function
      ----------
      """

      self.do_skip_connect = do_skip_connect

      scale_param = scale_param if scale_param is not None else \
                                              {"kernel_size":3,
                                               "stride":2,
                                               "padding":0}

      conv_param = conv_param if conv_param is not None else \
                                              {"kernel_size":3,
                                               "stride":1,
                                               "padding":0,
                                               "bias":False}


      if reverse:
        conv = nn.ConvTranspose2d
        scale_func = nn.ConvTranspose2d(cout, cout, bias=False,
                                **scale_param) if do_scale else nn.Identity()


      else:
        conv = nn.Conv2d
        scale_func = nn.MaxPool2d(**scale_param) \
                                if do_scale else  nn.Identity()


      layers = [nn.Sequential(OrderedDict([
              ("conv", conv( in_channels=cin if i==0 else cout,
                             out_channels=cout,
                             **conv_param)
              ),
              ("bnorm", nn.BatchNorm2d(cout) if do_normalization else nn.Identity()),
              ("act", act_func),
              ("drop", nn.Dropout2d(dropout_prob))])

      ) for i in range(num_blocks)]

      self.layers = nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer
                                              in enumerate(layers)]))

      self.residual = nn.Identity() if cin==cout else \
                      conv(cin, cout, kernel_size=1)

      self.scale = nn.Sequential(OrderedDict([
          (f"{'scale' if do_scale else 'no scale'}", scale_func)]))

      self.stabilazation_size = partial(nn.Upsample)

  def forward(self, x):

    layers_end = self.layers(x)
    skip_connect_transform = self.residual(self.stabilazation_size(
                             size=layers_end.size()[2:])(x))

    return self.scale(layers_end + skip_connect_transform) if self.do_skip_connect \
           else self.scale(layers_end)