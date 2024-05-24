import numpy as np
from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn

class AdaptiveShuffleResidual(nn.Module):
  def __init__(self,
               cin=3,
               cout=3,
               reverse=False,
               block_in_upscale=True,
               scale_param={'kernel_size':2,'stride':2,'padding':1},
               conv_param={'kernel_size':3,'stride':1,'padding':1,"bias":False},
               save_out_channels=True,
               pixel_shuffle_param= 2,
               num_blocks=3,
               do_skip_connect=True,
               dropout_prob=0.5,
               do_scale=True,
               do_normalization=True,
               act_func=nn.ReLU(),
               ):
      
      """
      Parameters:
      ----------
      cin: int - input channels
      cout: int - output channels
      reverse: bool - if True use Conv2dTranspose,else Conv2d
      block_in_upscale: bool or None - If true use pixelshuffle in block
                                       If false use pixelunshuffle in block
                                       If none - no scale in block

      scale_param: dict - parameters for scale function
      conv_param:dict - parameters for each convolution function
      save_out_channels: bool - parameters, which can you save your out channels if 
                                you use block_in_upscale parameters
      
      dropout_prob: float or int - parameter for Dropout2D
      pixel_shuffle_param: int - pixelshuffle scale parameter
      num_blocks: int - num convolutional operations
      do_skip_connect: bool - if true - use scip connection, else dont use
      do_scale: bool - if true use MaxPool2d or Conv2dTranspose after all blocks
      do_normalization: bool - if true, use Bathc2DNormalization, else - not
      act_func: Callable - your activation function
      ----------
      """
      super().__init__()


      self.do_skip_connect = do_skip_connect
      

      scale_param = scale_param if scale_param is not None else \
                                              {"kernel_size":3,
                                               "stride":2,
                                               "padding":1}

      conv_param = conv_param if conv_param is not None else \
                                              {"kernel_size":1, "stride":1,
                                               "padding":0, "dilation":1,
                                               "groups":1,
                                               "bias":False,
                                               "padding_mode":'zeros',
                                               "device":None,
                                               "dtype":None}

      if block_in_upscale==True:

          pixel_shuffler = nn.PixelShuffle(pixel_shuffle_param) 
          c_out = (lambda i: cout if i==0 else cout // pixel_shuffle_param**(2*i))
          
      if block_in_upscale==False:
          
          pixel_shuffler = nn.PixelUnshuffle(pixel_shuffle_param)
          c_out = lambda i: cout if i==0 else cout * pixel_shuffle_param**(2*i)

        
      if block_in_upscale is None:
          pixel_shuffler = nn.Identity()
          c_out = (lambda i: cout)



      if reverse:
        conv = nn.ConvTranspose2d
        scale_func = nn.ConvTranspose2d(c_out(num_blocks), c_out(num_blocks), bias=False,
                                **scale_param) if do_scale else nn.Identity()

      else:
        conv = nn.Conv2d
        scale_func = nn.MaxPool2d(**scale_param) \
                                if do_scale else  nn.Identity()

        
        
      layers = [nn.Sequential(OrderedDict([
              ('conv', conv(in_channels = cin if i==0 else c_out(i),
                             out_channels=c_out(i),
                              **conv_param)
              ),
          
              ('pxshuffler',pixel_shuffler),
              ("bnorm", nn.BatchNorm2d(c_out(i+1)) if do_normalization else \
                        nn.Identity()),
              ("act", act_func),
              ("drop", nn.Dropout2d(dropout_prob))])

      ) for i in range(num_blocks)]
    
      
      self.save_cout = conv(c_out(num_blocks), c_out(0), kernel_size=1) if save_out_channels \
                       else nn.Identity()

      self.layers = nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer
                                              in enumerate(layers)]))

      self.residual = nn.Identity() if cin==c_out(num_blocks) else \
                      conv(cin,c_out(num_blocks), kernel_size=1)

      self.endscale = nn.Sequential(OrderedDict([
          (f"{'scale' if do_scale else 'no scale'}", scale_func)]))

      self.res_scale = partial(nn.Upsample)


  def forward(self, x):

    sc = self.residual(x)
    x = self.layers(x)
    x = (x + self.res_scale(size=x.size()[2:])(sc)) if self.do_skip_connect else x
    x = self.endscale(x)
    x = self.save_cout(x)

    return x