import torch
import torch.nn as nn

import numpy as np
from collections import OrderedDict
from functools import partial


class Inception(nn.Module):
    def __init__(self,cin,
                      cout,
                      n_towers=3,
                      depth_tower=[1,2,3],
                      towers_param = [{'kernel_size':[1],'stride':[2],'padding':0,
                                       'dilation':1, 'groups':1, 'bias':True},
                                      {'kernel_size':[3,5],'stride':[1,1],'padding':0,
                                       'dilation':1, 'groups':1, 'bias':True  },
                                      {'kernel_size':[5,7,8],'stride':[1,1,1],'padding':0,
                                       'dilation':1, 'groups':1, 'bias':True }],
                      
                      p_dropouts = [0.1,0.2,0.1],
                      transform_size=(256, 256),
                      reverse=False,
                      act_func=nn.ReLU(),
                      xception_mode=True,
                      do_normalization=True,
                      do_scale=True,
                      endscale_param={'kernel_size':2,'stride':2,'padding':1}):
        super().__init__()
        """
        Parameters:
        ----------
        cin: int - input channels
        cout: int - output channels
        reverse: bool - if True use Conv2dTranspose,else Conv2d
        scale_param: dict - parameters for scale function
        conv_param:dict - parameters for each convolution function
        p_dropouts:float or int - parameter for Dropout2D
        transform_size: scale feature maps after tower conv block (need for concatenate images)
        do_scale: do scale with endscale_param parameters after concatenations
        num_blocks: int - num convolutional operations
        do_normalization: bool - if true, use Bathc2DNormalization, else - not
        act_func: Callable - your activation function
        ----------
        """
        def istrueformat(arg_par, num_par):
            
            if isinstance(arg_par, list):
                if len(arg_par) < num_par:
                    arg_par += [arg_par[-1]]*(num_par - len(arg_par))
                if len(arg_par) > num_par:
                    arg_par = arg_par[:num_par]
        
            if isinstance(arg_par, int):
                    arg_par = [arg_par]*num_par
            
            return arg_par

        
        towers_param = istrueformat(towers_param,n_towers )
        p_dropouts = istrueformat(p_dropouts, n_towers)
        depth_tower = istrueformat(depth_tower,n_towers)

        if isinstance(towers_param, list):
            towers_param_res = []
            for i, par in enumerate(towers_param):
                tower_param = {}
                for key,val in par.items():
                    tower_param[key] = istrueformat(val, depth_tower[i])
                towers_param_res.append(tower_param)
        
        self.n_towers = n_towers
        self.depth_tower = depth_tower
        self.towers_param = towers_param_res

        if reverse:
            conv = nn.ConvTranspose2d
            scale_func = nn.ConvTranspose2d(cout*n_towers, cout*n_towers,**endscale_param)
        else:
            conv = nn.Conv2d
            scale_func = nn.MaxPool2d(**endscale_param)
        

        self.layers = [nn.Sequential(
                        *[nn.Sequential(OrderedDict(

                        [(f'{1}',conv(in_channels=cin if j==0 else cout,
                        out_channels=(cin if j==0 else cout) \
                                      if xception_mode else cout,
                        kernel_size=par['kernel_size'][j],
                        stride=par['stride'][j],
                        padding=par['padding'][j],
                        dilation=par['dilation'][j],
                        groups=par['groups'][j],
                        bias=par['bias'][j])) ,

                        (f'{2}',conv(in_channels=cin if j==0 else cout,
                        out_channels=cout,
                        kernel_size=par['kernel_size'][j],
                        stride=par['stride'][j],
                        padding=par['padding'][j],
                        dilation=par['dilation'][j],
                        groups=par['groups'][j],
                        bias=par['bias'][j]) if xception_mode else nn.Identity()),
                        (f'{3}', nn.BatchNorm2d(cout) if do_normalization else nn.Identity()),
                        (f'{4}', nn.Identity() if j==self.depth_tower[i]-1 else act_func),
                        (f'{5}',nn.Dropout2d(p_dropouts[i]))]))

                 for j in range(self.depth_tower[i])]) for i, par in enumerate(self.towers_param)]
        

        self.layers = nn.Sequential(OrderedDict(
            [(f'{i}', self.layers[i]) for i in range(n_towers)]
        ))

        self.upsample = nn.Upsample(transform_size)

        self.residual_block = nn.Identity if cin==cout else \
                              conv(cin,cout, 1)
        
        self.scale = nn.Sequential(OrderedDict([('scale', scale_func)])) if do_scale else nn.Identity()
        
    def forward(self, x):

        res = []
        sc = self.upsample(self.residual_block(x))
        for block in self.layers:

            out = block(x)
            out = self.upsample(out)
            res.append(out + sc)

        
        return self.scale(torch.cat(res, dim=1))