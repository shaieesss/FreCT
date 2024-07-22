import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import DAC_structure, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
from tkinter import _flatten

class Convolution(nn.Module):
    def __init__(self, win, groups):
        super(Convolution, self).__init__()
        self.len = win
        self.groups = groups
        self.pointconv1 = nn.Conv1d(self.len, self.len, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.pointconv2 = nn.Conv1d(self.len, self.len, kernel_size=3, stride=1, padding=0, bias=True, groups=1)
        self.pointconv3 = nn.Conv1d(self.len, self.len, kernel_size=5, stride=1, padding=0, bias=True, groups=1)
        
        self.act = nn.Sigmoid()
        
        self.proj = nn.Conv1d(self.len, self.len, kernel_size=2, stride=3, padding=3, dilation=2, bias=True, groups=1)
        
        
    def forward(self, x):
        # batch, len, head, d_model
        #x = self.layerNorm(x)
        B, L, H, D = x.shape
        if H == 1:
            x = torch.squeeze(x, dim=2)
        else:
            x = x.view(B, L, -1)
        
        # [64, 90, 256]
        x1 = self.pointconv1(x)
        # [64, 90, 254]
        x2 = self.pointconv2(x)
        # [64, 90, 252]
        x3 = self.pointconv3(x)
        
        # [64, 90, 762]
        x = torch.cat((torch.cat((x1, x2), dim=-1), x3), dim=-1)
        
        
        return self.act(self.proj(x)).unsqueeze(dim=2)
        
        

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        
    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        
        for attn_layer in self.attn_layers:
            series, prior = attn_layer[0](x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series = attn_layer[1](series)
            prior = attn_layer[1](prior)
            
            
            series_list.append(series)
            prior_list.append(prior)
            
        return series_list, prior_list



class FreCT(nn.Module):
    def __init__(self, win_size, enc_in, c_out, batchsize, groups, n_heads=1, d_model=256, e_layers=3, patch_size=[3,5,7], channel=55, dropout=0.0, activation='gelu', output_attention=True):
        super(FreCT, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        self.batch = batchsize
        # Patching List  
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size//patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)
         
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                (
                    nn.Sequential(
                        AttentionLayer(
                            DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout, output_attention=output_attention),
                            d_model, patch_size, channel, n_heads, win_size
                        ),
                        Convolution(win_size, groups)
                    )
                
                
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        #x = revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        # Mutil-scale Patching Operation 
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l') #Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l') #Batch channel win_size
            
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p = patchsize) 
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize) 
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)
            
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series), prior_patch_mean.append(prior)

        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))
            
        
        return series_patch_mean, prior_patch_mean

