from torch.utils.cpp_extension import load
import torch

from torch.autograd import Function

import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


#import loralib as lora 

from math import sqrt
from scipy.stats.mstats import gmean
import wandb

global lba_fc_op 
lba_fc_op = load(name='conv2d', sources=['lba_fc/kernels/lba_fc_cuda.cpp', 'lba_fc/kernels/lba_fc_cuda_kernel.cu'])

def calc_bit_mask(n):
    if n <= 0:
        return -(2**(-n))
    else:
        return -(2**(23-n))

class lba_matmul(Function):
    @staticmethod
    def forward(ctx, x, w, bit_mask_man, exp, chunk_size, mode, exp_bias, amode, eta, log=None):
        
        
        ctx.save_for_backward(x, w)
        
        if exp < 0:
            ctx.saved_params = (bit_mask_man, exp, chunk_size, 0, exp_bias, amode, eta, log)
            return torch.matmul(x.contiguous(), w.transpose(0,1).contiguous())
        else:
            ctx.saved_params = (bit_mask_man, exp, chunk_size, mode, exp_bias, amode, eta, log)
            #print('####',x.type(),w.type())
            return lba_fc_op.matmul(x.contiguous(), w.contiguous(), bit_mask_man, exp, chunk_size, exp_bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, w,  = ctx.saved_tensors
        ## w [d1 d0]
        ## g [b0 b1 d1]
        ## x [b0 b1 d0]

        bit_mask_man, exp, chunk_size, mode, exp_bias, amode, eta, log = ctx.saved_params

        if mode == 1 or mode == 2 or mode >= 4:
            alphas = lba_fc_op.matmul_alphas(x, w, bit_mask_man, exp, chunk_size, exp_bias, amode)
            ##alphas = alphas.clamp(0.25, 4)
            ## alphas : [b0, b1, d1, d0]

            if log is not None:
                wandb.log({f"{log}_alphas": (alphas.sum().cpu().item()/alphas.numel())}, commit = False)
                ##print("sent to wandb")
        else:
            alphas = 1

        x_grad = w_grad = None

        d0 = x.size(2)
        b0 = x.size(0)
        b1 = x.size(1)
        d1 = w.size(0)

        if mode == 9 or mode == 10: ## "STE mode"
            alphas = alphas.float()
            alphas = alphas.reshape(b0,b1, d1, chunk_size , -1)
            alphas[:,:,:,-1,:] = alphas[:,:,:,-1,:].flip(3).cumprod(3).flip(3)
            alphas = alphas.flip(3).cumprod(3).flip(3)
            alphas = alphas.reshape(b0, b1, d1, d0)


        if ctx.needs_input_grad[0]:
            if mode == 2 or mode >= 4:
                g = grad_output.reshape(b0*b1, 1, d1)

                weighted_weight = w.unsqueeze(0).unsqueeze(1) * alphas
                ##weighted_weight = weighted_weight/128.0
                weighted_weight = weighted_weight.view(b0*b1, d1, d0)
                ## [b0*b1, 1, d1] * [b0*b1, d1, d0] -> [b0*b1, 1, d0]

                x_grad = torch.bmm(g , weighted_weight).reshape(b0, b1, d0)
                if mode == 4:
                    x_grad = 0.95*x_grad + 0.05*(grad_output @ w)
                ##x_grad = torch.einsum('btj,btji,ji->bti', grad_output, alphas, w)

                if mode == 8 or mode == 10:
                    x_grad = x_grad - eta*torch.einsum('ji,bti->bti', w**2, x)

                ## [b0*b1, 1, d0] -> [b0, b1, d0]
            else:
                ## [b0, b1, d1] * [d1, d0] -> [b0, b1, d0]
                x_grad = grad_output @ w
            

        if ctx.needs_input_grad[1]:


            if mode == 2 or mode == 1 or mode >= 4:
                g = grad_output.reshape(b0*b1, d1, 1)
                xb = x.view(b0*b1, 1, d0)
                ## [b0*b1, d1, 1] * [b0*b1, b1, d0] -> [b0*b1, d1, d0]
                w_grad = torch.bmm(g, xb).view(b0,b1, d1, d0)
                ## [b0*b1, d1, d0] -> [b0,b1,d1, d0]
                w_grad = (w_grad * alphas).sum([0,1])
                ##w_grad = w_grad/128.0
                ## [b0,b1,d1, d0] -> [d1, d0]
                if mode == 4:
                    w_grad = 0.95*w_grad + 0.05*torch.einsum('btj,bti->ji', grad_output, x)
                ##w_grad= torch.einsum('btj,btji,bti->ji', grad_output, alphas, x)

            if mode == 7 or mode == 8 or mode == 10:
                w_grad = w_grad - eta*torch.einsum('ji,bti->ji', w, x**2)


            else:
                w_grad= torch.einsum('btj,bti->ji', grad_output, x)


        return x_grad, w_grad, None, None, None, None, None, None, None, None, None


def lba_bmm(X,Y ,bit_mask_man, exp, chunk_size, mode, exp_bias, amode, eta, ways=1):
    ## X [b0 b1 d0 d1]
    ## Y [b0 b1 d2 d1]
    #breakpoint()
    if X.size(-1) !=Y.size(-1):
        Y = Y.transpose(-2, -1).contiguous()

    B0 = X.size(0)
    B1 = X.size(1)
    D0 = X.size(2)
    D1 = X.size(3)
    D2 = Y.size(2)

    out = torch.zeros(B0, B1, D0, D2, device=X.device)

    for b0 in range(B0):
        for b1 in range(B1):

            start = 0 
            per_instance = D2 // ways
            assert(D2 % ways == 0)
            for _ in range(ways):
                end = start + per_instance
                out[b0,b1,:, start:end] = lba_matmul.apply(X[b0,b1:b1+1,:,:], 
                                                           Y[b0,b1, start:end,:],  
                                                           bit_mask_man, 
                                                           exp, 
                                                           chunk_size, 
                                                           mode, 
                                                           exp_bias, 
                                                           amode,
                                                           eta)
                start = end

    return out





class LBA_Linear(torch.nn.Linear):

    counter = 0

    def __init__(self, in_features, out_features, man, exp, chunk_size, mode, exp_bias, amode, eta, split , bias,  **kwargs):

        self.bit_mask_man = calc_bit_mask(man)

        super(LBA_Linear, self).__init__(in_features, out_features, bias)
        self.has_bias = bias
        self.exp = exp
        self.chunk_size = chunk_size
        self.mode = mode
        self.split = split
        self.exp_bias = exp_bias
        ## debug message to verify correct initilization

        self.label = f"op{LBA_Linear.counter}"
        self.iterations_counter = 0
        self.amode = amode
        self.eta = eta

        print(f"{self.label}: ({self.iterations_counter})   in_features: {in_features}, out_features: {out_features}, M{man}E{exp}, chunk = {chunk_size}, mode = {mode}, exp_bias = {exp_bias}, bias = {bias}, split = {split}")
        
        self.quant_enabled = True
        self.unit_scaling = False
        if self.unit_scaling:
            from unit_scaling.scale import scale_bwd, scale_fwd

        LBA_Linear.counter = LBA_Linear.counter + 1


    def enable_quant(self, enable):
        self.quant_enabled = enable
        
    def set_quantized(self, quantized):
        self.enable_quant(quantized)

    def sort_weights(self):
        c = self.chunk_size

        w = self.weight.clone()

        if self.in_features % c != 0:
            w = torch.cat([w, torch.zeros(self.out_features, c - self.in_features % c, device=w.device)], dim=1)

        w = w.reshape(self.out_features, self.in_features // c, c)

        _, indices = w.abs().sort(dim=2)
        w = w.gather(2, indices)

        norms = w.norm(dim=2, keepdim=True)
        norms = norms.expand(w.shape)

        _, indices = norms.sort(dim=1)
        w = w.gather(1, indices)

        self.weight.data = w.reshape(self.out_features, -1)[:, :self.in_features]

    def forward(self, x):
        need_squeeze = False
        w = self.weight 
        bias = self.bias 

        self.iterations_counter = (self.iterations_counter+1)*2


        if self.unit_scaling:
            fan_out, fan_in = w.shape
            batch_size = x.numel() // fan_in

            output_scale = fan_in**-0.5
            grad_input_scale = fan_out**-0.5
            grad_weight_scale = grad_bias_scale = batch_size**-0.75

            output_scale = sqrt(output_scale**2  + grad_input_scale**2)

            x = scale_bwd(x, grad_input_scale)  # i.e. x * grad_input_scale, but only in the bwd pass
            w = scale_bwd(w, grad_weight_scale)
            if self.has_bias:
                bias = scale_bwd(bias, grad_bias_scale)

        ##auto factor
        # factor_x = 2**(x.abs().max().log2().floor())
        # factor_w = 2**(w.abs().max().log2().floor())



        if x.dim() == 2:
            need_squeeze = True
            x = x.unsqueeze(1)

        if self.quant_enabled:
            exp = self.exp
            bit_mask_man = self.bit_mask_man
        else:
            exp = -1
            bit_mask_man = -1
        
        

        dout = w.size(0)

        X = []
        start = 0
        per_instance = dout // self.split
        assert(dout % self.split == 0)
        if (x.device.index == 0): ## and self.iterations_counter % 100 == 0):
            label = self.label
        else:
            label = None

        for _ in range(self.split):
            end = start + per_instance
            y = lba_matmul.apply(x, w[start:end,:],  
                                 bit_mask_man, exp, 
                                 self.chunk_size, 
                                 self.mode, 
                                 self.exp_bias, 
                                 self.amode,
                                 self.eta,
                                 label)
            X.append(y)
            start = end
            label = None

        assert(start == dout)

        x = torch.cat(X, dim=2)

        if need_squeeze:
            x = x.squeeze(1)

        if self.has_bias:
            x = x + self.bias
        



        if self.unit_scaling:
            x = scale_fwd(x, output_scale)
            


        return x
'''
class LBA_lora_Linear(lora.Linear):    
    def __init__(self, in_features, out_features, man, exp, chunk_size, mode, split , bias, rank,  **kwargs):

        self.bit_mask_man = calc_bit_mask(man)

        super(LBA_lora_Linear, self).__init__(in_features, out_features, rank)
        self.has_bias = bias
        self.exp = exp
        self.chunk_size = chunk_size
        self.mode = mode
        self.split = split
        ## debug message to verify correct initilization
        print(f"LBA_lora_in features: {in_features}, out_features: {out_features}, M{man}E{exp}, chunk = {chunk_size}, mode = {mode}, bias = {bias}, split = {split}")


        self.quant_enabled = True

    def enable_quant(self, enable):
        self.quant_enabled = enable

    def set_quantized(self, quantized):
        self.enable_quant(quantized)

    def forward(self, x):
        
        need_squeeze = False

        x0 =x

        if x.dim() == 2:
            need_squeeze = True
            x = x.unsqueeze(1)

        if self.quant_enabled:
            exp = self.exp
            bit_mask_man = self.bit_mask_man
        else:
            exp = -1
            bit_mask_man = -1
        
        
        w = self.weight

        dout = w.size(0)


        X = []
        start = 0
        per_instance = dout // self.split
        assert(dout % self.split == 0)
        for _ in range(self.split):
            end = start + per_instance
            y = lba_matmul.apply(x, w[start:end,:],  bit_mask_man, exp, self.chunk_size, self.mode)
            X.append(y)
            start = end

        assert(start == dout)

        x = torch.cat(X, dim=2)

        if need_squeeze:
            x = x.squeeze(1)

        if self.has_bias:
            x = x + self.bias

        if self.r > 0 and not self.merged:
            x += (self.lora_dropout(x0) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling

        return x
'''
###--------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------

def lba_bmm_es(X,Y ,bit_mask_man, exp, chunk_size, mode, ways=1):
    ## X [b0 b1 d0 d1]
    ## Y [b0 b1 d2 d1]


    Y = Y.transpose(-2, -1).contiguous()

    B0 = X.size(0)
    B1 = X.size(1)
    D0 = X.size(2)
    D1 = X.size(3)
    D2 = Y.size(3)

    out = torch.zeros(B0, B1, D0, D2, device=X.device)

    for b0 in range(B0):
        for b1 in range(B1):

            start = 0 
            per_instance = D2 // ways
            assert(D2 % ways == 0)
            for _ in range(ways):
                end = start + per_instance
                out[b0,b1,:, start:end] = torch.matmul(X[b0,b1:b1+1,:,:], Y[b0,b1, :,start:end])
                start = end

    return out

class LBA_Linear_es(torch.nn.Linear):    
    def __init__(self, in_features, out_features, man, exp, chunk_size, mode, split , bias,  **kwargs):

        self.bit_mask_man = calc_bit_mask(man)
        
        super(LBA_Linear_es, self).__init__(in_features, out_features, bias)
        self.has_bias = bias
        self.exp = exp
        self.chunk_size = chunk_size
        self.mode = mode
        self.split = split
        ## debug message to verify correct initilization
        print(f"in_features: {in_features}, out_features: {out_features}, M{man}E{exp}, chunk = {chunk_size}, mode = {mode}, bias = {bias}, split = {split}")


    def forward(self, x):
        need_squeeze = False
        if x.dim() == 2:
            need_squeeze = True
            x = x.unsqueeze(1)

        w = self.weight.transpose(-2, -1).contiguous()

        dout = w.size(1)

        X = torch.zeros(x.size(0), x.size(1), dout, device=x.device)
        start = 0
        per_instance = dout // self.split
        assert(dout % self.split == 0)

        for _ in range(self.split):
            end = start + per_instance
            X[:,:,start:end] = torch.matmul(x, w[:,start:end])
            start = end

        assert(start == dout)

        if need_squeeze:
            X = X.squeeze(1)

        if self.has_bias:
            X = X + self.bias
        return X
