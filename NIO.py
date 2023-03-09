import torch
import numpy as np
import pylab as plt
import torch.nn.functional as F
import torch.nn as nn
from torchinfo import summary

def kernel(in_chan=2, up_dim=32):
    """
        Kernel network apply on grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, 1, bias=False)
            )
    return layers

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, dim2,modes1 = None, modes2 = None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim1 = dim1 #output dimensions
        self.dim2 = dim2
        #self.batch_norm = nn.BatchNorm2d(out_channels)
        #self.dropout = nn.Dropout(p=0.42)
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2 + 1 #if not given take the highest number of modes can be taken
            self.modes2 = dim2//2 
        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes

        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        #print("Out FT Shape = ", out_ft.shape)
        #print("x_ft Shape = ", x_ft.shape)
        #print("Modes1 = ", self.modes1, " Modes2 = ", self.modes2, " self.dim1 = ", self.dim1, " self.dim2//2 + 1 = ", self.dim2//2+1)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2))
        #x = self.dropout(x)
        #x = self.batch_norm(x)
        return x


class pointwise_op(nn.Module):
    def __init__(self, in_channel, out_channel,dim1, dim2):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv2d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        #self.alpha_drop = nn.AlphaDropout(p=0.42)
        #self.layer_norm = nn.LayerNorm([int(out_channel), self.dim1, self.dim2])
        #self.batch_norm = nn.BatchNorm2d(int(out_channel))
        #self.upsample = nn.Upsample(size=[self.dim1, self.dim2], mode='bicubic', align_corners=True)

    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        #x_out = self.upsample(x_out)
        #x_out = self.alpha_drop(x_out)
        #x_out = self.layer_norm(x_out)
        #x_out = self.batch_norm(x_out)
        return x_out

class UNO(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, pad = 0, factor = 16/4):
        super(UNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.factor2 = factor/4
        self.padding = pad  # pad the domain if input is non-periodic
        self.reduceColor = nn.Conv2d(3, 1, kernel_size=1)
        self.interpolation = nn.Conv2d(3, 3, kernel_size=1)

        self.fc0 = nn.Linear(self.in_d_co_domain, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.d_co_domain, 4*factor*self.d_co_domain, 16, 16, 32, 32)

        self.conv1 = SpectralConv2d(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16, 26,26)

        self.conv2 = SpectralConv2d(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8,13,13)
        
        self.conv2_1 = SpectralConv2d(16*factor*self.d_co_domain, 32*factor*self.d_co_domain, 4, 4,7,7)
        
        self.conv2_9 = SpectralConv2d(32*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8,7,7)
    

        self.conv3 = SpectralConv2d(32*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16,13,13)

        self.conv4 = SpectralConv2d(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32,26,26)

        self.conv5 = SpectralConv2d(8*factor*self.d_co_domain, self.d_co_domain, 48, 48,32,32) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,4*factor*self.d_co_domain,75, 75) #
        
        self.w1 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 50, 50) #
        
        self.w2 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 25, 25) #
        
        self.w2_1 = pointwise_op(16*factor*self.d_co_domain, 32*factor*self.d_co_domain, 12, 12)
        
        #VAE Code begin
        self.linearFlatten1 = nn.Linear(384*12*12, 128)
        self.linearFlatten2 = nn.Linear(128, 4)
        self.linearFlatten3 = nn.Linear(128, 4)
        self.softmax1 = nn.Softmax(dim=1)
        self.N = torch.distributions.Normal(0, 10)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        self.linearUnflatten = nn.Linear(4, 128)
        self.relu = nn.ReLU(True)
        self.linearUnflatten2 = nn.Linear(128, 384*12*12)
        self.relu2 = nn.ReLU(True)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(384, 12, 12))
        #VAE Code End
        
        self.w2_9 = pointwise_op(32*factor*self.d_co_domain, 16*factor*self.d_co_domain, 25, 25)
        
        self.w3 = pointwise_op(32*factor*self.d_co_domain, 8*factor*self.d_co_domain, 50, 50) #
        
        self.w4 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 75, 75)
        
        self.w5 = pointwise_op(8*factor*self.d_co_domain, self.d_co_domain, 100, 100) # will be reshaped

        self.fc1 = nn.Linear(2*self.d_co_domain, 4*self.d_co_domain)
        self.fc2 = nn.Linear(4*self.d_co_domain, 3)
        self.increaseColor = nn.Conv2d(1, 3, kernel_size=1)

    def forward(self, x):
        
        grid = self.get_grid(x[0].shape, x.device)
        #print(grid.shape)
        
        #print(x[0].shape)
        #x_l = torch.cat((x[0], grid), dim=-1).cuda()
        #x_r = torch.cat((x[1], grid), dim=-1).cuda()

        x_l = x[0]
        x_r = x[1]
        #print(x_l.shape)
        x1 = self.interpolation(x_l)
        x2 = self.interpolation(x_r)
        ##print(x1.shape)
        x = x1 + x2
        x3 = x
        #x = self.reduceColor(x)
        x = x.permute(0,2,3,1)
        #x = torch.cat((x, grid), dim=-1).cuda()
        #print(x.shape)
        x_fc0 = self.fc0(x)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        
        x1_c0 = self.conv0(x_fc0,int(D1*self.factor2),int(D2*self.factor2))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor2),int(D2*self.factor2))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )

        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)
        """
        #Variational Autoencoder part - Extract mu and sigma
        x_c2_1 = torch.flatten(x_c2_1, start_dim=1)
        x_c2_1 = self.linearFlatten1(x_c2_1)
        mu = self.linearFlatten2(x_c2_1)
        sigma = self.linearFlatten3(x_c2_1)
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)
        z = sigma*self.N.sample(mu.shape) + mu
        #z = mu + std*eps

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        z = self.linearUnflatten(z)
        z = self.linearUnflatten2(z)
        x_c2_1 = self.unflatten(z)
        #print("x_c2_1 shape = ", x_c2_1.shape)
        #Variational Autoencoder -end
        """
        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1)

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor2),int(D2*self.factor2))
        x2_c4 = self.w4(x_c3,int(D1*self.factor2),int(D2*self.factor2))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)


        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]

        x_c5 = x_c5.permute(0, 2, 3, 1)
        
        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        
        #x3 = torch.permute(x3, (0, 2,3,1))
        x3 = (x_l+x_r)/2
        #print(x3.shape)
        x3 = torch.permute(x_out, (0,3,1,2))
        #x_fin = x_out + x3
        #x_fin = torch.permute(x_fin, (0, 3, 1, 2))
        #x_out = self.interpolation(x_fin)
        #x_out = torch.permute(x_out, (0, 3, 1, 2))
        #print(x_out.shape)
        #x_out = self.increaseColor(x_out)
        return x3
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).cuda()

class UNO_vanilla(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, pad = 0, factor = 16/4):
        super(UNO_vanilla, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.factor2 = factor/4
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.in_d_co_domain, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.d_co_domain, 4*factor*self.d_co_domain, 16, 16, 42, 42)

        self.conv1 = SpectralConv2d(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16, 21,21)

        self.conv2 = SpectralConv2d(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8,10,10)
        
        self.conv2_1 = SpectralConv2d(16*factor*self.d_co_domain, 32*factor*self.d_co_domain, 4, 4,5,5)
        
        self.conv2_9 = SpectralConv2d(32*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8,5,5)
    

        self.conv3 = SpectralConv2d(32*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16,10,10)

        self.conv4 = SpectralConv2d(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32,21,21)

        self.conv5 = SpectralConv2d(8*factor*self.d_co_domain, self.d_co_domain, 48, 48,42,42) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,4*factor*self.d_co_domain,75, 75) #
        
        self.w1 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 50, 50) #
        
        self.w2 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 25, 25) #
        
        self.w2_1 = pointwise_op(16*factor*self.d_co_domain, 32*factor*self.d_co_domain, 12, 12)

        self.w2_9 = pointwise_op(32*factor*self.d_co_domain, 16*factor*self.d_co_domain, 25, 25)
        
        self.w3 = pointwise_op(32*factor*self.d_co_domain, 8*factor*self.d_co_domain, 50, 50) #
        
        self.w4 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 75, 75)
        
        self.w5 = pointwise_op(8*factor*self.d_co_domain, self.d_co_domain, 100, 100) # will be reshaped

        self.fc1 = nn.Linear(2*self.d_co_domain, 4*self.d_co_domain)
        self.fc2 = nn.Linear(4*self.d_co_domain, self.d_co_domain)

    def forward(self, x):
        
        grid = self.get_grid(x[0].shape, x.device)
        x = x.permute(0,2,3,1)

        x_fc0 = self.fc0(x)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        
        x1_c0 = self.conv0(x_fc0,int(D1*self.factor2),int(D2*self.factor2))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor2),int(D2*self.factor2))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )

        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)

        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1)

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor2),int(D2*self.factor2))
        x2_c4 = self.w4(x_c3,int(D1*self.factor2),int(D2*self.factor2))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)


        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]

        x_c5 = x_c5.permute(0, 2, 3, 1)
        
        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        x_out = x_out.permute(0,3,1,2)
        x_out = F.gelu(x_out)
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).cuda()

class NIO(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, pad = 0, factor = 10/4, patch_size=(1,1), width=256, height=256, weight=0.01):
        super(NIO, self).__init__()
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.factor2 = factor/4
        self.width = width
        self.height = height
        self.weight = weight
        self.tokenize = nn.Conv2d(3, 3, kernel_size=patch_size, stride=patch_size)
        self.UNO_block_1 = UNO(self.in_d_co_domain, d_co_domain, pad=0, factor=16/4)
        #Uncomment the lines below to include ResNet style architecture
        #self.UNO_block_2 = UNO(self.in_d_co_domain, d_co_domain, pad=0, factor=16/4)
        #self.UNO_block_3 = UNO(self.in_d_co_domain, d_co_domain, pad=0, factor=16/4)
        #self.UNO_block_4 = UNO(self.in_d_co_domain, d_co_domain, pad=0, factor=16/4)
        #self.UNO_block_5 = UNO(self.in_d_co_domain, d_co_domain, pad=0, factor=20/4)
    def forward(self, x):
        #uno_1_out = self.UNO_block_1(x)
        #print(uno_1_out.shape)
        #print(x[0].shape)
        #print(x.shape)
        self.height = x[0].shape[2]
        self.width = x[0].shape[3]
        x_int_l = self.tokenize(x[0])
        x_int_r = self.tokenize(x[1])
        x_int = 0.5*x_int_l + 0.5*x_int_r
        #x_int = torch.permute(x_int, (0,2,3,1))
        uno_1_out = self.UNO_block_1(x_int)
        
        """
        RESNET Stuff. Uncomment to add resnet instead of UNO
        y = torch.stack((x[0], uno_1_out))
        
        y = x[0] + uno_1_out
        #print("Y Shape = ", y.shape)
        uno_2_out = self.UNO_block_2(y)
        y = torch.stack((x[1], uno_2_out))
        y = x[1] + uno_2_out
        uno_3_out = self.UNO_block_3(y)
        y = torch.stack((x[0], uno_3_out))
        uno_4_out = self.UNO_block_4(x)
        y = torch.stack((x_int, uno_4_out))
        x = self.UNO_block_5(y)
        """
        uno_1_out = torch.nn.functional.interpolate(uno_1_out, size = (self.height, self.width),mode = 'bicubic',align_corners=True)
        return 0.5*x[0]+0.5*x[1] + self.weight*uno_1_out
        #return 0.01*uno_2_out + 0.99*x[0]
class NIO_split_channel(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, pad=0, factor=3/4):
        super(NIO_split_channel, self).__init__()
        #self.UNO_block_red = UNO_vanilla(1, 1, pad=0, factor=16/4)
        #self.UNO_block_green = UNO_vanilla(1, 1, pad=0, factor=16/4)
        #self.UNO_block_blue = UNO_vanilla(1, 1, pad=0, factor=16/4)
        self.UNO_block_1 = UNO_vanilla(6, 3, pad=0, factor=16/4)
        self.UNO_block_2 = UNO_vanilla(9, 3, pad=0, factor=16/4)
        #self.UNO_block_3 = UNO_vanilla(12, 3, pad=0, factor=16/4)
        #self.UNO_block_4 = UNO_vanilla(15, 3, pad=0, factor=16/4)
        #self.UNO_block_merge = UNO_vanilla(3, 3, pad=0, factor=16/4)
        
    def forward(self, x):
        x_l = x[0].permute(0,3,1,2)
        x_r = x[1].permute(0,3,1,2)
        x_l_colors = x_l.permute(1,0,2,3)
        x_r_colors = x_r.permute(1,0,2,3)
        
        input_stack_1 = torch.cat((x_l, x_r), dim=1)
        
        stack_out_1 = self.UNO_block_1(input_stack_1)

        #print(input_stack.shape)
        #input_sum = (x_l+x_r)/2.0
        
        #input_stack_2 = torch.cat((stack_out_1, input_stack_1), dim=1)
        #print(input_stack_2.shape)
        #stack_out_2 = self.UNO_block_2(input_stack_2)
        
        #input_stack_3 = torch.cat((stack_out_2, input_stack_2), dim=1)
        #stack_out_3 = self.UNO_block_3(input_stack_3)
        
        #input_stack_4 = torch.cat((stack_out_3, input_stack_3), dim=1)
        #stack_out_4 = self.UNO_block_4(input_stack_4)
        
        #add_out = self.UNO_block_add(input_sum)
        #x_out = self.UNO_block_merge(stack_out_4)
        """
        #print(x_l_colors.shape)
        x_l_red = x_l_colors[0]
        x_l_green = x_l_colors[1]
        x_l_blue = x_l_colors[2]
        
        x_r_red = x_r_colors[0]
        x_r_green = x_r_colors[1]
        x_r_blue = x_r_colors[2]
        
        #x_red = torch.stack((x_l_red, x_r_red)).permute(1,0,2,3)
        #x_green = torch.stack((x_l_green, x_r_green)).permute(1,0,2,3)
        #x_blue = torch.stack((x_l_blue, x_r_blue)).permute(1,0,2,3)
        x_red = (x_r_red + x_l_red).reshape(x.shape[1], 1, x.shape[2], x.shape[3])
        x_green = (x_r_green + x_l_green).reshape(x.shape[1], 1, x.shape[2], x.shape[3])
        x_blue = (x_l_blue + x_r_blue).reshape(x.shape[1], 1, x.shape[2], x.shape[3])
        
        r = self.UNO_block_red(x_red)
        g = self.UNO_block_green(x_green)
        b = self.UNO_block_blue(x_blue)
        uno_x = torch.cat((r,g,b), dim=1) + 0.001*self.sub_mean(x_l)[0] + 0.001*self.sub_mean(x_r)[0]
        x_out = self.UNO_block_merge(uno_x)
        """
        return stack_out_1
    
    def sub_mean(self, x):
        mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
        x -= mean
        return x, mean
        
class NIO_fine(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, pad=0, factor=3/4):
        super(NIO_fine, self).__init__()
        self.in_d_co_domain = in_d_co_domain
        self.d_co_domain = d_co_domain
        self.factor = factor
        self.factor2 = factor/4
        self.UNO_block = UNO(self.in_d_co_domain, d_co_domain, pad=0, factor=16/4)
        #self.UNO_block2 = UNO(self.in_d_co_domain, d_co_domain, pad=0, factor=12/4)
        """self.layer1 = nn.Conv2d(3, 3, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(3)
        self.layer2 = nn.Conv2d(3, 3, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(3)
        self.layer3 = nn.Conv2d(3, 3, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(3)
        self.layer4 = nn.Conv2d(3, 3, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(3)
        self.layer5 = nn.Conv2d(3, 3, 3, padding=1)
        self.norm5 = nn.BatchNorm2d(3)
        #self.layer2 = nn.Conv2d(3, 3, 3, padding=1)"""
    def forward(self, x):
        UNO_out = self.UNO_block(x)
        #UNO_out = torch.stack((x[0], UNO_out))
        #UNO_out = self.UNO_block2(UNO_out)
        #UNO_out = torch.permute(UNO_out, (0,3, 1, 2))
        #layer1out = self.norm1(self.layer1(UNO_out))
        #layer2out = self.norm2(self.layer2(layer1out+UNO_out))
        #layer3out = self.norm3(self.layer3(layer2out+layer1out))
        #layer4out = self.norm4(self.layer4(layer3out+layer2out))
        #layer5out = self.norm5(self.layer5(layer4out+layer3out))
        #layer5out = torch.permute(layer5out, (0,2,3,1))
        #output = layer5out
        return UNO_out

class NIO_single(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, pad=0, factor=3/4):
        super(NIO_single, self).__init__()
        self.UNO_block_1 = UNO_vanilla(3, 3, pad=0, factor=16/4)
        #self.UNO_block_2 = UNO_vanilla(9, 3, pad=0, factor=16/4)
        
    def forward(self, x):
        
        input_stack_1 = x
        
        stack_out_1 = self.UNO_block_1(input_stack_1)
        return stack_out_1