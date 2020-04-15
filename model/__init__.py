import torch
import torch.nn as nn

from model.pointnet import SimplePointnet
from model.decoder import PointDecoder

class RegistrationNetwork(nn.Module):

    def __init__(self, c_dim=128, dim=3, hidden_dim=512, n_points=1024, decoder=False, device=None):
        super().__init__()

        self.encoder = SimplePointnet(c_dim=c_dim, dim=dim, hidden_dim=hidden_dim)

        if decoder:
            self.decoder = PointDecoder(dim=dim, c_dim=c_dim, n_points=n_points)
        else:
            self.decoder = None
        
        self._device = device

    def forward(self, pointcloud, **kwargs):
        '''
        Forward pass through the network.

        Args:
            pointcloud (tensor): pointcloud to embed.
        '''

        c = self.encoder(pointcloud)
    
        if self.decoder is not None:
            pointcloud_reconstruct = self.decoder(c)

        result = {}
        result['c'] = c
        result['p'] = pointcloud_reconstruct

        return result
