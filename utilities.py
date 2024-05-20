import torch
import numpy as np

def average_C(data, n_c = 21):
    if not torch.is_tensor(data):
        data = torch.tensor(data,dtype=torch.int16)
    
    eps = 1e-4
    dataset_onehot = torch.eye(n_c)[data.long()]
    freqs = dataset_onehot.mean(0)
    #freqs = torch.clamp(freqs, min=eps, max=(1. - eps))
    
    return freqs

def covariance_C(data, n_c = 21):
    if not torch.is_tensor(data):
        data = torch.tensor(data,dtype=torch.int16)
    
    num_data, L = data.size()
    dataset_onehot = torch.eye(n_c)[data.long()].reshape(-1, n_c * L).float()

    weights = torch.ones((num_data, 1), dtype=torch.float32)
    norm_weights = weights.reshape(-1, 1) / weights.sum()
    data_mean = (dataset_onehot * norm_weights).sum(0, keepdim=True)
    cov_matrix_torch = ((dataset_onehot * norm_weights).mT @ dataset_onehot) - (data_mean.mT @ data_mean)
    
    return cov_matrix_torch.reshape(L,n_c,L,n_c).permute(0,2,1,3)


def compute_output_Potts_C(data, W, n_c = 21):
    if not torch.is_tensor(data):
        data = torch.tensor(data,dtype=torch.int16)
        
    dataset_onehot = torch.eye(n_c)[data.long()].float()
    
    return torch.einsum("lkaj,ikj->ila",W, dataset_onehot )

def dot_Potts_C(data, h, n_c = 21):
    if not torch.is_tensor(data):
        data = torch.tensor(data,dtype=torch.int16)
        
    dataset_onehot = torch.eye(n_c)[data.long()].float()
    return torch.tensordot(dataset_onehot, h)

def bilinear_form_Potts(data, W, n_c = 21):
    if not torch.is_tensor(data):
        data = torch.tensor(data,dtype=torch.int16)
        
    dataset_onehot = torch.eye(n_c)[data.long()].float()
    
    bilin_form = torch.einsum("lkab,ila,ikb->i",W,dataset_onehot,dataset_onehot)
    
    return bilin_form

def Potts_sampling( x, fields_eff, B, N, n_c, fields0, couplings,beta = 1):
    
    if not torch.is_tensor(fields_eff):
        fields_eff = torch.tensor(fields_eff, dtype = torch.float32)
    if not torch.is_tensor( x ):
        x = torch.is_tensor(x, dtype=torch.int32 )
    
    for _ in range(N):
        x_old = x.detach().clone()
        positions = torch.randint(0,N, size = [B])
        probs = torch.exp( beta*fields_eff[torch.arange(B),positions, : ] + (1 - beta) * fields0[torch.arange(B),positions, :])
        probs /= torch.sum(probs,dim= 1, keepdim = True )
        
        tmp_mut = torch.multinomial( probs, num_samples = 1, replacement = True).to(torch.int32).squeeze(dim = 1)
        x[torch.arange(B), positions ] = tmp_mut
        
        couplings_diff = couplings[:, positions, :, x[torch.arange(B), positions] ] -  couplings[:, positions, :, x_old[torch.arange(B), positions] ]
        #couplings_diff = couplings_diff.unsqueeze(1)
        fields_eff +=  couplings_diff
        
    return x, fields_eff

def invert_softmax(mu, eps=1e-6, gauge='zerosum'):
    n_c = mu.shape[1]
    fields = torch.log( (1 - eps) * mu + eps / n_c)
    if gauge == 'zerosum':
        fields -= fields.sum(1)[:, np.newaxis] / n_c
    return fields

def reshape_in(x, xdim=2):
    xshape = list(x.shape)
    ndims = len(xshape)
    if ndims == xdim:
        x = x[None]
    elif ndims > xdim + 1:
        x = x.reshape([np.prod(xshape[:-xdim])] + xshape[-xdim:])
    return x, xshape


def reshape_out(y, xshape, xdim=2):
    ndims = len(xshape)
    if ndims == xdim:
        return y[0]
    elif ndims > xdim + 1:
        return y.reshape(list(xshape[:-xdim]) + list(y.shape)[1:])
    else:
        return y


def gauge_adjust_couplings(W, gauge='zerosum'):
    if gauge == 'zerosum':
        return W
    else:
        print('adjust_couplings -> gauge not supported. Setting zerosum gauge')
        return W
