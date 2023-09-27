import torch
from torch import nn
import numpy as np

class MSELoss(nn.Module):
    def __init__(self, hparams):
        super(MSELoss, self).__init__()
        self.MSEloss = nn.MSELoss(reduction='mean')
        self.lambda_depth = hparams['loss.lambda_depth']
        self.lambda_normal = hparams['loss.lambda_normal']
        self.lambda_distortion = hparams['loss.lambda_distortion']
        self.lambda_orientation = hparams['loss.lambda_orientation']
        self.coarse_mult_loss = hparams['loss.coarse_loss_mult']
        self.lambda_empty_loss = hparams['loss.lambda_empty_loss']
        self.lambda_near_loss = hparams['loss.lambda_near_loss']
        self.lambda_prop_loss = hparams['loss.lambda_prop_loss']
        self.prop_mlp=hparams['nerf.mlp.prop_mlp'] 
        self.max_steps = hparams['optimizer.max_steps']
        self.mlp_normals = hparams['nerf.mlp.mlp_normals']
    
    def forward(self, results, targets, mask=None, step = 1):
        #if mask is None: 
        mask = None
        dmask = (targets['dmask'])
        depth_loss = self.MSEloss(results['depth'][~dmask], targets['depth'][~dmask])
        
        if self.mlp_normals:
            normal_loss = norm_loss_func(pred_ = results['mlp_normals'], gt_ = results['density_normals'], w_ = results['weights'], self_supervised=True)
            chord_loss = chord_distance(pred_ = results['mlp_normals'], gt_ = results['density_normals'], w_ = results['weights'], self_supervised=True)

            mlp_normal_loss = norm_loss_func(pred_ = results['mlp_normals'], gt_ = targets['normal'], w_ = results['weights'])
            mlp_chord_loss = chord_distance(pred_ = results['mlp_normals'], gt_ = targets['normal'], w_ = results['weights'])

            normal_loss = (normal_loss + mlp_normal_loss)/2
            chord_loss = (chord_loss + mlp_chord_loss)/2
        else:             
            normal_loss = norm_loss_func(pred_ = results['density_normals'], gt_ = targets['normal'], w_ = results['weights'])
            chord_loss = chord_distance(pred_ = results['density_normals'], gt_ = targets['normal'], w_ = results['weights'])
        
        orientation_loss = orientation_loss_func(results['density_normals'], targets['normal'], results['weights'], targets['dirs'])
        normal_loss_lowerbounded = results['normal_lowerbound_error'].sum()
        #distortion_loss = self.lossfun_distortion(results['t_samples'], results['weights'])
        distortion_loss = results['distortion'].mean()
        near_loss = results['near_loss']#.sum()
        empty_loss = results['empty_loss']

        incomplete_penalty = 0. #results['distortion'][~dmask].mean()


        if mask is not None:               
            #rgb_fine = self.MSEloss(results['rgb_fine'][mask, :], targets['rgb'][mask, :])     
            rgb_fine = charbonnier_loss(results['rgb_fine'][mask, :], targets['rgb'][mask, :])       
            if self.prop_mlp: rgb_coarse = 0
            else: rgb_coarse = self.MSEloss(results['rgb_coarse'][mask, :], targets['rgb'][mask, :])
            rgb_loss = rgb_fine + self.coarse_mult_loss * rgb_coarse
            with torch.no_grad(): psnr = calc_psnr(results['rgb_fine'][mask, :], targets['rgb'][mask, :])
        else:
            #rgb_fine = self.MSEloss(results['rgb_fine'], targets['rgb'])        
            rgb_fine = charbonnier_loss(results['rgb_fine'], targets['rgb'])     
            if self.prop_mlp: rgb_coarse = 0     
            else: rgb_coarse = self.MSEloss(results['rgb_coarse'], targets['rgb'])  
            rgb_loss = rgb_fine + self.coarse_mult_loss * rgb_coarse
            with torch.no_grad(): psnr = calc_psnr(results['rgb_fine'], targets['rgb'])
        
        
        #anneal_lr = max(1.0 - 2.5*step/self.max_steps, 0)
        #anneal_distortion = max(1.0 - 5*step/self.max_steps, 0)
        anneal_lr = 1
        #anneal_lr_reverse = 2 - anneal_lr
        total_loss = rgb_loss +  \
                        (self.lambda_depth * depth_loss + \
                        #self.lambda_normal * normal_loss_lowerbounded + \
                        self.lambda_normal * normal_loss + \
                        #self.lambda_normal * chord_loss + \
                        self.lambda_orientation * orientation_loss + \
                        self.lambda_distortion * distortion_loss + \
                        #anneal_distortion * self.lambda_distortion * distortion_loss + \
                        anneal_lr * self.lambda_near_loss * near_loss + \
                        anneal_lr * self.lambda_empty_loss * empty_loss #+ 0.1*self.lambda_near_loss * incomplete_penalty \
                        )

        loss_dict =     {'total': total_loss, 
                        'rgb': rgb_loss, 
                        'depth': depth_loss, 
                        'normal': normal_loss, 
                        'psnr': psnr, 
                        'orientation': orientation_loss, 
                        'distortion': distortion_loss, 
                        'empty': empty_loss, 
                        'near': near_loss,
                        'normal_loss_lowerbounded': normal_loss_lowerbounded,
                        'chord': chord_loss,
                        'incomplete': incomplete_penalty
                        } 
                        
        if self.prop_mlp:
            loss_dict['envelope'] = results['envelope_loss'].mean()
            loss_dict['total'] = loss_dict['total'] + self.lambda_prop_loss * loss_dict['envelope']
        else:
            loss_dict['envelope'] = 0.
            
        return loss_dict

    def GNNL_Loss(self, depth_pred, depth_gt, var, depth_gt_var):
        mask = torch.logical_or(var > depth_gt_var, torch.abs(depth_gt - depth_pred) > depth_gt_var)
        #mask = torch.abs(depth_gt.squeeze() - depth_pred.squeeze()) > depth_gt_var.squeeze()
        
        loss = self.GNNLoss(depth_pred[mask], depth_gt[mask], var[mask])
        #loss = ((depth_pred[mask] - depth_gt[mask])**2 ).mean() #/ var**2
        #loss = nn.MSELoss()(depth_pred[mask].squeeze(), depth_gt[mask].squeeze())
        if torch.isnan(loss): loss = 0.
        return loss

    def lossfun_distortion(self, t, w):   #t=z_vals, w=weights. Loss from mip-nerf 360
        """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
        # The loss incurred between all pairs of intervals.
        ut = (t[..., 1:] + t[..., :-1]) / 2
        dut = torch.abs(ut[..., :, None] - ut[..., None, :])
        loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, axis=-1), axis=-1)

        # The loss incurred within each individual interval with itself.
        loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), axis=-1) / 3

        return loss_inter + loss_intra

def charbonnier_loss(x: torch.Tensor, y: torch.Tensor, eps = 0.001):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps)) #omitted eps**2

#### INDEXING TENSORS REQUIRES GPU SYNC. WRITE GATHER INSTRUCTION INSTEAD
@torch.jit.script
def calc_mse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    return torch.mean((x - y) ** 2)

@torch.jit.script
def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr


@torch.jit.script
def calc_psnr_(mse: torch.Tensor):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    psnr = -10.0 * torch.log10(mse)
    return psnr

#cosine similarity
#multiplying with binary vector (in floats) is faster than masked indexing because it doesnt call cuda::nonzero
#sum over sum of mask to have proper mean
@torch.jit.script
def norm_loss_func(pred_: torch.Tensor, gt_: torch.Tensor, w_: torch.Tensor, self_supervised: bool = False):
    # use only those normals that have non-zero (or above threshhold) depth
    #mask = targets['mask'] # (BS)
    n_pred = pred_ # (BS, num_samples, 3)
    n = gt_ # (BS, 3)
    if not self_supervised: 
        mask = (torch.any(gt_, dim=-1))
        valid_pixels = mask.sum()
        w = w_ * mask[..., None] # (BS, num_raysamples) --> set invalid pixels to zero (by making all the sample weights zero via mask)        
        # MAE of Normal alignment
        loss = (1 -  torch.sum(n[:,None,:] * n_pred, dim=-1))
        loss = torch.sum(w * loss, -1) # --> (BS, Num_samples)
        loss = (loss).sum(dim=-1) # --> (BS)
        loss = torch.sum(loss) / valid_pixels # --> float, mean via valid pixels rather than batch size
    else:
        loss = torch.mean((w_ * (1.0 - torch.sum(n * n_pred, dim=-1))).sum(dim=-1))
    return loss

def l2_normalize(x, eps=1e-6):        
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(torch.maximum(torch.sum(x**2, axis=-1, keepdims=True), torch.as_tensor(eps, x.device)))

@torch.jit.script
def orientation_loss_func(pred_: torch.Tensor, gt_: torch.Tensor, w_: torch.Tensor, dirs: torch.Tensor):
        # use only those normals that have non-zero (or above threshhold) depth
        #mask = targets['mask'] # (BS) 
        mask = (torch.any(gt_, dim=-1))
        valid_pixels = mask.sum()
        w = w_ * mask[..., None] # (BS, N_raysamples)
        n = pred_#(BS, N_raysamples, 3)

        # Negate viewdirs to represent normalized vectors from point to camera.
        v = -1. * dirs # (BS, 3)
        n_dot_v = (n * v[..., None, :]) #(BS, N_raysamples, 3)
        n_dot_v = n_dot_v.sum(dim=-1) # (BS, N_raysamples)
        return torch.sum((w * torch.minimum(
            torch.as_tensor(0.0, device=v.device), n_dot_v)**2
            ).sum(dim=-1))/valid_pixels # --> float, mean via valid pixels rather than batch size

#@torch.jit.script
#chord distance (Euclidean distance on a unit circle/circumference, which is the subspace of distances between normalized vectors)   
def chord_distance(pred_: torch.Tensor, gt_: torch.Tensor, w_: torch.Tensor, self_supervised: bool=False):
     # use only those normals that have non-zero (or above threshhold) depth
    n_pred = pred_ # (BS, num_samples, 3)
    n = gt_ # (BS, 3)
    if not self_supervised:
        mask = (torch.any(gt_, dim=-1))
        valid_pixels = mask.sum()
        w = w_ * mask[..., None] # (BS, N_raysamples)
        loss = n[:,None,:] - n_pred # (BS, num_samples, 3)
        loss = torch.norm(loss, dim=-1) # (BS, num_samples, 1)
        loss = (loss**2)/2 # (BS, num_samples)
        loss = torch.sum(w * loss, dim=1) # (BS) per ray loss
        loss = torch.sum(loss)/valid_pixels # float, averaged by valid pixels
    else:
        loss = torch.mean(torch.sum(w_*0.5*torch.norm(n - n_pred, dim=-1)**2, dim=1))
    return loss

loss_dict = {'mse': MSELoss}

if __name__ == "__main__":
    hparams = {'loss.lambda_depth': 0.1,'loss.lambda_normal': 0.0, 'loss.lambda_distortion': 0.0,
                'loss.lambda_orientation': 0.0, 'loss.coarse_loss_mult': 0.1}
    
    loss = MSELoss(hparams)
    results = {'rgb_coarse':torch.rand((2048, 3), device='cuda'),'rgb_fine':torch.rand((2048, 3), device='cuda'), 
                'depth': torch.rand((2048, 1), device='cuda'), 'normal': torch.rand((2048, 128, 3), device='cuda'),           
                'weights':torch.rand((2048, 128), device='cuda') }
    targets = {'rgb':torch.rand((2048, 3), device='cuda'), 'depth': torch.rand((2048, 1), device='cuda'), 'normal': torch.rand((2048, 3), device='cuda'),
            'weights':torch.rand((2048, 128), device='cuda'), 'dirs': torch.rand((2048, 3), device='cuda') }
    targets['mask'] = (targets['depth'] > 0.6).to(torch.float).squeeze().to('cuda')
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            loss(results, targets)
        pass
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))