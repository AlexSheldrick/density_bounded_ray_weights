from platform import java_ver
import torch
from pytorch_lightning import LightningModule
from models.mip_nerf import MipNerf
from models.mip import rearrange_render_image, distloss
from models.loss import loss_dict
from utils.metrics import calc_psnr
from datasets import dataset_dict
from collections import defaultdict, namedtuple
import collections
import numpy as np
import math
from torchvision.transforms.functional import equalize
from PIL import Image
from pathlib import Path


from utils.lr_schedule import MipLRDecay
from torch.utils.data import DataLoader
from utils.vis import visualize_depth, visualize_normal, l2_normalize, to8b

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'depth', 'normal', 'mask', 'depth_vars', 'cam_idx'))


class MipNeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(MipNeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        if self.hparams['precision'] == 16: 
            self.amp = True
            print('training in 16bit precision')
        else: self.amp = False
        #self.automatic_optimization=False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.train_randomized = hparams['train.randomized']
        self.val_randomized = hparams['val.randomized']
        self.white_bkgd = hparams['train.white_bkgd']
        self.val_chunk_size = hparams['val.chunk_size']
        self.batch_size = hparams['train.batch_size']
        self.compute_density_normals = hparams['nerf.compute_density_normals']
        self.mlp_normals =  hparams['nerf.mlp.mlp_normals']
        self.loss = loss_dict['mse'](hparams)
        self.mip_nerf = MipNerf(
            num_samples=hparams['nerf.num_samples'],
            num_levels=hparams['nerf.num_levels'],
            resample_padding=hparams['nerf.resample_padding'],
            stop_resample_grad=hparams['nerf.stop_resample_grad'],
            use_viewdirs=hparams['nerf.use_viewdirs'],
            disparity=hparams['nerf.disparity'],
            depth_sampling=hparams['nerf.depth_sampling'],
            ray_shape=hparams['nerf.ray_shape'],
            min_deg_point=hparams['nerf.min_deg_point'],
            max_deg_point=hparams['nerf.max_deg_point'],
            deg_view=hparams['nerf.deg_view'],
            density_activation=hparams['nerf.density_activation'],
            density_noise=hparams['nerf.density_noise'],
            density_bias=hparams['nerf.density_bias'],
            rgb_activation=hparams['nerf.rgb_activation'],
            rgb_padding=hparams['nerf.rgb_padding'],
            disable_integration=hparams['nerf.disable_integration'],
            append_identity=hparams['nerf.append_identity'],
            mlp_net_depth=hparams['nerf.mlp.net_depth'],
            mlp_net_width=hparams['nerf.mlp.net_width'],
            mlp_net_depth_condition=hparams['nerf.mlp.net_depth_condition'],
            mlp_net_width_condition=hparams['nerf.mlp.net_width_condition'],
            mlp_skip_index=hparams['nerf.mlp.skip_index'],
            mlp_num_rgb_channels=hparams['nerf.mlp.num_rgb_channels'],
            mlp_num_density_channels=hparams['nerf.mlp.num_density_channels'],
            mlp_net_activation=hparams['nerf.mlp.net_activation'],
            prop_mlp=hparams['nerf.mlp.prop_mlp'],
            num_glo_embeddings = hparams['nerf.mlp.num_glo_embeddings'],
            num_glo_features = hparams['nerf.mlp.num_glo_features'],
            mlp_normals=hparams['nerf.mlp.mlp_normals'],
        )
        self.val_idx = 0


    def forward(self, batch_rays: torch.Tensor, randomized: bool, white_bkgd: bool, compute_normals: bool = False, eps = 1.0, zero_glo = True):
        # TODO make a multi chunk
        res = self.mip_nerf(batch_rays, randomized, white_bkgd, compute_normals, eps, zero_glo)  # num_layers result
        return res

    def setup(self, stage):
        dataset = dataset_dict[self.hparams['dataset_name']]

        self.train_dataset = dataset(data_dir=self.hparams['data_path'],
                                     split='train',
                                     white_bkgd=self.hparams['train.white_bkgd'],
                                     batch_type=self.hparams['train.batch_type'],
                                     num_images=self.hparams['train.num_images']
                                     )
        self.val_dataset = dataset(data_dir=self.hparams['data_path'],
                                   split='val',
                                   white_bkgd=self.hparams['val.white_bkgd'],
                                   batch_type=self.hparams['val.batch_type']
                                   )
        self.test_dataset = dataset(data_dir=self.hparams['data_path'],
                                   split='test',
                                   white_bkgd=self.hparams['val.white_bkgd'],
                                   batch_type=self.hparams['val.batch_type']
                                   )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.mip_nerf.parameters(), lr=self.hparams['optimizer.lr_init'], fused= False) #self.amp)
        optimizer.zero_grad(set_to_none=True)
        scheduler = MipLRDecay(optimizer, self.hparams['optimizer.lr_init'], self.hparams['optimizer.lr_final'],
                               self.hparams['optimizer.max_steps'], self.hparams['optimizer.lr_delay_steps'],
                               self.hparams['optimizer.lr_delay_mult'])
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams['train.num_work'],
                          batch_size=self.hparams['train.batch_size'],
                          pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        # must give 1 worker
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=1,
                          # validate one image (H*W rays) at a time
                          batch_size=1,
                          pin_memory=True,
                          persistent_workers=True)

    def test_dataloader(self):
        # must give 1 worker
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=1,
                          # validate one image (H*W rays) at a time
                          batch_size=1,
                          pin_memory=True,
                          persistent_workers=True)
                          
    def training_step(self, batch, batch_nb):
        eps = max(0.45 - 0.4*math.sqrt(4*self.global_step/(self.hparams['optimizer.max_steps'])), 0.15)
        #eps = 0.1
        #if self.global_step < 8000:
        #eps = max(0.2 - 0.2*math.sqrt(4*self.global_step/(self.hparams['optimizer.max_steps'])), 0.04)
        #else:
        #    eps = max(0.09 - 0.07*3*(self.global_step-8000)/(self.hparams['optimizer.max_steps']) , 0.02)
        #0.17
        #eps = 0.1
        #eps = 0.08 #, 0.12
        #eps = 0.03
        rays, rgbs = batch
        ret = self(rays, self.train_randomized, self.white_bkgd, self.compute_density_normals, eps, zero_glo = False)
        targets = {'rgb': rgbs[..., :3], 'depth': rays.depth.view(-1), 'normal': rays.normal, 'dirs': rays.viewdirs, 'var':rays.depth_vars, 'dmask': rays.mask.view(-1)}
        loss_dict = self.loss(ret, targets, step = self.global_step)
        self.logging(loss_dict, mode='train') 
        self.log('s', ret['s'])        
        return loss_dict['total']
        
    def validation_step(self, batch, batch_nb):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
            rays, rgbs = batch
            ret = self.render_image(batch, chunk_size=self.val_chunk_size)
            targets = {'rgb': rgbs[..., :3].view(-1,3), 'depth': rays.depth.view(-1), 'normal': rays.normal.view(-1,3), 'dirs': rays.viewdirs.view(-1,3), 
                        'mask': ret['mask'], 'var':rays.depth_vars, 'dmask': rays.mask.view(-1)} #
            loss_dict = self.loss(ret, targets, mask=ret['mask'])
            if batch_nb == 4: #self.val_idx:
                self.write_imgs_to_tensorboard(ret, rays, rgbs, 'val')
            return loss_dict

    def test_step(self, batch, batch_nb):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
            rays, rgbs = batch
            with torch.inference_mode(False):                
                ret = self.render_image(batch, chunk_size=self.val_chunk_size)
        make_img = lambda x: (torch.clamp(x, min=0, max=1) * 255).to(torch.uint8)
        rgbs = equalize(make_img(rgbs[..., :3]))
        ret['rgb_fine'] = make_img(equalize(ret['rgb_fine'].view(rgbs.shape)).view(-1,3))
        targets = {'rgb': rgbs[..., :3].view(-1,3), 'depth': rays.depth.view(-1), 'normal': rays.normal.view(-1,3), 'dirs': rays.viewdirs.view(-1,3), 
                        'mask': ret['mask'], 'var':rays.depth_vars, 'dmask': rays.mask.view(-1)} #

        loss_dict = self.loss(ret, targets, mask=ret['mask'])
        self.write_imgs_to_tensorboard(ret, rays, rgbs, 'test', '_img_'+str(batch_nb))
        return loss_dict

    def validation_epoch_end(self, outputs):
        self.val_idx = torch.randint(low=0, high=12, size=(1,))[0]
        self.mean_metrics(outputs, mode='val')
    
    def test_epoch_end(self, outputs):
        try:
            mean_loss = torch.stack([x['total'] for x in outputs]).mean()
            mean_rgb = torch.stack([x['rgb'] for x in outputs]).mean()
            mean_depth = torch.stack([x['depth'] for x in outputs]).mean()
            mean_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
            mean_normal = torch.stack([x['normal'] for x in outputs]).mean()
            mean_orientation = torch.stack([x['orientation'] for x in outputs]).mean()
            mean_distortion = torch.stack([x['distortion'] for x in outputs]).mean()
            mean_near = torch.stack([x['near'] for x in outputs]).mean()
            mean_empty = torch.stack([x['empty'] for x in outputs]).mean()
            mean_envelope = torch.stack([x['envelope'] for x in outputs]).mean()
            mean_normal_loss_lowerbounded = torch.stack([x['normal_loss_lowerbounded'] for x in outputs]).mean()
            mean_chord_loss = torch.stack([x['chord'] for x in outputs]).mean()
            #mean_incomplete = torch.stack([x['incomplete'] for x in outputs]).mean()

            mean_losses = {'total': mean_loss, 'rgb': mean_rgb, 'depth': mean_depth, 'psnr': mean_psnr, 'normal': mean_normal,
                            'orientation': mean_orientation, 'distortion': mean_distortion, 'near': mean_near, 'empty': mean_empty,
                            'chord': mean_chord_loss, 'normal_loss_lowerbounded': mean_normal_loss_lowerbounded, 'envelope': mean_envelope#,
                            #'incomplete': mean_incomplete
                            #'expected_depth': mean_expected_depth, 'termination': mean_termination
                            }
            self.logging(mean_losses, mode='test')
        except:
            pass

    def mean_metrics(self, outputs, mode='val'):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
            ret, rays, rgbs = self.render_image(None, mode='train', zero_glo=False)
            self.write_imgs_to_tensorboard(ret, rays, rgbs, 'train')
        try:
            mean_loss = torch.stack([x['total'] for x in outputs]).mean()
            mean_rgb = torch.stack([x['rgb'] for x in outputs]).mean()
            mean_depth = torch.stack([x['depth'] for x in outputs]).mean()
            mean_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
            mean_normal = torch.stack([x['normal'] for x in outputs]).mean()
            mean_orientation = torch.stack([x['orientation'] for x in outputs]).mean()
            mean_distortion = torch.stack([x['distortion'] for x in outputs]).mean()
            mean_near = torch.stack([x['near'] for x in outputs]).mean()
            mean_empty = torch.stack([x['empty'] for x in outputs]).mean()
            mean_envelope = torch.stack([x['envelope'] for x in outputs]).mean()
            mean_normal_loss_lowerbounded = torch.stack([x['normal_loss_lowerbounded'] for x in outputs]).mean()
            mean_chord_loss = torch.stack([x['chord'] for x in outputs]).mean()
            #mean_incomplete = torch.stack([x['incomplete'] for x in outputs]).mean()

            mean_losses = {'total': mean_loss, 'rgb': mean_rgb, 'depth': mean_depth, 'psnr': mean_psnr, 'normal': mean_normal,
                            'orientation': mean_orientation, 'distortion': mean_distortion, 'near': mean_near, 'empty': mean_empty,
                            'chord': mean_chord_loss, 'normal_loss_lowerbounded': mean_normal_loss_lowerbounded, 'envelope': mean_envelope#,
                            #'incomplete': mean_incomplete
                            #'expected_depth': mean_expected_depth, 'termination': mean_termination
                            }
            self.logging(mean_losses, mode=mode)
        except:
            pass

    def logging(self, loss_dict, mode='train'):
        if mode == 'train': 
            self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])            
        self.log(f'{mode}/loss', loss_dict['total'])
        self.log(f'{mode}/rgb', loss_dict['rgb'])
        self.log(f'{mode}/depth', loss_dict['depth'])
        self.log(f'{mode}/normal', loss_dict['normal'])
        self.log(f'{mode}/chord', loss_dict['chord'])
        self.log(f'{mode}/orientation', loss_dict['orientation'])
        self.log(f'{mode}/distortion', loss_dict['distortion'])
        self.log(f'{mode}/psnr', loss_dict['psnr'], prog_bar=True)
        self.log(f'{mode}/near', loss_dict['near'])        
        self.log(f'{mode}/empty', loss_dict['empty'])        
        self.log(f'{mode}/normal_loss_lowerbounded', loss_dict['normal_loss_lowerbounded'])        
        self.log(f'{mode}/envelope', loss_dict['envelope'])    
        #self.log(f'{mode}/incomplete', loss_dict['incomplete'])  

    #def on_before_backward(self, loss: torch.Tensor) -> None:
    #    return self.scaler.scale(loss) / self.hparams['train.batch_size']

    def render_image(self, batch, mode='val', chunk_size = None, zero_glo = True):
        if chunk_size == None: chunk_size = self.val_chunk_size
        results = defaultdict(list)
        if mode=='val':
            rays, rgbs = batch
            rgbs = rgbs[..., :3]
        elif mode=='train':
            N, H, W, C = (1, self.train_dataset.h, self.train_dataset.w, 3)
            if self.train_dataset.n_examples < 3:
                img_idx = np.random.choice(np.arange((self.train_dataset.n_examples)))
            else: img_idx = 2 #starts from 0
            rays, rgbs = self.train_dataset[W*H*img_idx:W*H*(1+img_idx)]
            rays = [torch.from_numpy(getattr(rays, key)).to(self.device) for key in rays._fields]
            rays = Rays(*[rays_attr for rays_attr in rays])
            #rays = rays.to(self.device)
            rgbs = torch.from_numpy(rgbs).reshape(N, H, W, C).to(self.device)
            
        elif mode=='test':
            rays = batch['rays']
        
        single_image_rays, val_mask = rearrange_render_image(rays, chunk_size)
        
        with torch.no_grad():
            for batch_rays in single_image_rays:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
                    ret_chunks  = self(batch_rays, self.val_randomized, self.white_bkgd, compute_normals=True, zero_glo=zero_glo)
                    for k, v in ret_chunks.items():
                        results[k] += [v]                    
        
            for k, v in results.items():
                results[k] = torch.cat(v, 0)

        results['mask'] = val_mask.view(-1).to(torch.bool)
        
        if mode == 'test' or mode =='val':
            return results
        elif mode == 'train':
            return results, rays, rgbs #hacky way to do train rendering

    def write_imgs_to_disk(self, results):
        depth_pred = results['depth']
        pass
        # save rgb normals, depth for every frame
    
    def make_movie(self, path):
        pass
        #collect all the images in given path and make a gif

    def write_imgs_to_tensorboard(self, results: dict, rays: namedtuple, rgbs: torch.Tensor, mode: bool='val', batch_nb: str ='') -> None: 
        N, H, W, C = rgbs.shape  # N H W C
        #depth
        depth_pred = results['depth']
        depth_pred = visualize_depth(depth_pred.view(1, H, W)) # H W
        depth_gt = visualize_depth(rays.depth.view(H,W))

        #normals
        #depth_mask = (rays.depth < 0.1).view(-1) # (H*W)
        #normals_pred = results['normal']        #(H*W,3)
        normals_gt = rays.normal.view(H, W, 3)#.permute(2, 0, 1).cpu() # (3, H, W)
        normals_gt = visualize_normal(normals_gt) 
        #weight_mask = (torch.cumsum(results['weights'], dim = -1) < 0.75).to(results['weights'].dtype).unsqueeze(-1)
        mask = (~torch.any(rays.normal, dim=-1)) #(1,H,W)
        mask = mask.view(H*W) #(H,W)
        if not self.mlp_normals:
            normals_pred = torch.sum(results['weights'][...,None]*results['density_normals'], dim=1) # (H*W, N_samples + Samples_fine (128), 3) --> (H*W, 3)
        else:
            normals_pred = torch.sum(results['weights'][...,None]*results['mlp_normals'], dim=1) # (H*W, N_samples + Samples_fine (128), 3) --> (H*W, 3)
        
        normals_pred = l2_normalize(normals_pred)
        #normals_pred[depth_mask,...] = 0
        normals_pred[mask, ...] = 0
        #normals_pred = (normals_pred + 1)/2
        #normals_pred[mask, ...] = 0
        normals_pred = visualize_normal(normals_pred.view(H,W,3))     
        #rgb
        #coarse_rgb = results['rgb_coarse'].view(N, H, W, C)  # N H W C
        fine_rgb = results['rgb_fine'].view(N, H, W, C)  # N H W C

        canvas = np.zeros((3,2*H,3*W))
        canvas[:, 0:H, 0:W] = rgbs.squeeze(0).permute(2, 0, 1).cpu()
        canvas[:, 0:H, W:2*W] = depth_gt
        canvas[:, 0:H, 2*W:3*W] = normals_gt.squeeze(0).permute(2, 0, 1).cpu()
        canvas[:, H:2*H, 0:W] = fine_rgb.squeeze(0).permute(2, 0, 1).cpu()
        canvas[:, H:2*H, W:2*W] = depth_pred
        canvas[:, H:2*H, 2*W:3*W] = normals_pred.squeeze(0).permute(2, 0, 1).cpu()
        self.logger.experiment.add_image(f'{mode}/GT_pred{batch_nb}', canvas, self.global_step)
        
        im = Image.fromarray((canvas.transpose(1,2,0)*255).astype(np.uint8))
        out_path = Path(self.hparams['out_dir'], 'images')
        out_path.mkdir(exist_ok=True)
        out_path = Path(out_path, self.hparams['exp_name'])
        out_path.mkdir(exist_ok=True)
        im.save(str(out_path) + f"/{mode}_{self.global_step}.jpeg")
