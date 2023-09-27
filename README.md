# Probabilistically Bounded Ray Densities from Depthmaps

This scene has been reconstructed from 18 images.

https://github.com/AlexSheldrick/mipnerf_pl/assets/59337109/a222517e-d366-43fc-b3d8-5bd3c038491d

This project is a fork of [mipnerf_pl](https://github.com/hjxwhy/mipnerf_pl) which served as a fantastic platform for iterating on NeRF ideas. This fork extends the logic of [Mip-NeRF](https://jonbarron.info/mipnerf/) contributions for depth supervision and novel techniques from [Mip-NeRF360](https://github.com/google-research/multinerf), which are ported from JAX to PyTorch with Lightning.

The theoretical contributions of the work are compactly derived here [Method](https://github.com/AlexSheldrick/mipnerf_pl/blob/depth-mipnerf/media/Method_SWB.pdf).

The following Mipnerf360 techniques have been ported to this fork:
- Proposal network with 4 layers and 256 hidden units, replacing the coarse & fine stages of sampling.
- Log linearly annealed learning rate (2e<sup>-3</sup> to 2e<sup>-5</sup>) with 512 warmup steps.
- Output pixels are conditioned on per camera embeddings to account for changes in per-image lighting conditions and motion blur (Global latent optimization).
- Network accepts real world RGB-D sensor readings (e.g. partially complete), or monocular depth prediction network outputs (e.g. Omnidata), or a composite.

A room sized scene takes about 30 minutes to train and consistently outperforms related works, namely the standard L2 depth-loss, and loss formulations from Urban Radiance Fields [URF](https://urban-radiance-fields.github.io/) and Depth-Supervised Nerf [DS-NeRF](https://github.com/dunbar12138/DSNeRF).


Lego truck from just three views:

https://github.com/AlexSheldrick/mipnerf_pl/assets/59337109/a35166c5-b8ea-4a76-a16a-0bfbe16c9bef

Quantitatively we found the following:

<table border="1">
    <thead>
        <tr>
            <th colspan="1" style="text-align: left;">Compared Works</th>
            <th colspan="4" style="text-align: right;">PSNR</th>
        </tr>
        <tr>
            <th></th>
            <td style="text-align: center;">3 Views</th>
            <td style="text-align: center;">6 Views</th>
            <td style="text-align: center;">100 Views</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-weight: bold;">Ours w/o Depth</td>
            <td style="text-align: center;">17.73</td>
            <td style="text-align: center;">23.87</td>
            <td style="text-align: center;">36.65</td>
        </tr>
        <tr>
            <td>Rendered L2 Depth</td>
            <td style="text-align: center;">20.32</td>
            <td style="text-align: center;">25.90</td>
            <td style="text-align: center;">35.09</td>
        </tr>
        <tr>
            <td>URF</td>
            <td style="text-align: center;">20.09</td>
            <td style="text-align: center;">26.12</td>
            <td style="text-align: center;">36.36</td>
        </tr>
        <tr>
            <td style="font-weight: bold;">Ours</td>
            <td style="text-align: center;"><strong>23.81</td>
            <td style="text-align: center;"><strong>27.91</td>
            <td style="text-align: center;"><strong>36.96</td>
        </tr>
    </tbody>
</table>

And lastly to showcase the potential for real world usage: my living room from 10 images, without Camera Parameters or Depth sensors (completely inferred from monocular depth map prediction networks, scaled and matched to keypoints extracted from COLMAP).


https://github.com/AlexSheldrick/mipnerf_pl/assets/59337109/fdf1684e-2dee-4fc5-98ce-997898c33b0a

And additional outputs (ScanNet Room738, 18 images, camera parameters inferred from COLMAP)

https://github.com/AlexSheldrick/mipnerf_pl/assets/59337109/6286f4b7-f218-4ae0-a5fd-a986a31367f1



FAQ for dataset and installation to be added soon.

--------------------------------------------------------------------------------------------------------------

# mipnerf_pl
Unofficial pytorch-lightning implement of [Mip-NeRF](https://jonbarron.info/mipnerf/), Here are some results generated by this repository (pre-trained models are provided below):

[![Multi-scale render result](https://res.cloudinary.com/marcomontalbano/image/upload/v1652148920/video_to_markdown/images/youtube--3MxfZVUOIps-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/3MxfZVUOIps "Multi-scale render result")

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-baqh" colspan="12">Multi Scale Train And Multi Scale Test</th>
    <th class="tg-0lax" colspan="2">Single Scale</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">PNSR</span></td>
    <td class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">SSIM</span></td>
    <td class="tg-0lax">PSNR</td>
    <td class="tg-0lax">SSIM</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-c3ow">Full Res</td>
    <td class="tg-c3ow">1/2 Res</td>
    <td class="tg-c3ow">1/4 Res</td>
    <td class="tg-c3ow">1/8 Res</td>
    <td class="tg-c3ow">Aveage <br>(PyTorch)</td>
    <td class="tg-c3ow">Aveage <br>(Jax)</td>
    <td class="tg-0pky">Full Res</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">1/2 Res</span></td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">1/4 Res</span></td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">1/8 Res</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">Average</span><br><span style="font-weight:400;font-style:normal">(PyTorch)</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">Average</span><br><span style="font-weight:400;font-style:normal">(Jax)</span></td>
    <td class="tg-baqh" colspan="2">Full Res</td>
  </tr>
  <tr>
    <td class="tg-0pky">lego</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">34.412</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">35.640</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">36.074</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">35.482</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">35.402</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">35.736</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.9719</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.9843</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.9897</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.9912</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.9843</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.9843</span></td>
    <td class="tg-0lax">35.198</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">0.985</span></td>
  </tr>
</tbody>
</table>

<img src="media/image_comp.png" width="600"/>

The top image of each column is groundtruth and the bottom image is Mip-NeRF render in different resolutions.

The above results are trained on the `lego` dataset with 300k steps for single-scale and multi-scale datasets respectively, and the pre-trained model can be found [here](https://drive.google.com/drive/folders/1QWhWkI37JDQRTcRjx6JfpUhjfKl_v8Rr?usp=sharing).
Feel free to contribute more datasets.

## Installation
We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. Run the following commands:
```
# Clone the repo
git clone https://github.com/hjxwhy/mipnerf_pl.git; cd mipnerf_pl
# Create a conda environment
conda create --name mipnerf python=3.9.12; conda activate mipnerf
# Prepare pip
conda install pip; pip install --upgrade pip
# Install PyTorch
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# Install requirements
pip install -r requirements.txt
```
## Dataset
Download the datasets from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and unzip `nerf_synthetic.zip`. You can generate the multi-scale dataset used in the paper with the following command:

```
# Generate all scenes
python datasets/convert_blender_data.py --blender_dir UZIP_DATA_DIR --out_dir OUT_DATA_DIR
# If you only want to generate a scene, you can:
python datasets/convert_blender_data.py --blender_dir UZIP_DATA_DIR --out_dir OUT_DATA_DIR --object_name lego
```
## Running
### Train
To train a single-scale `lego` Mip-NeRF:
```
# You can specify the GPU numbers and batch size at the end of command,
# such as num_gpus 2 train.batch_size 4096 val.batch_size 8192 and so on.
# More parameters can be found in the configs/lego.yaml file. 
python train.py --out_dir OUT_DIR --data_path UZIP_DATA_DIR --dataset_name blender exp_name EXP_NAME
```
To train a multi-scale `lego` Mip-NeRF:

```
python train.py --out_dir OUT_DIR --data_path OUT_DATA_DIR --dataset_name multi_blender exp_name EXP_NAME
```

### Evaluation

You can evaluate both single-scale and multi-scale models under the `eval.sh` guidance, changing all directories to your directory. Alternatively, you can use the following command for evaluation.

```
# eval single scale model
python eval.py --ckpt CKPT_PATH --out_dir OUT_DIR --scale 1 --save_image
# eval multi scale model
python eval.py --ckpt CKPT_PATH --out_dir OUT_DIR --scale 4 --save_image
# summarize the result again if you have saved the pnsr.txt and ssim.txt
python eval.py --ckpt CKPT_PATH --out_dir OUT_DIR --scale 4 --summa_only
```

### Render Spheric Path Video
It also provide a script for rendering spheric path video
```
# Render spheric video
python render_video.py --ckpt CKPT_PATH --out_dir OUT_DIR --scale 4
# generate video if you already have images
python render_video.py --gen_video_only --render_images_dir IMG_DIR_RENDER
```

### Visualize All Poses

The script modified from [nerfplusplus](https://github.com/Kai-46/nerfplusplus) supports visualize all poses which have been reorganized to right-down-forward coordinate. Multi-scale have different camera focal length which is equivalent to different resolutions.




<img src="media/single-scale.png" width="42%"/><img src="media/multi-scale.png" width="48%"/>


## Citation
Kudos to the authors for their amazing results:

```
@misc{barron2021mipnerf,
      title={Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields},
      author={Jonathan T. Barron and Ben Mildenhall and Matthew Tancik and Peter Hedman and Ricardo Martin-Brualla and Pratul P. Srinivasan},
      year={2021},
      eprint={2103.13415},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


# Acknowledgements
Thansks to [mipnerf](https://github.com/google/mipnerf),
[mipnerf-pytorch](https://github.com/AlphaPlusTT/mipnerf-pytorch),
[nerfplusplus](https://github.com/Kai-46/nerfplusplus),
[nerf_pl](https://github.com/kwea123/nerf_pl)