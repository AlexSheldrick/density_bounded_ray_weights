import numpy as np
from PIL import Image
from glob import glob
from  pathlib import Path

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    name = path.split('/')[-1]
    return image, name

def subselect_image(image, mode='pred', crop=[[20,380],[40,300]]): #[[left, right], [top, bottom]]
    #crop TL, TR, BL, BR
    shapes = image.shape
    h = shapes[0] // 2
    w = shapes[1] // 3
    #crop = [[0,0],[0,0]]
    gt_rgb = image[crop[1][0]:crop[1][1], crop[0][0]:crop[0][1], :]
    gt_depth = image[crop[1][0]:crop[1][1], w+crop[0][0]:w+crop[0][1], :]
    gt_normal = image[crop[1][0]:crop[1][1], 2*w+crop[0][0]:crop[0][1]+2*w, :]
    pred_rgb = image[h+crop[1][0]:h+crop[1][1], crop[0][0]:crop[0][1], :]    
    pred_depth = image[h+crop[1][0]:h+crop[1][1], w+crop[0][0]:w+crop[0][1], :]
    pred_normal = image[h+crop[1][0]:h+crop[1][1], 2*w+crop[0][0]:crop[0][1]+2*w, :]
    if mode == 'pred': return pred_rgb, pred_depth, pred_normal
    else: return gt_rgb, gt_depth, gt_normal

def np_to_image(nparr):
    return Image.fromarray(nparr.astype(np.uint8)) #.transpose(1,2,0)*255
    #out_path = Path(out_path, 'images')
    #out_path.mkdir(exist_ok=True)
    #out_path = Path(out_path)
    #out_path.mkdir(exist_ok=True)
    #im.save(str(out_path) + f"/{mode}_{self.global_step}.jpeg")

def save_subselected_image(path, mode='pred'):
    image, name = load_image(path)
    rgb, d, n = subselect_image(image, mode)
    experiment = path.split('/')[-2]
    images = [rgb, d, n]
    for i, suffix in enumerate(['rgb', 'depth', 'normal']):
        outdir = '/home/alex/Desktop/msc material' + f'/split_images/{experiment}/'
        outname = name.replace('.jpeg', f'_{mode}_{suffix}.jpeg')
        Path(outdir).mkdir(exist_ok=True)
        np_to_image(images[i]).save(str(outdir + outname))

if __name__ == '__main__':
    in_dir = '/media/alex/SSD Datastorage/mipnerf_pl/OUT/new/images/'
    experiment = 'lego+URF-6'
    #i = '500'
    split = 'val'
    for i in [500, 1000, 2000, 4000, 8000, 16000]:
        save_subselected_image(f'{in_dir}/{experiment}/{split}_{str(i)}.jpeg', 'pred')