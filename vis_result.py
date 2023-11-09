import os.path as osp
from PIL import Image
from ibasis import ibasisF as fn
from ibasis import inusc
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm


# merge 6 camera images -> 2x3


def vis_single(img_dir, save_dir):
    out_path = osp.join(save_dir, f"{osp.split(img_dir)[-1]}.jpg")
    if osp.exists(out_path):
        return
    paths = fn.get_paths(img_dir, file_type='.jpg', is_lis=True, is_sort=True)
    img_lis = list()
    for path in paths:
        img = Image.open(path)
        img_lis.append(img)

    img_lis = [
        img_lis[2],
        img_lis[0],
        img_lis[1],
        img_lis[4],
        img_lis[3],
        img_lis[5],
    ]

    inusc.show_img_lis(img_lis, (28, 11))
    out_path = osp.join(save_dir, f"{osp.split(img_dir)[-1]}.jpg")
    fn.make_path_dir(out_path)
    plt.savefig(out_path)


if __name__ == '__main__':
    f = partial(vis_single, save_dir='visual_dir_24_out')
    dirs = fn.get_dirs('visual_dir_24')
    for dir_ in tqdm(dirs):
        f(dir_)
