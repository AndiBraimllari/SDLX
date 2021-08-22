import os
import shutil
from tqdm import tqdm


def copy_sparse_ct_images(src_dir, out_dir, limit=None):
    """
    Copy files within the src_dir directory to out_dir `shutil.copy2`.
    """
    if not os.path.exists(src_dir):
        raise ValueError('The provided src_dir directory does not exist')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    paths = os.listdir(src_dir)

    if limit is not None:
        paths = paths[:limit]

    for file_name in tqdm(paths):
        shutil.copy2(src_dir + '/' + file_name, out_dir)
