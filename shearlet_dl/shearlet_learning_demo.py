import torch
from matplotlib import pyplot as plt
from skimage import color
from skimage import io

from shearlet_dl.PhantomNet import PhantomNet
from shearlet_transform.shearlet_transform_algorithm import applyShearletTransform as SH


def shearlet_learning_demo():
    pass


# TODO ideal plan
#  1.apply a limited-angle trajectory generator to the data
#  2.apply the shearlet transform to these altered data
#  3.apply the custom ADMM to promote edges (L1 reg. promotes sparsity)
#  4.train the PhantomNet model with this as X and the initial data as Y

# TODO use only square images for now
# TODO not entirely sure how important applying ADMM is here
# TODO note that proper training appears to require A LOT of resources


# altered_data = LimitedAngleTrajectoryGenerator(data)

# shearleted_data = SH(data)

# sparsified_data = SHADMM(...)

# num_of_subbands = ...

# model = PhantomNet(num_of_subbands=num_of_subbands)

# model.train()

# torch.save(model.state_dict(), './phantom_net_model.pth')


shearlet_learning_demo()
