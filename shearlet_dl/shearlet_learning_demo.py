from matplotlib import pyplot as plt

from shearlet_dl.PhantomNet import PhantomNet
from shearlet_transform.shearlet_transform_algorithm import applyShearletTransform, applyInverseShearletTransform


def generate_limited_angle_ct_data(dir):
    for i in range(len(data)):
        # fetch the next sample
        ct_image = read_ct_image(i)

        # generate limited angle trajectory
        auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(...)

        # create a projector based on it
        SiddonsMethod projector(ct_image.getDataDescriptor(), *sinoDescriptor)

        # simulate its sinogram
        auto sinogram = projector.apply(ct_image)

        # setup reconstruction problem
        WLSProblem wlsProblem(projector, sinogram)

        # solve the reconstruction problem
        # TODO consider a faster solver
        CG cgSolver(wlsProblem)
        # TODO how many iterations to use?
        auto cgReconstruction = cgSolver.solve(100)

        # write the limited angle reconstruction out
        EDF::write(cgReconstruction, dir + ct_image.getName() + "_cg_recon_" + i + ".edf")


def generate_sparse_ct_data(dir):
    """
    Similar to `generate_limited_angle_ct_data` but with CircleTrajectoryGenerator instead of
    LimitedAngleTrajectoryGenerator
    """
    pass


def solve_limited_angle_and_sparse_ct():
    """
    Elaborating here a plan for tackling limited angle/sparse CT
    """

    # read the original training data
    origs_dir = 'orig'
    data = read(origs_dir)

    # read the original test data
    tests_dir = 'tests'
    test_data = read(tests_dir)

    # generate limited angle/sparse ct scans based on existing ones
    recons_dir = 'recons'
    generate_limited_angle_ct_data(recons_dir)
    # or generate_sparse_ct_data(...), or even both and mix them up

    # get the limited angle/sparse images
    recons_data = read(recons_dir)

    # apply our custom ADMM to promote edges in the data
    # TODO we don't have this for now, but I'd still like to try this without it, we should still get interpretable
    #  results
    admm_recons_data = SHADMM(recons_data)

    # apply the shearlet transform to the output of ADMM
    shearleted_data = [applyShearletTransform(datum) for datum in admm_recons_data]

    # train the PhantomNet
    model = PhantomNet(...)
    model.train(shearleted_data, data)
    learned_coefficients = model.predict(test_data)

    # last step of the LTI paper, combine visible and invisible coefficients
    solved_data = applyInverseShearletTransform(learned_coefficients + shearleted_data)

    # compare a sample with the output of the model
    _plot_sample(test_data[0], 'sample data')
    _plot_sample(solved_data[0], 'model recon')


def _plot_sample(img, title):
    """
    Helper function for plotting gray images with a title
    """
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()


"""
Additional notes
    1.use only square images for now (the LTI paper only considers square images), perhaps later we can use non-squared
        images
    2.not entirely sure how important applying ADMM is here, I assume it should aid the DL model to learn better, but
        still worth trying without it
    3.note that proper training appears to require A LOT of resources, overfit on a handful of samples locally, then
        move to the CIIP machines
"""
