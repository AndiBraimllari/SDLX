from pyellipsoid import drawing
import numpy as np
import scipy.ndimage as nd
import argparse
import random
import os

"""
Script for creating a dataset of ellipses. Initial code provided by Robert Graf.
"""


class EllipsesGenerator:
    def __init__(self, volume_shape=(640, 640, 2), ell_shape=(200, 200, 2)):
        self.ell_shape = ell_shape
        self.volume_shape = volume_shape

    def get_canvas(self):
        return np.zeros(self.volume_shape, dtype=np.float32)

    def get_ellipsoid(self, ell_shape, poss_rand):
        # define an image shape, axis order is: Z, Y, X
        image_shape = np.asarray(ell_shape)

        # define an ellipsoid, axis order is: X, Y, Z
        min_radii = image_shape[1] * 0.1
        max_radii = image_shape[1] / 2

        ell_radii = min_radii + np.random.rand(3) * (max_radii - min_radii)
        if poss_rand:
            ell_center = (
                image_shape[2] * np.random.rand() * 2, image_shape[1] * np.random.rand(),
                image_shape[0] * np.random.rand())
        else:
            ell_center = (np.random.rand() * image_shape[2], image_shape[1] / 2, image_shape[0] / 2)

        # order of rotations is X, Y, Z
        ell_angles = np.deg2rad(np.random.randint(180, size=3))

        # draw a 3D binary image containing the ellipsoid
        image = drawing.make_ellipsoid_image(image_shape, ell_center, ell_radii, ell_angles).astype(np.float32)

        # single value
        ell = 0.1 + 0.9 * np.random.rand()
        ell2 = np.random.rand()
        ell3 = 0.1 + 0.9 * np.random.rand()
        return ell * image, ell2 * image, ell3 * image

    def get_img_base(self):
        canvas = (self.get_canvas(), self.get_canvas(), self.get_canvas())
        temp_canvas = self.get_canvas()

        def overlay(canvas_param, shape, temp_canvas_param, poss_rand=False):
            objList = self.get_ellipsoid(shape, poss_rand)
            max_offsetX = self.volume_shape[0] - shape[0]
            max_offsetY = self.volume_shape[1] - shape[1]
            max_offsetZ = 0
            offsetX = random.randint(0, max_offsetX)
            offsetY = random.randint(0, max_offsetY)
            offsetZ = random.randint(0, max_offsetZ)

            for i in range(len(objList)):
                temp_canvas_param *= 0

                temp_canvas_param[offsetX:shape[0] + offsetX, offsetY:shape[1] + offsetY] = objList[i][...,
                                                                                            offsetZ:shape[2] + offsetZ]

                mask = temp_canvas_param > 0
                canvas_param[i][mask] = temp_canvas_param[mask]
            return canvas_param

        # big
        a = np.random.randint(5)
        for i in range(a):
            print('big ellipsoids', i + 1, '/', a)
            canvas = overlay(canvas, self.volume_shape, temp_canvas, poss_rand=True)
        # medium
        a = np.random.randint(9)
        for i in range(a):
            print('medium-big ellipsoids', i + 1, '/', a)
            canvas = overlay(canvas, self.ell_shape, temp_canvas)
        # medium
        a = np.random.randint(12)
        for i in range(a):
            print('medium ellipsoids', i + 1, '/', a)
            canvas = overlay(canvas, (100, 100, 2), temp_canvas)
        # small
        a = np.random.randint(4)
        for i in range(a):
            print('small ellipsoids', i + 1, '/', a)
            canvas = overlay(canvas, (32, 32, 2), temp_canvas)
        a = np.random.randint(4)
        for i in range(a):
            print('very small ellipsoids', i + 1, '/', a)
            canvas = overlay(canvas, (16, 16, 2), temp_canvas)
        return canvas

    def get_phantoms(self, img_base):
        img_att = img_base[0] * 0.0005
        img_dfi = img_base[1] * 0.0005
        img_dfi = img_dfi * 0.34

        img_dpc = img_base[2] * 0.25
        sigma = 0.5
        img_att = nd.gaussian_filter(img_att, sigma)
        img_dfi = nd.gaussian_filter(img_dfi, sigma)
        img_dpc = nd.gaussian_filter(img_dpc, sigma)
        return img_att, img_dfi, img_dpc

    def generate(self, phantom_id, destination):
        print('generating phantom: ' + str(phantom_id))
        img_base = self.get_img_base()
        phantoms = self.get_phantoms(img_base)
        file_name = destination + '/ellipse_' + str(phantom_id) + '.npy'
        np.save(file_name, phantoms[0][:, :, 0])

    def generateAndSplit(self, num_phantoms, destination='.'):
        for phantom_id in range(num_phantoms):
            self.generate(phantom_id, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dst", type=str, default='.', help="path to destination dataset")
    parser.add_argument("-pht", "--phantoms", type=int, default=200, help="number of phantoms to be generated")

    args = parser.parse_args()

    dest = args.dst + '/ellipses_dataset'
    if not os.path.exists(dest):
        print('making dir: ', dest)
        os.makedirs(dest)

    gen = EllipsesGenerator()
    gen.generateAndSplit(args.phantoms, destination=dest)
