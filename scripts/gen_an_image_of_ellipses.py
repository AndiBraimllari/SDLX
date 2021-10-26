from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# create image with ellipses
image = ImageDraw.Draw(Image.new("RGB", (512, 512)))
image.ellipse([90, 190, 400, 400])
image.ellipse([72, 40, 400, 80])
image.ellipse([49, 115, 140, 150])
image.ellipse([90, 20, 300, 250])
image.ellipse([160, 30, 400, 170])

plt.imsave('some_name.png', np.array(image))
