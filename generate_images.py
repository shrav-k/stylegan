# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config


file_path = '/home/andeeptoor_google_com/src/stylegan/results/00012-sgan-all-creatures-v3-1024-8gpu/network-snapshot-007030.pkl'
output_dir = '/home/andeeptoor_google_com/src/stylegan/output'
num_images = 120
def main():
    # Initialize TensorFlow.
    tflib.init_tf()
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained network.
    with open(file_path, "rb") as f:
    	_G, _D, Gs = pickle.load(f)
    # Print network details.
    Gs.print_layers()
    rnd = np.random.RandomState(5)
    for i in range(num_images):
	    # Pick latent vector.
	    print(i)
	    latents = rnd.randn(1, Gs.input_shape[1])

	    # Generate image.
	    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
	    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

	    # Save image.
	    png_filename = os.path.join(output_dir, '%d.png' % (i))
	    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
