from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import pandas as pd
import numpy as np
import random
import matplotlib
# Circumvent error when X11 forwarding is not available
if 'DISPLAY' not in os.environ: matplotlib.use('Pdf')
import matplotlib.pyplot as plt

import utils

outdir = '.'

def write_image_to_file(row):
    # FIXME: CORRECT SPELLING
    data = np.reshape(row['Adverserial Image'], [28,28])
    # data = np.reshape(row['Adversarial Image'], [28,28])
    true_label = row['True Label']
    adv_label = row['Predicted Label Adverserial']
    # adv_label = row['Predicted Label Adversarial']
    img = plt.imshow(data, cmap='Greys')
    plt.axis('off')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
    filename = '/{}-{}-{}.png'.format(
        str(int(true_label)),
        str(int(adv_label)),
        utils.random_string(8))
    plt.savefig(outdir + filename, bbox_inches='tight', pad_inches = 0)
    print("Saved " + outdir + filename)

def persist_adversarial_samples(df, n=1):
    utils.ensure_dir(outdir)
    sampled_df = df.sample(n=n)
    sampled_df.apply(write_image_to_file, axis=1)

def main(argv=None):
    global outdir
    if len(argv) < 2 or len(argv) > 3:
        print("usage: python {} <pickle_file_path> [num_samples_to_persist]".format(argv[0]))

    pkl_path = argv[1]
    if not os.path.isfile(pkl_path):
        print("error: {} is not a valid file path".format(pkl_path))

    if len(argv) == 3:
        try:
            n = int(argv[2])
        except:
            print("error: {} is not a valid input for [num_samples_to_persist]".format(argv[2]))
    else:
        n = 1

    print("Reading pickle file...")
    df = pd.read_pickle(pkl_path)
    num_rows = len(df.index)
    valid_adv_sample_rows = df[(df['Predicted Label'] != df['Predicted Label Adverserial']) \
        & (df['Predicted Label'] == df['True Label'])]
    num_valid_adv_sample_rows = len(valid_adv_sample_rows.index)

    print("{} out of {} of the samples generated were successful adversarial samples"
        .format(num_valid_adv_sample_rows, num_rows))

    outdir = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(pkl_path))), 'advsamples')

    print("Persisting a random selection of {} adversarial samples to".format(n), outdir)
    persist_adversarial_samples(valid_adv_sample_rows, n=n)

if __name__ == '__main__':
    main(sys.argv)
