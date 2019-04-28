"""
Compute the Mean Absolute Percentage Error between two images:
https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
"""

import sys
from scipy.misc import imread
import numpy as np

def mape_score(ground_truth, prediction):

    # Get image as a 3D array: Width, Height, RGB
    gt_arr = np.asarray(imread(ground_truth, mode='RGB'), dtype=np.intc)
    p_arr = np.asarray(imread(prediction, mode='RGB'), dtype=np.intc)

    # Compute the score
    score = np.sum(np.abs(gt_arr/255 - p_arr/255)/((gt_arr+0.01)/255))
    score /= len(gt_arr) * len(gt_arr[0]) * len(gt_arr[0][0])

    # Round to 4 decimal places
    return round(score,4)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Two file paths to images must be given. Terminating.")
    else:
        ground_truth = sys.argv[1]
        prediction = sys.argv[2]
        score = mape_score(ground_truth, prediction)

        print("Score: " + str(score))
        print("MAPE: " + str(score*100) + "%")

