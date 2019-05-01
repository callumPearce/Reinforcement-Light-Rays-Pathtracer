"""
Compute the squared difference between pixel values of two image
and output the image
"""

import sys
from scipy.misc import imread
from PIL import Image
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

def compute_sqd_image(ground_truth, prediction):

    # Get image as a 3D array: Width, Height, RGB
    gt_arr = np.asarray(imread(ground_truth, mode='RGB'), dtype=np.intc)
    p_arr = np.asarray(imread(prediction, mode='RGB'), dtype=np.intc)

    # Setup new image arrow
    diff_arr = np.zeros((len(gt_arr), len(gt_arr[0]), len(gt_arr[0][0])), dtype=np.intc)
    for x in range(len(gt_arr)):
        for y in range(len(p_arr)):
            err = (np.sum(np.absolute(gt_arr[x][y] - p_arr[x][y]))/3)**2
            diff_arr[x][y][0] = err
            diff_arr[x][y][1] = err
            diff_arr[x][y][2] = err

    # diff_arr = np.square(np.absolute(gt_arr - p_arr))
    diff_arr = diff_arr.astype(np.uint8)

    # Img name
    score = mape_score(ground_truth, prediction)
    name = prediction[:-4] + "_mape_" + str(score) + ".png"

    # Compute and save image
    img = Image.fromarray(diff_arr, 'RGB')
    img.save(name)
    img.show()


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Two file paths to images must be given. Terminating.")
    else:
        ground_truth = sys.argv[1]
        prediction = sys.argv[2]
        compute_sqd_image(ground_truth, prediction)