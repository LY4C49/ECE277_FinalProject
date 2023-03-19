import time
import torch
from py_interface import Sharpening, DCT_compression


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # You can replace with other kernel, the central value is bigger, the result will be more obvious
    kernel = [[0, -1, 0],
              [-1, 8, -1],
              [0, -1, 0]]

    # You can modify the mask to increase/decrease compression rate
    mask = [[1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]

    Sharpening.sharpening("img/img1.png", "img/img1", 4096, device, kernel)

    DCT_compression.DCT_compression("img/img1.png", "img/img1", 4096, device, mask)

    DCT_compression.DCT_compression_cpu("img/img1.png", "img/img1", 4096, mask)

    #Sharpening.sharpening_cpu("img/img1.png", "img/img1", 4096, kernel)