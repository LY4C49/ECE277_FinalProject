import numpy as np
import cv2
import time
import torch
from ops import dct_op, dct2_op, comp_op
from math import cos, sqrt, pi

def DCT_compression(img_path, save_path, resize, device, mask):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (resize, resize))

    cv2.imwrite(save_path + "/resized.png", image)

    # You have to input an RGB picture!
    channels = cv2.split(image)

    # Calculate the kernel of a DCT transformation
    dct_array = np.ones((8, 8))
    for p in range(8):
        for q in range(8):
            if p == 0:
                dct_array[p][q] = 1 / sqrt(8)
            else:
                dct_array[p][q] = sqrt(2 / 8) * cos((pi * (2 * q + 1) * p) / (2 * 8))

    torch.cuda.synchronize()
    start_time = time.time()  # record the beginning time

    result_list = []
    for channel in channels:
        image_np = np.array(channel)
        image_tensor = torch.from_numpy(image_np)

        # CPU to GPU
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32, device=device, requires_grad=False)
        mask = torch.tensor(mask, dtype=torch.float32, device=device, requires_grad=False)

        T = torch.tensor(dct_array, dtype=torch.float32, device=device, requires_grad=False)
        T_t = torch.tensor(dct_array.T, dtype=torch.float32, device=device, requires_grad=False)

        # DCT
        step1 = dct_op(T, image_tensor)
        step2 = dct2_op(step1, T_t)

        # Compressing
        step2 = comp_op(step2, mask)

        # IDCT
        step3 = dct_op(T_t, step2)
        step4 = dct2_op(step3, T)
        res = step4.cpu().numpy()
        result_list.append(res)

    ans_img = cv2.merge(result_list)
    cv2.imwrite(save_path + "/DCT_compressed_img.png", ans_img)

    torch.cuda.synchronize()
    end_time = time.time()  # record the ending time

    print("Using DCT to compress a {H} * {W} image and spending {time:.3f} s".format(H=resize, W=resize, time=(end_time - start_time)))
