import numpy as np
import cv2
import time
import torch
from ops import dct_op, dct2_op, comp_op
from math import cos, sqrt, pi


def DCT_compression_cpu(img_path, save_path, resize, mask):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (resize, resize))

    cv2.imwrite(save_path + "/resized_cpu_dctcp.png", image)

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

    start_time = time.time()  # record the beginning time
    result_list = []
    mask = np.array(mask)
    mask = mask.astype(np.float)
    dct_array = np.array(dct_array)
    dct_array = dct_array.astype(np.float)
    for channel in channels:
        image_np = np.array(channel)
        image_np = image_np.astype(np.float)

        height, width = image_np.shape[:2]
        block_y = height // 8
        block_x = width // 8
        height_ = block_y * 8
        width_ = block_x * 8
        img_f32_cut = image_np[:height_, :width_]
        img_dct = np.zeros((height_, width_), dtype=np.float32)
        new_img = np.zeros((height_, width_), dtype=np.float32)
        for h in range(block_y):
            for w in range(block_x):
                img_block = img_f32_cut[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]
                img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] = dct_array.dot(img_block).dot(dct_array.T)

                img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] = img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] * mask
                dct_block = img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]
                img_block_new = (dct_array.T).dot(dct_block).dot(dct_array)

                new_img[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] = img_block_new

        result_list.append(new_img)

    ans_img = cv2.merge(result_list)
    cv2.imwrite(save_path + "/DCT_compressed_img_CPU.png", ans_img)

    end_time = time.time()  # record the ending time
    print("Using DCT to compress a {H} * {W} image and spending {time:.3f} s by CPU".format(H=resize, W=resize, time=(end_time - start_time)))



def DCT_compression(img_path, save_path, resize, device, mask):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (resize, resize))

    cv2.imwrite(save_path + "/resized_gpu_dctcp.png", image)

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

        dct_tensor = torch.tensor(dct_array, dtype=torch.float32, device=device, requires_grad=False)
        dct_tensor_t = torch.tensor(dct_array.T, dtype=torch.float32, device=device, requires_grad=False)

        # DCT
        step1 = dct_op(dct_tensor, image_tensor)
        step2 = dct2_op(step1, dct_tensor_t)

        # Compressing
        step2 = comp_op(step2, mask)

        # IDCT
        step3 = dct_op(dct_tensor_t, step2)
        step4 = dct2_op(step3, dct_tensor)
        res = step4.cpu().numpy()
        result_list.append(res)

    ans_img = cv2.merge(result_list)
    cv2.imwrite(save_path + "/DCT_compressed_img_GPU.png", ans_img)

    torch.cuda.synchronize()
    end_time = time.time()  # record the ending time

    print("Using DCT to compress a {H} * {W} image and spending {time:.3f} s by GPU".format(H=resize, W=resize, time=(end_time - start_time)))
