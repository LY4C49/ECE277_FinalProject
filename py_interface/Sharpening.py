import numpy as np
import cv2
import time
import torch
import torch.nn.functional as F
from ops import sharpening_add_op, sharpening_op


def sharpening_cpu(img_path, save_path, resize, kernel):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (resize, resize))

    cv2.imwrite(save_path + "/sharpening_resized_CPU.png", image)

    # You have to input an RGB picture!
    channels = cv2.split(image)
    start_time = time.time()  # record the beginning time
    result_list = []
    for channel in channels:
        image_np = np.array(channel)
        res = np.ones((resize-2, resize-2))

        for r in range(0, resize - 2):
            for c in range(0, resize - 2):
                cur_input = image_np[r:r + 3, c:c + 3]
                cur_output = cur_input * kernel
                conv_sum = np.sum(cur_output)
                res[r, c] = conv_sum
        res = np.pad(res, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        res += channel
        result_list.append(res)

    ans_img = cv2.merge(result_list)
    cv2.imwrite(save_path + "/sharpening_img_cpu.png", ans_img)
    end_time = time.time()  # record the ending time
    print("Sharpening a {H} * {W} image and spending {time:.3f} s by CPU".format(H=resize, W=resize, time=(end_time - start_time)))


def sharpening(img_path, save_path, resize, device, kernel):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (resize, resize))

    cv2.imwrite(save_path + "/sharpening_resized_GPU.png", image)

    # You have to input an RGB picture!
    channels = cv2.split(image)

    torch.cuda.synchronize()
    start_time = time.time()  # record the beginning time

    result_list = []
    for channel in channels:
        image_np = np.array(channel)
        image_tensor = torch.from_numpy(image_np)

        # CPU to GPU
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32, device=device, requires_grad=False)
        kernel = torch.tensor(kernel, dtype=torch.float32, device=device, requires_grad=False)

        # GPU kernel function
        sharpening_result = sharpening_op(image_tensor, kernel)
        sharpening_result = F.pad(input=sharpening_result, pad=(1, 1, 1, 1), mode='constant', value=0)

        # GPU kernel function
        res = sharpening_add_op(image_tensor, sharpening_result)
        res = res.cpu().numpy()  # GPU to CPU
        result_list.append(res)

    ans_img = cv2.merge(result_list)
    cv2.imwrite(save_path + "/sharpening_img_GPU.png", ans_img)

    torch.cuda.synchronize()
    end_time = time.time()  # record the ending time

    print("Sharpening a {H} * {W} image and spending {time:.3f} s by GPU".format(H = resize, W = resize, time = (end_time - start_time)))