import numpy as np
import cv2
import time
import torch
import torch.nn.functional as F
from ops import sharpening_add_op, sharpening_op


def sharpening(img_path, save_path, resize, device, kernel):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (resize, resize))

    cv2.imwrite(save_path + "/resized.png", image)

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

        sharpening_result = sharpening_op(image_tensor, kernel)
        sharpening_result = F.pad(input=sharpening_result, pad=(1, 1, 1, 1), mode='constant', value=0)

        res = sharpening_add_op(image_tensor, sharpening_result)
        res = res.cpu().numpy()  # GPU to CPU
        result_list.append(res)

    ans_img = cv2.merge(result_list)
    cv2.imwrite(save_path + "/sharpening_img.png", ans_img)

    torch.cuda.synchronize()
    end_time = time.time()  # record the ending time

    print("Sharpening a {H} * {W} image and spending {time:.3f} s".format(H = resize, W = resize, time = (end_time - start_time)))