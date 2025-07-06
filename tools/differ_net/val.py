import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
import torch
from torch.autograd import Variable

from data import MVTEC_MEAN_STD

GRADIENT_MAP_DIR = 'gradient_maps'

def loss_function(z, jacobian):
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jacobian) / z.shape[1]

def save_imgs(inputs, grad, cnt, category):
    export_dir = os.path.join(GRADIENT_MAP_DIR, category)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for g in range(grad.shape[0]):
        normed_grad = (grad[g] - np.min(grad[g])) / (
                np.max(grad[g]) - np.min(grad[g]))
        orig_image = inputs[g]
        for image, file_suffix in [(normed_grad, '_gradient_map.png'), (orig_image, '_orig.png')]:
            plt.clf()
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(os.path.join(export_dir, str(cnt) + file_suffix), bbox_inches='tight', pad_inches=0)
        cnt += 1
    return cnt

def export_gradient_maps(model, loader, optimizer, n_batches=1, num_transform=16):
    plt.figure(figsize=(10, 10))
    loader.dataset.fixed_rotation_mode = True
    category = loader.dataset.category
    mean = MVTEC_MEAN_STD[category][0]
    std = MVTEC_MEAN_STD[category][1]
    
    cnt = 0
    degrees = -1 * np.arange(num_transform) * 360.0 / num_transform
    progress_bar = tqdm(total=n_batches)
    
    for i, data in enumerate(progress_bar):
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to('cuda')
        inputs = inputs.view(-1, *inputs.shape[-3:])
        
        labels = labels.to('cuda')
        inputs = Variable(inputs, requires_grad=True)

        emb, jacobian = model(inputs)
        loss = loss_function(emb, jacobian)
        loss.backward()

        grad = inputs.grad.view(-1, num_transform, *inputs.shape[-3:])
        grad = grad[labels > 0]
        if grad.shape[0] == 0:
            continue
        grad = grad.detach().cpu().numpy()

        inputs = inputs.view(-1, num_transform, *inputs.shape[-3:])[:, 0]
        inputs = np.transpose(inputs[labels > 0].detach().cpu().numpy(), [0, 2, 3, 1])
        inputs_unnormed = np.clip(inputs * std + mean, 0, 1)

        for i_item in range(num_transform):
            old_shape = grad[:, i_item].shape
            img = np.reshape(grad[:, i_item], [-1, *grad.shape[-2:]])
            img = np.transpose(img, [1, 2, 0])
            img = np.transpose(rotate(img, degrees[i_item], reshape=False), [2, 0, 1])
            img = gaussian_filter(img, (0, 3, 3))
            grad[:, i_item] = np.reshape(img, old_shape)

        grad = np.reshape(grad, [grad.shape[0], -1, *grad.shape[-2:]])
        grad_img = np.mean(np.abs(grad), axis=1)
        grad_img_sq = grad_img ** 2

        cnt = save_imgs(inputs_unnormed, grad_img_sq, cnt)

        if i == n_batches:
            break

    plt.close()
    loader.dataset.get_fixed = False
    