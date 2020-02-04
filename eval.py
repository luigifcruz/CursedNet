from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import utils


def eval_net(net, loader, device, n_val, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='vec', leave=False) as pbar:
        for batch in loader:
            inputs = batch["input"]
            output = batch["output"]

            inputs = inputs.to(device=device, dtype=torch.float32)
            output = output.to(device=device, dtype=torch.float32)

            output_pred = net(inputs)
            
            for out, pred in zip(output, output_pred):
                tot += F.mse_loss(pred, out).item()

            pbar.update(inputs.shape[0])

    return tot / n_val