import os
import shutil
from tqdm import tqdm
import numpy as np
import re
import os

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

from dataset import MultiChannelDataset
from eval import eval_net

import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    name = np.random.randint(2**32)
    logger = logging.getLogger(str(name))
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def run(model, root_dir, save_dir, batch_size, learning_rate, epochs, percentages, device):
    model = model(input_ch=1, output_ch=1)

    # Generate Necessary Files
    if os.path.isdir(save_dir):
        res = input(f"Directory already exists: {save_dir}\n1 - Use it anyway.\n2 - Delete.\n3 - Exit.\n>> ")
        if res == '2':
            shutil.rmtree(save_dir)
        if res == '3':
            exit(0)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Load Existing Models
    cks = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(save_dir)
           if re.match(r'(.*)\.(pth)', f)]
    cks.sort()
 
    if len(cks) > 0:
        epoch = cks[-1]
        latest = "model_save_epoch_{:02}.pth".format(epoch)

        print("Loading model Epoch:", epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, latest)))
    else:
        epoch = 0

    # Load Datasets
    dataset = MultiChannelDataset(root_dir)
    n_samples = [int((len(dataset) * p) + 0.5) for p in percentages] 
    samples = random_split(dataset, n_samples)

    train_loader = DataLoader(samples[0], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(samples[1], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Build the Network
    model = model.to(device)

    global_step = epoch * n_samples[0]
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=1)
    criterion = torch.nn.MSELoss()

    # Logging Settings
    logger = setup_logger(os.path.join(save_dir, "model_run.log"))
    writer = SummaryWriter(log_dir=save_dir, purge_step=global_step)

    info = f'''
    Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {learning_rate}
    Dataset Setting: {percentages}
    Training size:   {n_samples[0]}
    Validation size: {n_samples[1]}
    Test size:       {n_samples[2]}
    Save Directory:  {save_dir}
    Model Directory: {root_dir}
    Device:          {device.type}
    '''
    logger.info(info)
    print(info)

    # Run the Model
    for epoch in range(epoch, epochs):
        model.train()

        with tqdm(total=n_samples[0], desc=f'Epoch {epoch + 1}/{epochs}', unit='vec') as pbar:
            for batch in train_loader:
                inputs = batch["input"]
                output = batch["output"]

                inputs = inputs.to(device=device, dtype=torch.float32)
                output = output.to(device=device, dtype=torch.float32)

                output_pred = model(inputs)
                loss = criterion(output_pred, output)

                writer.add_scalar('Training Loss', loss.item(), global_step)
                pbar.set_postfix(**{'batch_loss': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(inputs.shape[0])
                global_step += 1

            val_loss = eval_net(model, val_loader, device, n_samples[1], writer, global_step)
            logger.info('Validation Loss: {}'.format(val_loss))
            writer.add_scalar('Validation Loss', val_loss, global_step)
            scheduler.step(val_loss)

            audio = output.cpu().detach().numpy()
            print("TMax:", np.max(audio), "TMin:", np.min(audio), audio.shape)
            writer.add_audio('training/truth', output[0], global_step, 32e3)

            audio = output_pred.cpu().detach().numpy()
            print("PMax:", np.max(audio), "PMin:", np.min(audio), audio.shape)
            writer.add_audio('training/pred', output_pred[0], global_step, 32e3)

            if (epoch % 5) == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, "model_save_epoch_{}.pth".format(epoch)))
                logger.info('Checkpoint {} saved!'.format(epoch))

    writer.close()
    logger.info('Training finished, exiting...')
    torch.save(model.state_dict(), os.path.join(save_dir, "model_save_epoch_{}.pth".format(epoch)))
    logger.info('Final checkpoint {} saved!'.format(epoch))
    del model