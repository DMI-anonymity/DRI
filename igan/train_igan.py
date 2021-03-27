from .igan_trainer import IganTrainer
from .discriminator import Discriminator
from .generator import Generator
import torch
import os
import torch.optim as optim
import numpy as np


def training_gan(train_dataloader, task_id, epochs, g=None, d=None):
    final_generator_path = 'igan/results/generator_{}.pt'.format(task_id)
    save_checkpoint_path = 'igan/checkpoints/checkpoint_{}.pt'.format(task_id)
    load_checkpoint_path = None

    dropout_rate = 0.5
    max_pixel_value = 1.0
    should_display_generations = True
    batch_size = 32

    # Input sanity checks
    final_generator_dir = os.path.dirname(final_generator_path) or os.getcwd()
    if not os.access(final_generator_dir, os.W_OK):
        raise ValueError(final_generator_path + " is not a valid filepath.")

    if type(train_dataloader.dataset.x1_examples[0]) == torch.Tensor:
        in_channels = 1
        img_size = 28
    else:
        in_channels = np.array(train_dataloader.dataset.x1_examples).shape[-1]
        img_size = np.array(train_dataloader.dataset.x1_examples).shape[2]

    if g is None:
        g = Generator(dim=img_size, channels=in_channels, dropout_rate=dropout_rate)
        d = Discriminator(dim=img_size, channels=in_channels * 2, dropout_rate=dropout_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g_opt = optim.Adam(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
    d_opt = optim.Adam(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

    flat_val_data = train_dataloader.dataset.x1_examples
    flat_val_data = None
    display_transform = train_dataloader.dataset.transform

    trainer = IganTrainer(
        generator=g,
        discriminator=d,
        gen_optimizer=g_opt,
        dis_optimizer=d_opt,
        batch_size=batch_size,
        device=device,
        critic_iterations=5,
        print_every=500,
        num_tracking_images=10,
        save_checkpoint_path=save_checkpoint_path,
        load_checkpoint_path=load_checkpoint_path,
        display_transform=display_transform,
        should_display_generations=should_display_generations,
    )
    trainer.train(data_loader=train_dataloader, epochs=epochs, val_images=flat_val_data)

    # Save final generator model
    torch.save(trainer.g, final_generator_path)

    return trainer.g, trainer.d

