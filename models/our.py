from copy import deepcopy
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
import torch.nn as nn
from torch.optim import SGD
from pytorch_metric_learning import losses as torch_losses
from models.utils.distance import CosineSimilarity
from igan.train_igan import training_gan
from PIL import Image
import PIL
from datasets.utils.continual_dataset import get_buffer_loaders_gan
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from igan.generator import Generator
from torchsummary import summary
import os


class Our(ContinualModel):
    NAME = 'our'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Our, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS).to(self.device)
        self.gan_generator = None
        self.d = None
        self.class_means = None
        self.old_net = None
        self.current_task = 0
        self.display_img = False
        self.task_net = None
        self.num_augment = 1
        self.task_class_means = [None] * self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        if self.dataset.SETTING == 'domain-il':
            self.task_class_means = [[None for i in range(self.dataset.N_TASKS)] for j in range(self.dataset.N_CLASSES_PER_TASK)]
        self.criterion = torch_losses.MultiSimilarityLoss(alpha=2, beta=10, base=0.5)
        self.loss_name = self.criterion._get_name()

    def forward(self, x, dataset=None):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means(dataset)

        feats = self.net(x)
        feats = feats.unsqueeze(1)
        pred = -(self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return pred

    def old_means_pre(self, x):
        feats = self.net(x)
        feats = feats.unsqueeze(1)
        if self.dataset.SETTING == 'domain-il':
            task_class_means = torch.tensor([np.array(torch.stack(self.task_class_means[i])).mean(0) for i in range(len(self.task_class_means))])
            task_class_means = task_class_means.to(self.device)
        else:
            task_class_means = self.task_class_means[:self.dataset.N_CLASSES_PER_TASK * self.current_task]
            task_class_means = torch.stack(task_class_means).to(self.device)
        task_class_means = (task_class_means + self.class_means) / 2.0
        pred = -(task_class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return pred

    def observe(self, inputs, labels, not_aug_inputs, old_model_feats=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.opt.zero_grad()

        loss = self.get_loss(inputs, labels, self.current_task)
        loss.backward()

        self.opt.step()

        return loss.item()

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """

        criterion = self.criterion
        l2_loss = nn.MSELoss()

        if not self.buffer.is_empty():

            buf_inputs, buf_labels, _, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.dataset.get_transform())
            buf_inputs, buf_labels = self.gan_augment(self.num_augment, task_idx, buf_inputs, buf_labels)

            with torch.no_grad():
                old_model_old_task_feats = self.old_net.features(buf_inputs)
            loss = self.args.beta * l2_loss(self.net.features(buf_inputs), old_model_old_task_feats)
            loss += self.args.alpha * criterion(self.net(buf_inputs), buf_labels)

            fra_float = self.dataset.args.batch_size / (task_idx + 1)
            idx_buf = torch.randint(len(inputs), (int(fra_float * task_idx * 1.2),))
            buf_inputs = buf_inputs[idx_buf]
            buf_labels = buf_labels[idx_buf]
            idx = torch.randint(len(inputs), (int(fra_float * 1),))
            inputs = inputs[idx]
            labels = labels[idx]

            inputs = torch.cat((inputs, buf_inputs), 0)
            labels = torch.cat((labels, buf_labels), 0)
            loss += criterion(self.net(inputs), labels)

        else:
            feats = self.net(inputs)
            loss = criterion(feats, labels)

        assert loss >= 0
        if self.args.wd_reg:
            loss.data += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)

        return loss

    def end_task(self, dataset, task_id) -> None:
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        with torch.no_grad():
            if self.dataset.SETTING == 'domain-il':
                self.fill_domain_buffer(self.buffer, dataset, self.current_task)
            else:
                self.fill_buffer(self.buffer, dataset, self.current_task)
        self.current_task += 1
        self.class_means = None

        if self.args.GAN:
            batch_size = 32
            gan_train_loader = dataset.get_data_loaders_gan(batch_size)
            buf_x, buf_y, _, _ = self.buffer.get_all_data()
            gan_buf_loader = get_buffer_loaders_gan(buf_x.cpu(), buf_y.cpu(), batch_size, self.dataset)

            self.gan_generator, self.d = training_gan(gan_train_loader, task_id, 20, self.gan_generator, self.d)
            self.gan_generator, self.d = training_gan(gan_buf_loader, task_id, 5, self.gan_generator, self.d)

    def middle_task(self, dataset) -> None:
        with torch.no_grad():
            self.fill_buffer(self.buffer, dataset, self.current_task)
        self.class_means = None

    def compute_class_means(self, dataset) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means

        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels, _, _ = self.buffer.get_all_data(transform)

        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)

            each_class_mean = self.net(x_buf).mean(0)
            each_class_mean = self.norm_fun(each_class_mean)
            class_means.append(each_class_mean)
        self.class_means = torch.stack(class_means)

    def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy.
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """
        mode = self.net.training
        self.net.eval()
        samples_per_class = mem_buffer.buffer_size // (self.dataset.N_CLASSES_PER_TASK * (t_idx + 1))

        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y, buf_f, buf_task_id = self.buffer.get_all_data()
            mem_buffer.empty()

            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y, _y_f, _y_task_id = buf_x[idx], buf_y[idx], buf_f[idx], buf_task_id[idx]
                mem_buffer.add_data_our(
                    examples=_y_x[:samples_per_class],
                    labels=_y_y[:samples_per_class],
                    logits=_y_f[:samples_per_class],
                    task_labels=_y_task_id[:samples_per_class]
                )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        # 2.1 Extract all features
        a_x, a_y, a_f, a_f_o = [], [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))
            feats = self.net(x)
            a_f.append(feats.cpu())
            feats_origin = self.net.features(x)
            a_f_o.append(feats_origin.cpu())
        a_x, a_y, a_f, a_f_o = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_f_o)

        # 2.2 Compute class means
        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y, _f = a_x[idx], a_y[idx], a_f_o[idx]
            feats = a_f[idx]
            mean_feat = feats.mean(0, keepdim=True)
            self.task_class_means[_y.unique()] = F.normalize(mean_feat[0], p=2, dim=mean_feat[0].dim() - 1, eps=1e-12)

            running_sum = torch.zeros_like(mean_feat)
            i = 0
            while i < samples_per_class and i < feats.shape[0]:
                mean_feat_norm = self.norm_fun(mean_feat)
                sum_norm = self.norm_fun((feats + running_sum) / (i + 1))
                cost = (mean_feat_norm - sum_norm).norm(2, 1)

                idx_min = cost.argmin().item()

                mem_buffer.add_data_our(
                    examples=_x[idx_min:idx_min + 1].to(self.device),
                    labels=_y[idx_min:idx_min + 1].to(self.device),
                    logits=_f[idx_min:idx_min + 1].to(self.device),
                    task_labels=torch.tensor([t_idx]).to(self.device)
                )

                running_sum += feats[idx_min:idx_min + 1]
                feats[idx_min] = feats[idx_min] + 1e6
                i += 1

        assert len(mem_buffer.examples) <= mem_buffer.buffer_size

        self.net.train(mode)

    def fill_domain_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy.
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """
        mode = self.net.training
        self.net.eval()
        samples_per_class = mem_buffer.buffer_size // (self.dataset.N_CLASSES_PER_TASK * (t_idx + 1))

        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y, buf_f, buf_task_id = self.buffer.get_all_data_domain()
            mem_buffer.empty()

            for _task in buf_task_id.unique():
                idx_task = (buf_task_id == _task)
                _task_x, _task_y, _task_f, _task_task_id = buf_x[idx_task], buf_y[idx_task], buf_f[idx_task], buf_task_id[idx_task]

                for _y in _task_y.unique():
                    idx = (_task_y == _y)
                    _y_x, _y_y, _y_f, _y_task_id = _task_x[idx], _task_y[idx], _task_f[idx], _task_task_id[idx]
                    mem_buffer.add_data_our(
                        examples=_y_x[:samples_per_class],
                        labels=_y_y[:samples_per_class],
                        logits=_y_f[:samples_per_class],
                        task_labels=_y_task_id[:samples_per_class]
                    )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        # 2.1 Extract all features
        a_x, a_y, a_f, a_f_o = [], [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))
            feats = self.net(x)
            a_f.append(feats.cpu())
            feats_origin = self.net.features(x)
            a_f_o.append(feats_origin.cpu())
        a_x, a_y, a_f, a_f_o = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_f_o)

        # 2.2 Compute class means
        for _y_id in a_y.unique():
            idx = (a_y == _y_id)
            _x, _y, _f = a_x[idx], a_y[idx], a_f_o[idx]
            feats = a_f[idx]
            mean_feat = feats.mean(0, keepdim=True)
            self.task_class_means[_y_id][t_idx] = F.normalize(mean_feat[0], p=2, dim=mean_feat[0].dim() - 1, eps=1e-12)
            running_sum = torch.zeros_like(mean_feat)
            i = 0
            while i < samples_per_class and i < feats.shape[0]:
                cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

                idx_min = cost.argmin().item()

                mem_buffer.add_data_our(
                    examples=_x[idx_min:idx_min + 1].to(self.device),
                    labels=_y[idx_min:idx_min + 1].to(self.device),
                    logits=_f[idx_min:idx_min + 1].to(self.device),
                    task_labels=torch.tensor([t_idx]).to(self.device)
                )

                running_sum += feats[idx_min:idx_min + 1]
                feats[idx_min] = feats[idx_min] + 1e6
                i += 1

        assert len(mem_buffer.examples) <= mem_buffer.buffer_size

        self.net.train(mode)

    def begin_task(self, dataset):
        denorm = None
        if denorm is None:
            denorm = lambda x: x
        if self.current_task > 0:
            dataset.train_loader.dataset.targets = np.concatenate(
                [dataset.train_loader.dataset.targets,
                 self.buffer.labels.cpu().numpy()[:self.buffer.num_seen_examples]])
            if type(dataset.train_loader.dataset.data) == torch.Tensor:
                dataset.train_loader.dataset.data = torch.cat(
                    [dataset.train_loader.dataset.data, torch.stack([denorm(
                        self.buffer.examples[i].type(torch.uint8).cpu())
                        for i in range(self.buffer.num_seen_examples)]).squeeze(1)])
            else:
                dataset.train_loader.dataset.data = np.concatenate(
                    [dataset.train_loader.dataset.data, torch.stack([((denorm(self.buffer.examples[i]) * 255).type(torch.uint8).cpu())
                        for i in range(self.buffer.num_seen_examples)]).numpy().swapaxes(1, 3)])

    def gan_augment(self, num_augment, task_idx, buf_inputs, buf_labels):
        if self.args.GAN:
            buf_inputs_gen = []
            self.gan_generator.eval()
            for i in range(num_augment):
                z = torch.randn((len(buf_inputs), self.gan_generator.z_dim)).to('cuda')
                buf_inputs_gen.append(self.gan_generator(buf_inputs, z))
            buf_inputs_gen = torch.cat(buf_inputs_gen)  # torch.Size([16, 1, 28, 28])

            buf_inputs = torch.cat((buf_inputs, buf_inputs_gen), 0)  # torch.Size([32, 1, 28, 28])
            buf_labels = torch.cat((buf_labels, buf_labels), 0)  # torch.Size([32])
            # idx = torch.randint(len(buf_inputs), (int(self.args.minibatch_size / (task_idx + 1) * task_idx),))
            # buf_inputs = buf_inputs[idx]
            # buf_labels = buf_labels[idx]

            self.gan_generator.train()

            return buf_inputs, buf_labels

    def render_img(self, arr, denorm):
        arr = denorm(arr) * 255
        arr = arr.cpu().detach().numpy()

        if self.gan_generator.channels == 1:
            arr = np.uint8(arr)
            # display(Image.fromarray(arr, mode="L").transpose(PIL.Image.TRANSPOSE))
            Image.fromarray(arr, mode="L").transpose(PIL.Image.TRANSPOSE).show()
            i = 0
            while os.path.exists("igan/results/image%s.png" % i):
                i += 1
            Image.fromarray(arr).transpose(PIL.Image.TRANSPOSE).save("igan/results/image%s.png" % i)
        else:
            arr = np.uint8(arr).swapaxes(0, 2)
            # display(Image.fromarray(arr, mode="RGB").transpose(PIL.Image.TRANSPOSE))
            Image.fromarray(arr, mode="RGB").transpose(PIL.Image.TRANSPOSE).show()
            i = 0
            while os.path.exists("igan/results/image_imageNet_%s.png" % i):
                i += 1
            Image.fromarray(arr).transpose(PIL.Image.TRANSPOSE).save("igan/results/image_imageNet_%s.png" % i)

    def norm_fun(self, feat):
        return F.normalize(feat, p=2, dim=feat.dim() - 1, eps=1e-12)
