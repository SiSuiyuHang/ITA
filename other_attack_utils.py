import cv2
import torch
import numpy as np
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
import torchvision
import math
from tqdm import tqdm
from nets.GAN import *
from torchvision.utils import save_image
from torchvision import transforms
from DCT import *
from DWT import *
from torch.autograd import Variable as V
from nets.CGAN import *
from nets.CDAGenerators import *
from nets.TTPGenerators import *



class AdvGAN_Attack:
    def __init__(self, name, device, model, model_num_labels, eps, epochs, g_save_path, box_min=0, box_max=1):
        self.name = name
        self.device = device
        self.model = model
        self.model_num_labels = model_num_labels
        self.eps = eps
        self.epochs = epochs
        self.g_save_path = g_save_path
        self.box_min = box_min
        self.box_max = box_max

        self.netG = Generator().to(device)
        self.netDisc = Discriminator().to(device)

        # initialize all weights
        self.netG.apply(self.weights_init)
        self.netDisc.apply(self.weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=0.001)

    # custom weights initialization called on netG and netD
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -self.eps, self.eps) + x   # todo
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()
            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader):
        for epoch in range(1, self.epochs+1):
            if epoch == 10:   # 原50
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=0.0001)
            if epoch == 30:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=0.00001)

            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            train_bar = tqdm(train_dataloader)
            for i, data in enumerate(train_bar, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            g_save_path = self.g_save_path + f'{self.name}/'+f'netG_withc_epoch_{epoch}' + '.pth'  # todo
            print(f'save_path:{g_save_path}')
            # save generator
            if epoch % 20 == 0:
                netG_file_name = g_save_path
                torch.save(self.netG.state_dict(), netG_file_name)
                print(f'netG_file_name:{netG_file_name}')


class Spectrum_Simulation_Attack:
    #  https://github.com/yuyang-long/SSA
    #  Spectrum Simulation Attack (ECCV'2022 ORAL)
    def __init__(self, name, device, model, model_num_labels, eps,
                  iterations, iterations_frequency):
        self.device = device
        self.model = model
        self.name = name
        self.model_num_labels = model_num_labels

        self.eps = eps
        self.iterations = iterations
        self.iterations_frequency = iterations_frequency
        self.alpha = self.eps / self.iterations   # （16/255）/ 10 = 1.6/255
        self.sigma = 16.0   # 按照论文设置
        self.rho = 0.5  # 按照论文设置 Tuning factor

    def clip_by_tensor(self, t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def attack(self, test_image, test_label):
        images_min = self.clip_by_tensor(test_image - self.eps, 0.0, 1.0)   # eps 设置的是16/255 论文16 因此又除以255
        images_max = self.clip_by_tensor(test_image + self.eps, 0.0, 1.0)
        x = test_image.clone()
        for t in range(self.iterations):
            image_width = x.shape[-1]
            noise = 0
            for it in range(self.iterations_frequency):
                gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (self.sigma / 255)
                gauss = gauss.to(self.device)
                x_dct = dct_2d(x + gauss).to(self.device)
                mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).to(self.device)  # 随机放大或缩小0.5-1.5
                x_idct = idct_2d(x_dct * mask).to(self.device)
                x_idct = V(x_idct, requires_grad=True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                # output_v3 = model(DI(x_idct))

                output_v3 = self.model(x_idct)
                loss = F.cross_entropy(output_v3, test_label)
                loss.backward()
                noise += x_idct.grad.data
            noise = noise / self.iterations_frequency

            # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
            # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

            # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
            # noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            # noise = momentum * grad + noise
            # grad = noise

            x = x + self.alpha * torch.sign(noise)
            x = self.clip_by_tensor(x, images_min, images_max)
        return x.detach()


class C_GSP:
    #  https://github.com/ShawnXYang/C-GSP/blob/master
    #  Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks eccv 2022
    def __init__(self, device, model, name, number_class, epsilon, epochs, g_save_path):
        self.device = device
        self.model = model
        self.name = name
        self.number_class = number_class
        self.epsilon = epsilon
        self.epochs = epochs
        self.g_save_path = g_save_path

        self.netG = ConGeneratorResnet().to(device)
        #self.netG.apply(self.weights_init)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss()
        self.method = 'C_GSP/'

        self.g_save_path = self.g_save_path + self.method + f'{self.name}/' + 'netG_epoch_'  # todo
        print(f'save_path:{self.g_save_path}')

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):
            loss_total = 0

            train_bar = tqdm(train_dataloader)
            num_batch = len(train_dataloader)
            for i, data in enumerate(train_bar, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                inverted = 1 - labels
                onehot_labels = torch.eye(n=self.number_class, device=self.device)[labels]
                inverted_onehot = 1 - onehot_labels
                self.netG.train()  # 看有没有dropout  todo
                self.optimizer_G.zero_grad()
                perturbation = self.netG(input=images, z_one_hot=inverted_onehot, eps=self.epsilon)
                # todo 扰动在生成后已经进行了限制，不需要再扰动裁剪了
                adv_images = perturbation + images
                adv_images = torch.clamp(adv_images, 0.0, 1.0)

                loss = self.criterion(self.model(adv_images), inverted)   # todo 注意标签
                loss.backward()
                self.optimizer_G.step()
                loss_total += loss
                train_bar.desc = f"train epoch:{epoch, epochs} " f"bce_loss:{loss_total / num_batch} "

            g_save_name = self.g_save_path + str(epoch) + '.pth'  # todo
            print(f'save_path:{g_save_name}')
            if epoch % 5 == 0:
                netG_file_name = g_save_name
                torch.save(self.netG.state_dict(), netG_file_name)
                print(f'netG_file_name:{netG_file_name}')


class CDA:
    #  https://github.com/Muzammal-Naseer/CDA
    #  Cross-Domain Transferability of Adversarial Perturbations  nips 2019
    def __init__(self, device, model, name, number_class, epsilon, g_save_path):
        self.device = device
        self.model = model
        self.name = name
        self.number_class = number_class
        self.epsilon = epsilon
        self.g_save_path = g_save_path
        self.netG = CDAGeneratorResnet().to(device)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.target = 1       # todo -1 if untargeted else 1'
        self.criterion = nn.CrossEntropyLoss()
        self.method = 'CDA/'
        self.g_save_path = self.g_save_path + self.method + f'{self.name}/' + 'netG_epoch_'  # todo
        print(f'save_path:{self.g_save_path}')

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):
            train_bar = tqdm(train_dataloader)
            for i, data in enumerate(train_bar):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                # whatever the model think about the input
                # todo  论文代码是用的是模型预测的标签，但是论文中说的是真实标签
                pre_label = self.model(images.clone().detach()).argmax(dim=-1).detach()

                if self.target == -1:   # 无目标攻击
                    targte_label = torch.LongTensor(images.size(0))
                    targte_label.fill_(self.target)
                    targte_label = targte_label.to(self.device)
                elif self.target == 1:
                    targte_label = 1 - labels

                self.netG.train()
                self.optimizer_G.zero_grad()
                # todo CDA生成器最后没有对像素进行裁剪，因此最后生成的就是对抗样本
                adv_image = self.netG(images)

                # Projection
                adv_image = torch.min(torch.max(adv_image, images - self.epsilon), images + self.epsilon)
                adv_image = torch.clamp(adv_image, 0.0, 1.0)

                if self.target == -1:
                    # Gradient accent (Untargetted Attack)
                    adv_out = self.model(adv_image)
                    img_out = self.model(images)
                    loss = - self.criterion(adv_out - img_out, labels)
                    # todo  论文是用的是模型预测的标签
                    #loss = - self.criterion(adv_out - img_out, pre_label)
                else:
                    # Gradient decent (Targetted Attack)
                    # loss = criterion(model(normalize(adv)), targte_label)
                    loss = self.criterion(self.model(adv_image), targte_label) + self.criterion(self.model(images), labels)
                    # todo  论文代码是用的是模型预测的标签，但是论文中说的是真实标签
                    #loss = self.criterion(self.model(adv_image), targte_label) + self.criterion(self.model(images), pre_label)
                loss.backward()
                self.optimizer_G.step()
                train_bar.desc = f"train epoch:{epoch, epochs} " f"loss:{loss} "

            g_save_name = self.g_save_path + str(epoch) + '.pth'  # todo
            print(f'save_path:{g_save_name}')
            if epoch % 5 == 0:
                netG_file_name = g_save_name
                torch.save(self.netG.state_dict(), netG_file_name)
                print(f'netG_file_name:{netG_file_name}')


class SSAH(nn.Module):
    """"
    Parameters:
    -----------

    """

    def __init__(self, model, device, num_iteration: int = 150, learning_rate: float = 0.001,
                 Targeted: bool = False, m: float = 0.2, alpha: float = 1,
                 lambda_lf: float = 0.1, wave: str = 'haar',) -> None:
        super(SSAH, self).__init__()
        self.model = model
        self.device = device
        self.lr = learning_rate
        self.target = Targeted
        self.num_iteration = num_iteration
        self.m = m
        self.alpha = alpha
        self.lambda_lf = lambda_lf

        # self.encoder_fea = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        # self.encoder_fea = nn.DataParallel(self.encoder_fea).to(self.device)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        # self.model = nn.DataParallel(self.model)

        self.encoder_fea = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.model = self.model.to(self.device)

        # self.normalize_fn = normalize_fn(self.dataset)

        self.DWT = DWT_2D_tiny(wavename=wave)
        self.IDWT = IDWT_2D_tiny(wavename=wave)

    def fea_extract(self, inputs: torch.Tensor) -> torch.Tensor:
        fea = self.encoder_fea(inputs)
        b, c, h, w = fea.shape
        fea = self.avg_pool(fea).view(b, c)
        return fea

    def cal_sim(self, adv, inputs):
        adv = F.normalize(adv, dim=1)
        inputs = F.normalize(inputs, dim=1)

        r, c = inputs.shape
        sim_matrix = torch.matmul(adv, inputs.T)
        mask = torch.eye(r, dtype=torch.bool).to(self.device)
        pos_sim = sim_matrix[mask].view(r, -1)
        neg_sim = sim_matrix.view(r, -1)
        return pos_sim, neg_sim

    def select_setp1(self, pos_sim, neg_sim):
        neg_sim, indices = torch.sort(neg_sim, descending=True)
        pos_neg_sim = torch.cat([pos_sim, neg_sim[:, -1].view(pos_sim.shape[0], -1)], dim=1)
        return pos_neg_sim, indices

    def select_step2(self, pos_sim, neg_sim, indices):
        hard_sample = indices[:, -1]
        ones = torch.sparse.torch.eye(neg_sim.shape[1]).to(self.device)
        hard_one_hot = ones.index_select(0, hard_sample).bool()
        hard_sim = neg_sim[hard_one_hot].view(neg_sim.shape[0], -1)
        pos_neg_sim = torch.cat([pos_sim, hard_sim], dim=1)
        return pos_neg_sim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            #inputs_fea = self.fea_extract(self.normalize_fn(inputs))
            inputs_fea = self.fea_extract(inputs)
        # low frequency component
        inputs_ll = self.DWT(inputs)
        inputs_ll = self.IDWT(inputs_ll)

        # changes of variables
        eps = 3e-7
        modifier = torch.arctanh(inputs * (2 - eps * 2) - 1 + eps)
        modifier = V(modifier, requires_grad=True)
        modifier = modifier.to(self.device)
        optimizer = torch.optim.Adam([modifier], lr=self.lr)

        lowFre_loss = nn.SmoothL1Loss(reduction='sum')

        for step in range(self.num_iteration):
            optimizer.zero_grad()
            self.encoder_fea.zero_grad()

            adv = 0.5 * (torch.tanh(modifier) + 1)
            adv_fea = self.fea_extract(adv)

            adv_ll = self.DWT(adv)
            adv_ll = self.IDWT(adv_ll)

            pos_sim, neg_sim = self.cal_sim(adv_fea, inputs_fea)
            # select the most dissimilar one in the first iteration
            if step == 0:
                pos_neg_sim, indices = self.select_setp1(pos_sim, neg_sim)

            # record the most dissimilar ones by indices and calculate similarity
            else:
                pos_neg_sim = self.select_step2(pos_sim, neg_sim, indices)

            sim_pos = pos_neg_sim[:, 0]
            sim_neg = pos_neg_sim[:, -1]

            w_p = torch.clamp_min(sim_pos.detach() - self.m, min=0)
            w_n = torch.clamp_min(1 + self.m - sim_neg.detach(), min=0)

            adv_cost = torch.sum(torch.clamp(w_p * sim_pos - w_n * sim_neg, min=0))
            lowFre_cost = lowFre_loss(adv_ll, inputs_ll)
            total_cost = self.alpha * adv_cost + self.lambda_lf * lowFre_cost

            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()

        adv = 0.5 * (torch.tanh(modifier.detach()) + 1)
        return adv

    def common(self, targets, pred):
        common_id = np.where(targets.cpu() == pred.cpu())[0]
        return common_id






