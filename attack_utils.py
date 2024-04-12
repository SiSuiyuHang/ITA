import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.optim import lr_scheduler
from nets.CGAN import *
from DWT import *
from torchvision.utils import save_image
from torchvision import transforms


def low_Frequency_loss(x, adv_image):
    """
    限制原始图片、对抗样本在低频部分上的相似度 尽可能一致
    噪声就添加到高频信息上
    :param x:
    :param adv_image:
    :return: 两张图片在低频部分的相似
    """
    DWT = DWT_2D_tiny(wavename='haar')
    IDWT = IDWT_2D_tiny(wavename='haar')

    x_ll = DWT(x)
    x_ll = IDWT(x_ll)

    adv_ll = DWT(adv_image)
    adv_ll = IDWT(adv_ll)

    lowFre_loss = nn.SmoothL1Loss(reduction='sum')
    lowFre_cost = lowFre_loss(adv_ll, x_ll)

    # adv_ll = torch.squeeze(adv_ll, dim=0)  # 把batch删去了
    # adv_numpy_data = adv_ll.permute(1, 2, 0).cpu().detach().numpy()
    #
    # x_ll = torch.squeeze(x_ll, dim=0)  # 把batch删去了
    # x_numpy_data = x_ll.permute(1, 2, 0).cpu().detach().numpy()
    #
    # plt.imshow(adv_numpy_data)
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(x_numpy_data)
    # plt.axis('off')
    # plt.show()

    # # 计算余弦相似度
    # cos_similarity = F.cosine_similarity(x_ll, adv_ll, dim=1)
    # # 可以选择使用1减余弦相似度以作为损失，这样越相似的图像损失越小
    # lowFre_cost = 1 - cos_similarity

    return lowFre_cost


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CondGeneratorResnet_Transfer_Attack:
    def __init__(self, device, model, name, number_class, epsilon, g_save_path):
        self.device = device
        self.model = model
        self.name = name
        self.number_class = number_class
        self.epsilon = epsilon
        self.g_save_path = g_save_path

        self.netG = ConGeneratorResnet().to(device)
        self.netG.train()  # 看有没有dropout  todo
        self.netG.apply(weights_init)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        self.scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=5, gamma=0.1)

    def train_batch(self, x, labels):

        for i in range(1):

            self.optimizer_G.zero_grad()
            onehot_labels = torch.eye(n=self.number_class, device=self.device)[labels]
            inverted_onehot = 1 - onehot_labels
            perturbation = self.netG(input=x, z_one_hot=inverted_onehot, eps=self.epsilon)  # todo 扰动在生成后已经进行了限制，不需要再裁剪了
            adv_images = perturbation + x
            adv_images = torch.clamp(adv_images, 0, 1)

            # cal low_frequency_loss
            low_fre_loss = low_Frequency_loss(x, adv_images)
            # cal l2-loss
            # perturb_loss = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1))
            # perturb_loss = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), p=float('inf'), dim=1))

            '''
            修改损失
            只关心两个 logits 值之间的相对距离
            只关心两张图片的logits值之间的相对距离,不需要解释绝对误差大小,更关心距离排名。这种情况下,建议使用RMSE会更好一些。

            原因如下:
            RMSE与目标值量纲相同,可以更直观地反映距离的相对大小。
            RMSE进行了sqrt调整,距离排名会与人眼判断更一致。
            MSE的平方运算可能会放大某些样本的距离差异,对距离排名判断产生影响。
            对于只比较距离排名的情况,RMSE通常会给出更合理的结果。
            所以对于您只要对比两张图片logits距离排名的情况,建议直接使用RMSE,而不要使用MSE。RMSE可以更准确地反映距离的相对大小,给出更符合直觉的距离排名。
            当然,如果后续需要进行误差的进一步数学运算和统计分析,那么可以再考虑使用MSE。但仅就您的这一具体需求来说,RMSE更为合适。
            '''
            # todo 1  使用RMSE  bceloss adv target-one-hot 效果不好    不如todo2
            # logits_loss_function = nn.MSELoss()
            # logits_rmse_loss = - torch.sqrt(logits_loss_function(self.model(adv_images), self.model(x)))
            # bce_loss_function = nn.BCEWithLogitsLoss()
            # bce_loss = bce_loss_function(self.model(adv_images), inverted_onehot)  # todo 这里是onehot

            # todo 2 使用RMSE bceloss adv gt
            logits_loss_function = nn.MSELoss()
            logits_rmse_loss = - torch.sqrt(logits_loss_function(self.model(adv_images), self.model(x)))
            bce_loss_function = nn.BCEWithLogitsLoss()
            bce_loss = - bce_loss_function(self.model(adv_images), onehot_labels)

            # todo 3 使用 MSE bceloss adv gt
            # logits_loss_function = nn.MSELoss()
            # logits_mse_loss = - logits_loss_function(self.model(adv_images), self.model(x))
            # bce_loss_function = nn.BCEWithLogitsLoss()
            # bce_loss = - bce_loss_function(self.model(adv_images), onehot_labels)


            low_loss_lambda = 1
            logits_rmse_loss_lambda = 0.2
            logits_mse_loss_lambda = 0.2
            bce_loss_lamdba = 1
            total_loss = low_loss_lambda * low_fre_loss + logits_rmse_loss_lambda * logits_rmse_loss + bce_loss_lamdba * bce_loss
            #total_loss = low_loss_lambda * low_fre_loss + bce_loss_lamdba * bce_loss
            #total_loss = low_loss_lambda * low_fre_loss + logits_mse_loss_lambda * logits_mse_loss + bce_loss_lamdba * bce_loss

            total_loss.backward()
            # # 更新模型参数
            self.optimizer_G.step()

        return bce_loss.item(), logits_rmse_loss.item(), low_fre_loss.item()
        #return bce_loss.item(), low_fre_loss.item()
        # return bce_loss.item(), logits_mse_loss.item(), low_fre_loss.item()


    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs + 1):

            bce_loss_sum = 0
            logits_loss_sum = 0
            low_fre_loss_sum = 0

            train_bar = tqdm(train_dataloader)
            num_batch = len(train_dataloader)
            for i, data in enumerate(train_bar, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                #bce_loss, logits_loss_batch, low_fre_batch = self.train_batch(images, labels)
                #bce_loss,  low_fre_batch = self.train_batch(images, labels)
                bce_loss, logits_loss_batch, low_fre_batch = self.train_batch(images, labels)

                bce_loss_sum += bce_loss
                logits_loss_sum += logits_loss_batch
                low_fre_loss_sum += low_fre_batch

                train_bar.desc = f"train epoch:{epoch, epochs} " \
                                 f"bce_loss:{bce_loss_sum / num_batch} " \
                                 f"logits_loss:{logits_loss_sum / num_batch} " \
                                 f"low_fre_loss:{low_fre_loss_sum / num_batch}"
                # train_bar.desc = f"train epoch:{epoch, epochs} " \
                #                  f"bce_loss:{bce_loss_sum / num_batch} " \
                #                  f"low_fre_loss:{low_fre_loss_sum / num_batch}"

            # print statistics
            # print(f'epoch:{epoch}')
            # print(f'bce_relative_loss:{bce_relative_loss_sum / num_batch}\n'
            #       f'perturb_loss:{perturb_loss_sum / num_batch}\n'
            #       f'low_fre_loss:{low_fre_loss_sum / num_batch}\n')

            # 更新学习率
            self.scheduler.step()

            #g_save_path = self.g_save_path + f'{self.name}/eps16/' + 'no_gauss_netG_epoch_' + str(epoch) + '.pth'  # todo
            #g_save_path = self.g_save_path + f'{self.name}/eps16-no-logits/' + 'netG_epoch_' + str(epoch) + '.pth'  # todo
            g_save_path = self.g_save_path + f'{self.name}/eps16-mseloss/' + 'netG_epoch_' + str(epoch) + '.pth'  # todo
            print(f'save_path:{g_save_path}')
            # save generator
            if epoch % 5 == 0:  # todo
                netG_file_name = g_save_path
                torch.save(self.netG.state_dict(), netG_file_name)
                print(f'netG_file_name:{netG_file_name}')

