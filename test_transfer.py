import argparse
import cv2
import torch
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms, datasets, models
from sklearn.metrics import roc_auc_score
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchattacks
from tqdm import tqdm
from data_utils import *
from nets.CGAN import *
from nets.GAN import *
from nets.Xception import *
from nets.Mesonet4 import *
from nets.MesoInception import *
from nets.Gramnet import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from other_attack_utils import *
from attack_utils import *



def check_img_data_range(img):
    if img.dtype == np.uint8:
        return 255
    else:
        return 1.0

def cal_psnr(img, adv_img):
    if type(img) == torch.Tensor:
        img = img.cpu().data.numpy()
    if type(adv_img) == torch.Tensor:
        adv_img = adv_img.cpu().data.numpy()
    mse = np.mean((img / 255. - adv_img / 255.) ** 2)
    # if mse < 1.0e-10:
    #     return 100   # todo
    return peak_signal_noise_ratio(img, adv_img, data_range=check_img_data_range(img))

def cal_ssim(img, adv_img):
    img_np = np.squeeze(img.cpu().data.numpy())    # 去除batch
    adv_img_np = np.squeeze(adv_img.cpu().data.numpy())  # 去除batch
    img_np32 = img_np.astype(np.float32)
    adv_img_np32 = adv_img_np.astype(np.float32)
    win_size = min(img_np.shape[0], img_np.shape[1])  # 设置窗口大小为图像较小一边的大小

    return structural_similarity(img_np32, adv_img_np32, win_size=win_size, channel_axis=-1,
                                 data_range=check_img_data_range(img))

def main():

    args = parse.parse_args()
    name = args.name
    print(f'name:{name}')
    attack_model = args.attack_model
    print(f'attack_model:{attack_model}')
    device = args.device
    print(f'device:{device}')
    number_class = args.number_class
    eps = args.eps
    print(f'eps:{eps}')
    Attack_method = args.Attack_method
    print(f'Attack_method is:{Attack_method}')

    image_path = args.test_path
    print(f'image_path:{args.test_path}')

    batch_size = args.batch_size
    nw = args.number_work

    print("加载代理模型")
    # if name == 'Vgg':
    #     Vgg_weight_path = args.Vgg_weight_path
    #     model = models.vgg16_bn(weights=None, num_classes=number_class).to(device)
    #     model.load_state_dict(torch.load(Vgg_weight_path, map_location=device))
    #     print(f'base_model_weight_path:{Vgg_weight_path}')
    # elif name == 'Resnet':
    #     Resnet_weight_path = args.Resnet_weight_path
    #     model = models.resnet50(weights=None, num_classes=number_class).to(device)
    #     model.load_state_dict(torch.load(Resnet_weight_path, map_location=device))
    #     print(f'base_model_weight_path:{Resnet_weight_path}')
    # elif name == 'Mesonet':
    #     Mesonet_weight_path = args.Mesonet_weight_path
    #     model = Mesonet4(num_classes=number_class).to(device)
    #     model.load_state_dict(torch.load(Mesonet_weight_path, map_location=device))
    #     print(f'base_model_weight_path:{Mesonet_weight_path}')
    # elif name == 'MesoInception':
    #     MesoInception_weight_path = args.MesoInception_weight_path
    #     model = MesoInception4(num_classes=number_class).to(device)
    #     model.load_state_dict(torch.load(MesoInception_weight_path, map_location=device))
    #     print(f'base_model_weight_path:{MesoInception_weight_path}')
    # elif name == 'EfficientNetb4':
    #     EfficientNetb4_weight_path = args.EfficientNetb4_weight_path
    #     model = models.efficientnet_b4(num_classes=number_class).to(device)
    #     model.load_state_dict(torch.load(EfficientNetb4_weight_path, map_location=device))
    #     print(f'base_model_weight_path:{EfficientNetb4_weight_path}')
    # elif name == 'Xception':
    #     Xception_weight_path = args.Xception_weight_path
    #     model = xception(num_classes=number_class).to(device)
    #     model.load_state_dict(torch.load(Xception_weight_path, map_location=device))
    #     print(f'base_model_weight_path:{Xception_weight_path}')
    #
    if name == 'Resnet':
        Resnet_weight_path = args.Resnet_weight_path
        model = models.resnet50(weights=None, num_classes=number_class).to(device)
        model.load_state_dict(torch.load(Resnet_weight_path, map_location=device))
        print(f'base_model_weight_path:{Resnet_weight_path}')
    elif name == 'Densenet169':
        Densenet169_weight_path = args.Densenet169_weight_path
        model = models.densenet169(weights=None, num_classes=number_class).to(device)
        model.load_state_dict(torch.load(Densenet169_weight_path, map_location=device))
        print(f'base_model_weight_path:{Densenet169_weight_path}')
    elif name == 'Densenet121':
        Densenet121_weight_path = args.Densenet121_weight_path
        model = models.densenet121(weights=None, num_classes=number_class).to(device)
        model.load_state_dict(torch.load(Densenet121_weight_path, map_location=device))
        print(f'base_model_weight_path:{Densenet121_weight_path}')

    test_dataset = datasets.ImageFolder(root=image_path, transform=data_transform["attack"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    print("加载被攻击模型")
    if attack_model == 'Vgg':
        attack_weight_path = args.Vgg_weight_path
        attack_adv_model = models.vgg16_bn(weights=None,num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'Resnet':
        attack_weight_path = args.Resnet_weight_path
        attack_adv_model = models.resnet50(weights=None, num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'Resnet152':
        attack_weight_path = args.Resnet152_weight_path
        attack_adv_model = models.resnet152(weights=None, num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'Mesonet':
        attack_weight_path = args.Mesonet_weight_path
        attack_adv_model = Mesonet4(num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'MesoInception':
        attack_weight_path = args.MesoInception_weight_path
        attack_adv_model = MesoInception4(num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'EfficientNetb4':
        attack_weight_path = args.EfficientNetb4_weight_path
        attack_adv_model = models.efficientnet_b4(weights=None, num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'Xception':
        attack_weight_path = args.Xception_weight_path
        attack_adv_model = xception(num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'Densenet121':
        attack_weight_path = args.Densenet121_weight_path
        attack_adv_model = models.densenet121(weights=None, num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'Densenet169':
        attack_weight_path = args.Densenet169_weight_path
        attack_adv_model = models.densenet169(weights=None, num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'MobilenetV2':
        attack_weight_path = args.MobilenetV2_weight_path
        attack_adv_model = models.mobilenet_v2(weights=None, num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'Maxvit':
        attack_weight_path = args.Maxvit_weight_path
        attack_adv_model = models.maxvit_t(weights=None, num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))
    elif attack_model == 'GramNet':
        attack_weight_path = args.GramNet_weight_path
        attack_adv_model = GramNet(num_classes=number_class).to(device)
        attack_adv_model.load_state_dict(torch.load(attack_weight_path, map_location=device))

    print(f'attack_weight_path:{attack_weight_path}')

    model.eval()  # todo 做攻击的时候一定要加 base_model
    attack_adv_model.eval()  # 要攻击的模型

    init_uncorrect_count = 0
    true_labels = []
    predicted_scores = []
    total_psnr = 0
    total_ssim = 0
    attack_model_suc_attack_num = 0
    attack_model_no_suc_attack_num = 0

    fake0Toreal1_num = 0
    real1Tofake0_num = 0
    test_bar = tqdm(test_loader)
    for i, data in enumerate(test_bar, 0):

        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        onehot_labels = torch.eye(n=2, device=device)[test_label]
        inverted_onehot = 1 - onehot_labels
        # base_model预测不正确 不攻击
        init_pred = model(test_img)
        init_pred_label = init_pred.max(1, keepdim=True)[1]
        if init_pred_label.item() != test_label.item():
            init_uncorrect_count += 1
            continue

        if Attack_method == 'BIM':
            attack = torchattacks.BIM(model, eps, alpha=1.6/255, steps=10)
            adv_image = attack(test_img, test_label)
        elif Attack_method == 'MIFGSM':
            attack = torchattacks.MIFGSM(model, eps, steps=10, decay=1.0)
            adv_image = attack(test_img, test_label)
        elif Attack_method == 'DIFGSM':
            attack = torchattacks.DIFGSM(model, eps, alpha=1.6 / 255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
            adv_image = attack(test_img, test_label)
        elif Attack_method == 'DI2FGSM':  # todo 动量参数
            attack = torchattacks.DIFGSM(model, eps, alpha=1.6 / 255, steps=10, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
            adv_image = attack(test_img, test_label)
        elif Attack_method == 'TIFGSM':
            attack = torchattacks.TIFGSM(model, eps, alpha=1.6 / 255, steps=10, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
            adv_image = attack(test_img, test_label)
        elif Attack_method == 'SINIFGSM':
            attack = torchattacks.SINIFGSM(model, eps, alpha=1.6 / 255, steps=10, decay=1.0, m=5)
            adv_image = attack(test_img, test_label)
        elif Attack_method == 'PGD':
            attack = torchattacks.PGD(model, eps, alpha=1.6 / 255, steps=10, random_start=True)
            adv_image = attack(test_img, test_label)
        elif Attack_method == 'AdvGan':
            Generator_weight_path = args.Generator_weight_path + 'AdvGan/' f'{name}/' + 'netG_withc_epoch_40.pth'
            G_attack = Generator().to(device)
            G_attack.load_state_dict(torch.load(Generator_weight_path))
            G_attack.eval()
            perturbation = G_attack(test_img)
            adv_image = torch.clamp(perturbation, -eps, eps) + test_img
            adv_image = torch.clamp(adv_image, 0, 1)
        elif Attack_method == 'SSA':
            # Spectrum Simulation Attack (ECCV'2022 ORAL)
            # https://github.com/yuyang-long/SSA/tree/master
            ssa_attack = Spectrum_Simulation_Attack(name, device, model, number_class, eps, iterations=10,
                                                    iterations_frequency=20)
            adv_image = ssa_attack.attack(test_img, test_label)

        elif Attack_method == 'SSAH':
            att = SSAH(model, num_iteration=150, learning_rate=0.001, device=device)
            adv_image = att(test_img)

        elif Attack_method =='C_GSP':
            #  https://github.com/ShawnXYang/C-GSP/blob/master
            #  Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks
            Generator_weight_path = args.Generator_weight_path + 'C_GSP/' f'{name}/' + 'netG_epoch_20.pth'
            cgsp_G = ConGeneratorResnet().to(device)
            cgsp_G.load_state_dict(torch.load(Generator_weight_path))
            cgsp_G.eval()
            adv_image = cgsp_G(test_img, inverted_onehot, eps) + test_img
            adv_image = torch.clamp(adv_image, 0, 1)
        elif Attack_method == 'CDA':
            #  https://github.com/Muzammal-Naseer/CDA
            #  Cross-Domain Transferability of Adversarial Perturbations
            Generator_weight_path = args.Generator_weight_path + 'CDA/' f'{name}/' + 'netG_epoch_20.pth'
            cda_G = CDAGeneratorResnet().to(device)
            cda_G.load_state_dict(torch.load(Generator_weight_path))
            kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).to(device)
            adv_image = kernel(cda_G(test_img))
            adv_image = torch.min(torch.max(adv_image, test_img - eps), test_img + eps)
            adv_image = torch.clamp(adv_image, 0.0, 1.0)

        elif Attack_method == 'MyAttack':
            #Generator_weight_path = args.MyGenerator_weight_path + f'{name}/eps16/' + 'netG_epoch_20.pth'
            #Generator_weight_path = args.MyGenerator_weight_path + f'{name}/eps16-no-logits/' + 'netG_epoch_20.pth'
            Generator_weight_path = args.MyGenerator_weight_path + f'{name}/eps16-mseloss/' + 'netG_epoch_20.pth'
            #Generator_weight_path = args.MyGenerator_weight_path + f'{name}/eps8/' + 'netG_epoch_20.pth'  #  todo
            #Generator_weight_path = args.MyGenerator_weight_path + f'{name}/' + 'netG_epoch_20.pth'
            CG_attack = ConGeneratorResnet().to(device)
            CG_attack.load_state_dict(torch.load(Generator_weight_path))
            CG_attack.eval()
            adv_image = CG_attack(test_img, inverted_onehot, eps) + test_img
            adv_image = torch.clamp(adv_image, 0, 1)

        adv_logits = attack_adv_model(adv_image)
        adv_model_pred_lab = torch.argmax(adv_logits, 1)

        true_labels.extend(test_label.cpu().numpy())  # 将真实标签添加到列表中
        predicted_scores.extend(adv_logits[:, 1].cpu().detach().numpy())  # 将模型的预测概率分数添加到列表中

        if adv_model_pred_lab == test_label:   # todo test时batch为1
            attack_model_no_suc_attack_num += 1
        else:
            attack_model_suc_attack_num += 1
            total_psnr += cal_psnr(test_img, adv_image)
            total_ssim += cal_ssim(test_img, adv_image)
            if adv_model_pred_lab.item() == 1:   # todo adv的标签为1
                fake0Toreal1_num += 1  # todo 由0攻击成了1
            elif adv_model_pred_lab.item() == 0:   # todo adv的标签为0
                real1Tofake0_num += 1  # todo 由1攻击成了0

    len_test_loader = len(test_loader)
    attack_count = (len_test_loader - init_uncorrect_count)
    base_model_init_acc = attack_count / len_test_loader
    after_attack_acc = attack_model_no_suc_attack_num / attack_count
    after_attack_asc = attack_model_suc_attack_num / attack_count
    after_attack_auc = roc_auc_score(true_labels, predicted_scores)
    average_psnr = total_psnr / attack_count
    average_ssim = total_ssim / attack_count

    #print(f'Generator_weight_path:{Generator_weight_path}')
    print(f'test dataset:{len_test_loader}')
    print(f'init_uncorrect_count:{init_uncorrect_count}')
    print(f'attack number count:{attack_count}')
    print(f'base_model_init_acc:{base_model_init_acc}')
    print(f'after attack still num_correct:{attack_model_no_suc_attack_num}')
    print(f'after attack acc:{after_attack_acc}')
    print(f'fake0Toreal1_num:{fake0Toreal1_num}')
    print(f'real1Tofake0_num:{real1Tofake0_num}')
    print(f'suc_attack_num:{attack_model_suc_attack_num}')
    print(f'successful attack:{after_attack_asc} ')
    print(f'after attack auc:{after_attack_auc} ')
    print(f'average psnr:{average_psnr}')
    print(f'average ssim:{average_ssim}')


if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='Densenet169', help='base model')  # Resnet  Densenet169
    # Vgg Resnet Resnet152 Mesonet MesoInception  EfficientNetb4 Xception
    # Densenet121 Densenet169 MobilenetV2  Maxvit GramNet
    # Resnet Resnet152 Mesonet EfficientNetb4 Xception Densenet121 Densenet169  Maxvit GramNet
    parse.add_argument('--attack_model', '-am', type=str, default='Densenet169', help='attack model')

    parse.add_argument('--test_path', '-tp', type=str, default='../datasets/FF++_dataset_by_ls/jpg/c23/total/every250_total_1k_test/')
    parse.add_argument('--batch_size', '-bz', type=int, default=1)  # 攻击时 batch设置为1
    parse.add_argument('--number_work', '-nw', type=int, default=0)
    parse.add_argument('--number_class', '-nc', type=int, default=2)
    parse.add_argument('--eps', '-eps', type=float, default=16/255, help='最大扰动')   # todo

    parse.add_argument('--Vgg_weight_path', '-vwp', type=str, default='./weights/all_FF++_Vgg.pth')
    parse.add_argument('--Resnet_weight_path', '-rwp', type=str, default='./weights/all_FF++_Resnet.pth')
    parse.add_argument('--Resnet152_weight_path', type=str, default='./weights/all_FF++_Resnet152.pth')
    parse.add_argument('--Mesonet_weight_path', '-mwp', type=str, default='./weights/all_FF++_Mesonet.pth')
    parse.add_argument('--MesoInception_weight_path', '-miwp', type=str, default='./weights/all_FF++_MesoInception.pth')
    parse.add_argument('--EfficientNetb4_weight_path', '-ewpb4', type=str, default='./weights/all_FF++_EfficientNetb4.pth')
    parse.add_argument('--Xception_weight_path', '-xwp', type=str, default='./weights/all_FF++_Xception.pth')
    parse.add_argument('--Densenet121_weight_path', '-denwp', type=str, default='./weights/all_FF++_Densenet121.pth')
    parse.add_argument('--Densenet169_weight_path', '-den169wp', type=str, default='./weights/all_FF++_Densenet169.pth')
    parse.add_argument('--MobilenetV2_weight_path', '-mobiwp', type=str, default='./weights/all_FF++_MobilenetV2.pth')
    parse.add_argument('--Maxvit_weight_path', '-maxvitwp', type=str, default='./weights/all_FF++_Maxvit.pth')
    parse.add_argument('--GramNet_weight_path', '-gramwp', type=str, default='./weights/all_FF++_GramNet.pth')
    #  BIM MIFGSM TIFGSM DIFGSM DI2FGSM  SINIFGSM  AdvGan SSA  SSAH C_GSP CDA  MyAttack
    parse.add_argument('--Attack_method', '-amt', type=str, default='C_GSP', help='选择攻击方式')  # todo
    parse.add_argument('--Generator_weight_path', '-gwp', type=str, default='./other_attack_weights/')
    parse.add_argument('--MyGenerator_weight_path', '-mygwp', type=str, default='./attack_weights/')
    parse.add_argument('--save_attack_example_path', '-sp', type=str, default=f'./adv-example/AdvGAN/')

    parse.add_argument('--device', '-d', type=str, default='cuda:3')
    torch.cuda.set_device('cuda:3')
    main()

