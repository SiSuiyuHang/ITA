import argparse
import os
import random
import numpy as np
import torch
from torchvision import transforms, datasets, models
from attack_utils import *
from data_utils import *
from nets.Xception import *
from nets.Mesonet4 import *
from nets.MesoInception import *
from other_attack_utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def main():

    #set_seed(123)
    args =parse.parse_args()
    name = args.name
    print(f'name:{name}')

    device = args.device
    print(f'device:{device}')
    number_class = args.number_class
    print(f'number_class:{number_class}')
    epsilon = args.epsilon
    print(f'epsilon:{epsilon}')
    epochs = args.epochs
    print(f'epochs:{epochs}')

    train_path = args.train_path
    print(f'train_path:{train_path}')
    batch_size = args.batch_size
    nw = args.number_work
    save_attack_weight_path = args.save_attack_weight_path
    print(f'save_attack_weight_path:{save_attack_weight_path}')


    if name == 'Resnet':
        Resnet_weight_path = args.Resnet_weight_path
        model = models.resnet50(num_classes=2).to(device)
        model.load_state_dict(torch.load(Resnet_weight_path, map_location=device))
        print(f'base_model_weight_path:{Resnet_weight_path}')
    elif name == 'Densenet169':
        Densenet169_weight_path = args.Densenet169_weight_path
        model = models.densenet169(weights=None, num_classes=number_class).to(device)
        model.load_state_dict(torch.load(Densenet169_weight_path, map_location=device))
    elif name == 'Densenet121':
        Densenet121_weight_path = args.Densenet121_weight_path
        model = models.densenet121(weights=None, num_classes=number_class).to(device)
        model.load_state_dict(torch.load(Densenet121_weight_path, map_location=device))

    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)


    model.eval()  # todo 做攻击的时候一定要加 base_model

    #  todo Advgan
    # advgan = AdvGAN_Attack(name, device, model, number_class, epsilon, epochs, save_attack_weight_path)
    # advgan.train(train_loader)

    # todo C_GSP Attack
    # c_gsp = C_GSP(device, model, name, number_class, epsilon, epochs, save_attack_weight_path)
    # c_gsp.train(train_loader, epochs)

    # todo CDA Attack
    # cda_attack = CDA(device, model, name, number_class, epsilon, save_attack_weight_path)
    # cda_attack.train(train_loader, epochs)

    # todo MyAttack
    cgattack = CondGeneratorResnet_Transfer_Attack(device, model, name, number_class, epsilon, save_attack_weight_path)
    cgattack.train(train_loader, epochs)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='Resnet')
    #  Resnet Densenet169 Densenet121
    # todo  数据集 0为假 1为真
    parse.add_argument('--train_path', '-tp', type=str, default='../datasets/FF++_dataset_by_ls/jpg/c23/total/train/')

    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--number_work', '-nw', type=int, default=2)
    parse.add_argument('--number_class', '-nc', type=int, default=2)
    parse.add_argument('--epsilon', '-eps', type=float, default=16/255)  # todo
    parse.add_argument('--epochs', '-iter', type=int, default=20)

    parse.add_argument('--Resnet_weight_path', '-rwp', type=str, default='./weights/all_FF++_Resnet.pth')
    parse.add_argument('--Densenet121_weight_path', '-den121wp', type=str, default='./weights/all_FF++_Densenet121.pth')
    parse.add_argument('--Densenet169_weight_path', '-den169wp', type=str, default='./weights/all_FF++_Densenet169.pth')
    #parse.add_argument('--save_attack_weight_path', '-sp', type=str, default='./other_attack_weights/')
    parse.add_argument('--save_attack_weight_path', '-sp', type=str, default='./attack_weights/')     # todo

    # todo 记得修改权重保存路径  other_attack_weights  attack_weights
    parse.add_argument('--device', '-d', type=str, default='cuda:2')   # todo
    torch.cuda.set_device('cuda:2')
    main()

