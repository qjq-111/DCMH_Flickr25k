# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import utils
import model
import config as conf
import torch.optim as optim
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class FlickrDataset(Dataset):
    def __init__(self, img_list_path, label_path, tag_path, transform=None):
        with open(img_list_path, 'r') as f:
            self.image_list = [line[:-1] for line in f]
        self.labels = np.load(label_path)
        self.tags = np.load(tag_path)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = torch.from_numpy(self.labels[idx]).float()
        tag = torch.from_numpy(self.tags[idx]).float()

        if self.transform:
            img = self.transform(img)
        # vgg-f时才用到
        img = img * 255.0
        return img, tag, label, idx

    def __len__(self):
        return len(self.labels)


def create_model(model_path, use_gpu):
    img_net = model.ImageNet(conf.bit, model_path, 0.5, 24)
    txt_net = model.TextNet(conf.y_dim, conf.bit)
    if use_gpu:
        img_net = img_net.cuda()
        txt_net = txt_net.cuda()
    return img_net, txt_net


def data_loader():
    transformations_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformations_q = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # mean and std of ImageNet
    ])

    # 路径（包括测试集，训练集，数据库集）
    img_q_path = 'MyFlickr25K_original/query/image_list.txt'
    label_q_path = 'MyFlickr25K_original/query/label.npy'
    tag_q_path = 'MyFlickr25K_original/query/tag.npy'

    img_t_path = 'MyFlickr25K_original/train/image_list.txt'
    label_t_path = 'MyFlickr25K_original/train/label.npy'
    tag_t_path = 'MyFlickr25K_original/train/tag.npy'

    img_d_path = 'MyFlickr25K_original/database/image_list.txt'
    label_d_path = 'MyFlickr25K_original/database/label.npy'
    tag_d_path = 'MyFlickr25K_original/database/tag.npy'
    test_loader = DataLoader(FlickrDataset(img_q_path, label_q_path, tag_q_path, transform=transformations_t),
                             batch_size=conf.batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(FlickrDataset(img_t_path, label_t_path, tag_t_path, transform=transformations_q),
                              batch_size=conf.batch_size, shuffle=True, num_workers=4)
    database_loader = DataLoader(FlickrDataset(img_d_path, label_d_path, tag_d_path, transform=transformations_q),
                                 batch_size=conf.batch_size, shuffle=False, num_workers=4)
    return test_loader, train_loader, database_loader


def train(test_loader, train_loader, database_loader, model_path, use_gpu):
    # F和G矩阵
    F = torch.randn(conf.num_train, conf.bit)
    G = torch.randn(conf.num_train, conf.bit)
    img_net, txt_net = create_model(model_path, use_gpu)

    # 获取训练数据集的label
    train_labelpath = 'MyFlickr25K_original/train/label.npy'
    trainL = np.load(train_labelpath)
    trainL = torch.from_numpy(trainL).float()
    if use_gpu:
        F, G = F.cuda(), G.cuda()
        trainL = trainL.cuda()

    # B矩阵
    B = torch.sign(F + G)

    lr = conf.lr
    optimizer_img = optim.SGD(img_net.parameters(), lr=lr)
    optimizer_txt = optim.SGD(txt_net.parameters(), lr=lr)
    learning_rate = np.linspace(conf.lr, np.power(10, -6.), conf.max_epoch + 1)
    # result = {'loss': []}

    max_mapi2t = max_mapt2i = 0.

    for epoch in range(conf.max_epoch):
        img_net.train()
        txt_net.train()
        hash_loss_i, hash_loss_t = 0.0, 0.0
        for img, tag, label, batch_ind in train_loader:
            # train image net
            optimizer_img.zero_grad()
            batchsize = len(img)
            ones = torch.ones(batchsize, 1)
            ones_ = torch.ones(conf.num_train - batchsize, 1)
            unupdated_ind = np.setdiff1d(range(conf.num_train), np.array(batch_ind))
            if use_gpu:
                img, label, tag = img.cuda(), label.cuda(), tag.cuda()
                ones = ones.cuda()
                ones_ = ones_.cuda()
            S = (label.mm(trainL.t()).long() > 0).float()

            cur_f = img_net(img)
            for i, ind in enumerate(batch_ind):
                F[ind, :] = cur_f.data[i]
            theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
            logloss_x = -torch.sum(S * theta_x - utils.Logtrick(theta_x, use_gpu))
            quantization_x = torch.sum(torch.pow(B[batch_ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))

            # 增加同模态内内相似度的损失
            theta_x_intra = 1.0 / 2 * torch.matmul(cur_f, F.t())
            logloss_x_intra = -torch.sum(S * theta_x_intra - utils.Logtrick(theta_x_intra, use_gpu))
            loss_x = logloss_x + conf.gamma * quantization_x + conf.eta * balance_x + logloss_x_intra
            loss_x /= (batchsize * conf.num_train)

            loss_x.backward()
            optimizer_img.step()
            hash_loss_i += float(loss_x.item())

            # train txt net
            optimizer_txt.zero_grad()

            tag = tag.unsqueeze(1).unsqueeze(-1).type(torch.float)
            cur_g = txt_net(tag)
            for i, ind in enumerate(batch_ind):
                G[ind, :] = cur_g.data[i]
            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(S * theta_y - utils.Logtrick(theta_y, use_gpu))
            quantization_y = torch.sum(torch.pow(B[batch_ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))

            # 增加模态内相似度损失
            theta_y_intra = 1.0 / 2 * torch.matmul(cur_g, G.t())
            logloss_y_intra = -torch.sum(S * theta_y_intra - utils.Logtrick(theta_y_intra, use_gpu))

            loss_y = logloss_y + conf.gamma * quantization_y + conf.eta * balance_y + logloss_y_intra
            loss_y /= (batchsize * conf.num_train)
            loss_y.backward()
            optimizer_txt.step()
            hash_loss_t += float(loss_y.item())

        # Update B
        B = torch.sign(F + G)

        img_net.eval()
        txt_net.eval()
        num = len(train_loader)
        print('...epoch: %3d, img_loss: %3.3f, txt_loss: %3.3f' % (epoch + 1, hash_loss_i / num, hash_loss_t / num))

        if conf.valid:
            mapi2t, mapt2i = valid(img_net, txt_net, test_loader, database_loader, use_gpu)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
            if mapi2t + mapt2i > max_mapi2t + max_mapt2i:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                img_net.save(img_net.module_name + '.pth')
                txt_net.save(txt_net.module_name + '.pth')
        # set learning rate
        lr = learning_rate[epoch + 1]
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr


def valid(img_net, txt_net, test_loader, database_loader, use_gpu):
    qBX, qBY, query_L = utils.generate_hash_code(test_loader, img_net, txt_net, use_gpu)
    rBX, rBY, retrieval_L = utils.generate_hash_code(database_loader, img_net, txt_net, use_gpu)
    mapi2t = utils.Calculate_mAP(query_L, qBX, retrieval_L, rBY)
    mapt2i = utils.Calculate_mAP(query_L, qBY, retrieval_L, rBX)
    return mapi2t, mapt2i


if __name__ == '__main__':
    torch.cuda.init()
    test_loader, train_loader, database_loader = data_loader()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(3)
    train(test_loader, train_loader, database_loader, conf.pretrain_model_path, use_gpu)


