import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import scipy.integrate as integrate
from time import gmtime, strftime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import argparse
import random
import glob
import copy
import sys
from tqdm import tqdm

from dataset import FeatDataLayer, LoadDataset, LoadDataset_NAB
from models import _netD, _netG, _param

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help='dataset to be used: CUB/NAB', default='CUB')
parser.add_argument('--splitmode', type=str, help='the way to split train/test data: easy/hard', default='easy')
parser.add_argument('--model_number', type=int, help='Model-Number: 1 for KL, 2 for Sharma-Entropy, 3 for Bachatera,'
                                                     '4 for Tsallis, 5 for Renyi, 6 K+1 Classification', default=2)
parser.add_argument('--exp_name', default='Reproduce', type=str, help='Experiment Name')
parser.add_argument('--main_dir', default='./', type=str,
                    help='Main Directory including data folder')

parser.add_argument('--creativity_weight', type=float, default=0.1, help='Weight of CIZSL loss- '
                                                                         'Varies by Dataset & SplitMode- '
                                                                         'Best values are in main function - '
                                                                         'Can be obtained by running cross-validation')
parser.add_argument('--validate', default=0, type=int, help='1 to validate and find best creativity weight, '
                                                             'otherwise use --creativity_weight')

parser.add_argument('--SM_Alpha', default='0.5', type=float, help='alpha weight of SM divergence')
parser.add_argument('--SM_Beta', default='0.9999', type=float, help='beta weight of SM divergence')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume', type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=200)
parser.add_argument('--evl_interval', type=int, default=1)

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter for training """
opt.GP_LAMBDA = 10  # Gradient penalty lambda
opt.CENT_LAMBDA = 1
opt.REG_W_LAMBDA = 0.001
opt.REG_Wz_LAMBDA = 0.0001

opt.lr = 0.0001
opt.batchsize = 1000

""" hyper-parameter for testing"""
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20  # knn: the value of K
max_accuracy = -1

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

main_dir = opt.main_dir

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class Scale(nn.Module):
    def __init__(self, num_scales):
        super(Scale, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(1, num_scales, bias=False))
        self.layer_module = ListModule(*self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

def train(creative_weight=1000, model_num=1, is_val=True):
    param = _param()
    if opt.dataset == 'CUB':
        dataset = LoadDataset(opt, main_dir, is_val)
        exp_info = 'CUB_EASY' if opt.splitmode == 'easy' else 'CUB_HARD'
    elif opt.dataset == 'NAB':
        dataset = LoadDataset_NAB(opt, main_dir, is_val)
        exp_info = 'NAB_EASY' if opt.splitmode == 'easy' else 'NAB_HARD'
    else:
        print('No Dataset with that name')
        sys.exit(0)
    param.X_dim = dataset.feature_dim
    opt.Creative_weight = creative_weight

    data_layer = FeatDataLayer(dataset.labels_train, dataset.pfc_feat_data_train, opt)
    result = Result()

    ones = Variable(torch.Tensor(1, 1))
    ones.data.fill_(1.0)

    netG = _netG(dataset.text_dim, dataset.feature_dim).cuda()
    netG.apply(weights_init)
    if model_num == 6:
        netD = _netD(dataset.train_cls_num + 1, dataset.feature_dim).cuda()
    else:
        netD = _netD(dataset.train_cls_num, dataset.feature_dim).cuda()
    netD.apply(weights_init)

    if model_num == 2:
        log_SM_ab = Scale(2)
        log_SM_ab = nn.DataParallel(log_SM_ab).cuda()
    elif model_num == 4 or model_num == 5:
        log_SM_ab = Scale(1)
        log_SM_ab = nn.DataParallel(log_SM_ab).cuda()

    exp_params = 'Model_{}_CAN{}_Eu{}_Rls{}_RWz{}_{}'.format(model_num, opt.Creative_weight, opt.CENT_LAMBDA,
                                                             opt.REG_W_LAMBDA, opt.REG_Wz_LAMBDA, opt.exp_name)

    out_subdir = main_dir + 'out/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists(out_subdir):
        os.makedirs(out_subdir)

    log_dir = out_subdir + '/log_{:s}.txt'.format(exp_info)
    with open(log_dir, 'a') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            netD.load_state_dict(checkpoint['state_dict_D'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if model_num == 2 or model_num == 4 or model_num == 5:
        nets = [netG, netD, log_SM_ab]
    else:
        nets = [netG, netD]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    if model_num == 2 or model_num == 4 or model_num == 5:
        optimizer_SM_ab = optim.Adam(log_SM_ab.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for it in tqdm(range(start_step, 3000 + 1)):
        # Creative Loss
        blobs = data_layer.forward()
        labels = blobs['labels'].astype(int)
        new_class_labels = Variable(
            torch.from_numpy(np.ones_like(labels) * dataset.train_cls_num)).cuda()
        text_feat_1 = np.array([dataset.train_text_feature[i, :] for i in labels])
        text_feat_2 = np.array([dataset.train_text_feature[i, :] for i in labels])
        np.random.shuffle(text_feat_1)  # Shuffle both features to guarantee different permutations
        np.random.shuffle(text_feat_2)
        alpha = (np.random.random(len(labels)) * (.8 - .2)) + .2

        text_feat_mean = np.multiply(alpha, text_feat_1.transpose())
        text_feat_mean += np.multiply(1. - alpha, text_feat_2.transpose())
        text_feat_mean = text_feat_mean.transpose()
        text_feat_mean = normalize(text_feat_mean, norm='l2', axis=1)
        text_feat_Creative = Variable(torch.from_numpy(text_feat_mean.astype('float32'))).cuda()
        z_creative = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()
        G_creative_sample = netG(z_creative, text_feat_Creative)

        """ Discriminator """
        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels

            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            # GAN's D loss
            D_real, C_real = netD(X)
            D_loss_real = torch.mean(D_real)
            C_loss_real = F.cross_entropy(C_real, y_true)
            DC_loss = -D_loss_real + C_loss_real
            DC_loss.backward()

            # GAN's D loss
            G_sample = netG(z, text_feat).detach()
            D_fake, C_fake = netD(G_sample)
            D_loss_fake = torch.mean(D_fake)
            C_loss_fake = F.cross_entropy(C_fake, y_true)

            DC_loss = D_loss_fake + C_loss_fake
            DC_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward()

            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()
            reset_grad(nets)

        """ Generator """
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()

            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            G_sample = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            _, C_real = netD(X)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # Auxiliary classification loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2

            GC_loss = -G_loss + C_loss

            # Centroid loss
            Euclidean_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        Euclidean_loss += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
                Euclidean_loss *= 1.0 / dataset.train_cls_num * opt.CENT_LAMBDA

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)

            # ||W_z||21 regularization, make W_z sparse
            reg_Wz_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_Wz_LAMBDA != 0:
                Wz = netG.rdc_text.weight
                reg_Wz_loss = Wz.pow(2).sum(dim=0).sqrt().sum().mul(opt.REG_Wz_LAMBDA)

            # D(C| GX_fake)) + Classify GX_fake as real
            D_creative_fake, C_creative_fake = netD(G_creative_sample)
            if model_num == 1:  # KL Divergence
                G_fake_C = F.log_softmax(C_creative_fake)
            else:
                G_fake_C = F.softmax(C_creative_fake)

            if model_num == 1:  # KL Divergence
                entropy_GX_fake = (G_fake_C / G_fake_C.data.size(1)).mean()
            elif model_num == 2:  # SM Divergence
                q_shape = Variable(torch.FloatTensor(G_fake_C.data.size(0), G_fake_C.data.size(1))).cuda()
                q_shape.data.fill_(1.0 / G_fake_C.data.size(1))

                SM_ab = F.sigmoid(log_SM_ab(ones))
                SM_a = 0.2 + torch.div(SM_ab[0][0], 1.6666666666666667).cuda()
                SM_b = 0.2 + torch.div(SM_ab[0][1], 1.6666666666666667).cuda()
                pow_a_b = torch.div(1 - SM_a, 1 - SM_b)
                alpha_term = (torch.pow(G_fake_C + 1e-5, SM_a) * torch.pow(q_shape, 1 - SM_a)).sum(1)
                entropy_GX_fake_vec = torch.div(torch.pow(alpha_term, pow_a_b) - 1, SM_b - 1)
            elif model_num == 3:  # Bachatera Divergence
                q_shape = Variable(torch.FloatTensor(G_fake_C.data.size(0), G_fake_C.data.size(1))).cuda()
                q_shape.data.fill_(1.0 / G_fake_C.data.size(1))
                SM_a = Variable(torch.FloatTensor(1, 1)).cuda()
                SM_a.data.fill_(opt.SM_Alpha)
                SM_b = Variable(torch.FloatTensor(1, 1)).cuda()
                SM_b.data.fill_(opt.SM_Alpha)
                pow_a_b = torch.div(1 - SM_a, 1 - SM_b)
                alpha_term = (torch.pow(G_fake_C + 1e-5, SM_a) * torch.pow(q_shape, 1 - SM_a)).sum(1)
                entropy_GX_fake_vec = -torch.div(torch.pow(alpha_term, pow_a_b) - 1, SM_b - 1)
            elif model_num == 4:  # Tsallis Divergence
                q_shape = Variable(torch.FloatTensor(G_fake_C.data.size(0), G_fake_C.data.size(1))).cuda()
                q_shape.data.fill_(1.0 / G_fake_C.data.size(1))

                SM_ab = F.sigmoid(log_SM_ab(ones))
                SM_a = 0.2 + torch.div(SM_ab[0][0], 1.6666666666666667).cuda()
                SM_b = SM_a
                pow_a_b = torch.div(1 - SM_a, 1 - SM_b)
                alpha_term = (torch.pow(G_fake_C + 1e-5, SM_a) * torch.pow(q_shape, 1 - SM_a)).sum(1)
                entropy_GX_fake_vec = -torch.div(torch.pow(alpha_term, pow_a_b) - 1, SM_b - 1)
            elif model_num == 5:  # Renyi Divergence
                q_shape = Variable(torch.FloatTensor(G_fake_C.data.size(0), G_fake_C.data.size(1))).cuda()
                q_shape.data.fill_(1.0 / G_fake_C.data.size(1))

                SM_ab = F.sigmoid(log_SM_ab(ones))
                SM_a = 0.2 + torch.div(SM_ab[0][0], 1.6666666666666667).cuda()
                SM_b = Variable(torch.FloatTensor(1, 1)).cuda()
                SM_b.data.fill_(opt.SM_Beta)
                pow_a_b = torch.div(1 - SM_a, 1 - SM_b)
                alpha_term = (torch.pow(G_fake_C + 1e-5, SM_a) * torch.pow(q_shape, 1 - SM_a)).sum(1)
                entropy_GX_fake_vec = -torch.div(torch.pow(alpha_term, pow_a_b) - 1, SM_b - 1)

            if model_num == 6:
                loss_creative = F.cross_entropy(C_creative_fake, new_class_labels)
            else:
                if model_num != 1:
                    # Normalize SM-Divergence & Report mean
                    min_e, max_e = torch.min(entropy_GX_fake_vec), torch.max(entropy_GX_fake_vec)
                    entropy_GX_fake_vec = (entropy_GX_fake_vec - min_e) / (max_e - min_e)
                    entropy_GX_fake = -entropy_GX_fake_vec.mean()
                loss_creative = -opt.Creative_weight * entropy_GX_fake

            disc_GX_fake_real = -torch.mean(D_creative_fake)
            total_loss_creative = loss_creative + disc_GX_fake_real

            all_loss = GC_loss + Euclidean_loss + reg_loss + reg_Wz_loss + total_loss_creative
            all_loss.backward()
            if model_num == 2 or model_num == 4 or model_num == 5:
                optimizer_SM_ab.step()
            optimizerG.step()
            reset_grad(nets)

        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(
                y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(
                y_true.data.size()[0])

            log_text =  'Iter-{}; rl: {:.4}%; fk: {:.4}%'.format(it, acc_real * 100, acc_fake * 100)
            with open(log_dir, 'a') as f:
                f.write(log_text + '\n')

        if it % opt.evl_interval == 0:
            netG.eval()
            cur_acc = eval_fakefeat_test(it, netG, dataset, param, result)
            cur_auc = eval_fakefeat_GZSL(netG, dataset, param, out_subdir, result)

            if cur_auc > result.best_auc:
                result.best_auc = cur_auc
                result.best_acc = cur_acc

                if it % opt.save_interval:
                    files2remove = glob.glob(out_subdir + '/Best_model*')
                    for _i in files2remove:
                        os.remove(_i)
                    torch.save({
                        'it': it + 1,
                        'state_dict_G': netG.state_dict(),
                        'state_dict_D': netD.state_dict(),
                        'random_seed': opt.manualSeed,
                        'log': log_text,
                    }, out_subdir + '/Best_model_AUC_{:.2f}.tar'.format(cur_auc))

            netG.train()
    return result


def eval_fakefeat_GZSL(netG, dataset, param, plot_dir, result):
    gen_feat = np.zeros([0, param.X_dim])
    for i in range(dataset.train_cls_num):
        text_feat = np.tile(dataset.train_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    visual_pivots = [gen_feat[i * opt.nSample:(i + 1) * opt.nSample].mean(0) \
                     for i in range(dataset.train_cls_num + dataset.test_cls_num)]
    visual_pivots = np.vstack(visual_pivots)

    """collect points for gzsl curve"""

    acc_S_T_list, acc_U_T_list = list(), list()
    seen_sim = cosine_similarity(dataset.pfc_feat_data_train, visual_pivots)
    unseen_sim = cosine_similarity(dataset.pfc_feat_data_test, visual_pivots)
    for GZSL_lambda in np.arange(-2, 2, 0.01):
        tmp_seen_sim = copy.deepcopy(seen_sim)
        tmp_seen_sim[:, dataset.train_cls_num:] += GZSL_lambda
        pred_lbl = np.argmax(tmp_seen_sim, axis=1)
        acc_S_T_list.append((pred_lbl == np.asarray(dataset.labels_train)).mean())

        tmp_unseen_sim = copy.deepcopy(unseen_sim)
        tmp_unseen_sim[:, dataset.train_cls_num:] += GZSL_lambda
        pred_lbl = np.argmax(tmp_unseen_sim, axis=1)
        acc_U_T_list.append((pred_lbl == (np.asarray(dataset.labels_test) + dataset.train_cls_num)).mean())

    auc_score = integrate.trapz(y=acc_S_T_list, x=acc_U_T_list) * 100.0
    plt.plot(acc_S_T_list, acc_U_T_list)
    plt.title("{:s}-{:s}-{}: {:.4}%".format(opt.dataset, opt.splitmode, opt.model_number, auc_score))
    plt.savefig(plot_dir + '/best_plot.png')
    plt.clf()
    plt.close()
    np.savetxt(plot_dir + '/best_plot.txt', np.vstack([acc_S_T_list, acc_U_T_list]))
    result.auc_list += [auc_score]
    return auc_score


def eval_fakefeat_test(it, netG, dataset, param, result):
    gen_feat = np.zeros([0, param.X_dim])
    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    # cosince predict K-nearest Neighbor
    sim = cosine_similarity(dataset.pfc_feat_data_test, gen_feat)
    idx_mat = np.argsort(-1 * sim, axis=1)
    label_mat = (idx_mat[:, 0:opt.Knn] / opt.nSample).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]

    # produce acc
    label_T = np.asarray(dataset.labels_test)
    acc = (preds == label_T).mean() * 100

    result.acc_list += [acc]
    return acc


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_auc = 0.0
        self.best_iter = 0.0
        self.acc_list = []
        self.auc_list = []


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.GP_LAMBDA
    return gradient_penalty


def return_best_creativity_weight_validation(creative_weights=[0.0001, 0.001, 0.1, 1, 10, 100, 1000]):
    # Validation
    max_acc, best_w = -1, 1
    for cr_w in creative_weights:
        print("{:s} - {:s}".format(opt.dataset, opt.splitmode))
        print("{} - {}".format(cr_w, opt.model_number))
        print('*' * 10)
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed_all(opt.manualSeed)
        result = train(cr_w, opt.model_number, is_val=False)
        if result.best_auc > max_acc:
            max_acc = result.best_auc
            max_res = result
            best_w = cr_w

    return best_w, max_res


if __name__ == "__main__":
    # Inference
    """
    Values of Cross-validation CIZSL loss weight (Sharma-Entropy)
    CUB-EASY: 0.0001
    CUB-HARD: 0.1
    NAB-EASY: 1
    NAB-HARD: 0.1
    """
    cr_weight = opt.creativity_weight
    if opt.validate == 1:
        cr_weight, _ = return_best_creativity_weight_validation()
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    result = train(cr_weight, opt.model_number, is_val=False)
    print('=' * 15)
    print('=' * 15)
    print(opt.exp_name, opt.dataset, opt.splitmode)
    print("Accuracy is {:.4}%, and Generalized AUC is {:.4}%".format(result.best_acc, result.best_auc))
