#!/usr/bin/env python36
# -*- coding: utf-8 -*-

######################################################
# Main Function of EA Mechanism. #
######################################################

import os
import argparse
import pickle
import time
from EAutils import Data, split_validation
from EAmodel import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cainiao',
                    help='dataset name: cainiao/encodeSample')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='temperature parameter used in NT_Xent loss')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=300, help='hidden state size')
parser.add_argument('--position_embeddingSize', type=int, default=300, help='embedding size of position')
parser.add_argument('--nhead', type=int, default=4, help='the number of heads of multi-head attention')
parser.add_argument('--layer', type=int, default=1, help='number of SAN layers')
parser.add_argument('--feedforward', type=int, default=4, help='the multipler of hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--TA', default=False, help='use target-aware or not')
parser.add_argument('--split_num', type=int, default=6, help='the ratio of dividing long-tail items and short-head items')
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--gpu', default='0', help='which gpu to use')

opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def main():
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    if opt.dataset == 'galanz':
        n_node = 43098
        len_max = 20  # take the last 10 items of the session as in NISER
    elif opt.dataset == 'cainiao':
        n_node = 37484
        len_max = 20

    model = trans_to_cuda(TAR_DNN(opt, n_node, len_max))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1

        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1

        print('Best Result:')
        print('\tRMAE:\t%.4f\tRRSE:\t%.4f\tEpoch:\t%d,\t%d' % (
        best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
