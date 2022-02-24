# -*- coding: utf-8 -*-

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
#import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import DpUtils

dtype=theano.config.floatX

#
class DotProcess(object):
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        print "initializing DotProcess ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            # initialize variables
            self.mu = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='mu'
            )
            self.alpha = theano.shared(
                numpy.ones(
                    (self.dim_process, self.dim_process),
                    dtype=dtype
                ), name='alpha'
            )
            self.delta = theano.shared(
                numpy.ones(
                    (self.dim_process, self.dim_process),
                    dtype=dtype
                ), name='delta'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.mu = theano.shared(
                model_pre_train['mu'], name='mu'
            )
            self.alpha = theano.shared(
                model_pre_train['alpha'], name='alpha'
            )
            self.delta = theano.shared(
                model_pre_train['delta'], name='delta'
            )
        #
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            self.mu, self.alpha, self.delta
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        # to evaluate per-event intensity predict
        # this should be filterd by mask
        self.lambda_samples = None
        self.num_of_samples = None
        #
    #
    #
    def compute_loss(
        self,
        seq_time_to_end, seq_time_to_current, seq_type_event,
        time_since_start_to_end,
        seq_mask, seq_mask_to_current
    ):
        '''
        use this function to compute negative log likelihood
        seq_time_to_end : T * size_batch -- T-t_i
        seq_time_to_current : T * T * size_batch --
        for each batch, it is T * T, and at each time step t,
        it tracks the ( t_i - t_i' ) for all t_i' < t_i
        seq_type_event : T * size_batch -- for each data
        and each time step, tracks the type of event k_i
        time_since_start_to_end : size_batch -- time for seq
        #
        seq_mask : T * size_batch -- 1/0
        seq_mask_to_current : T * T * size_batch -- 1/0
        '''
        print "computing loss function of Hawkes model ... "
        # first compute the 3rd term in loss
        alpha_over_seq = self.alpha[
            :, seq_type_event
        ] # dim_process * T * size_batch
        delta_over_seq = self.delta[
            :, seq_type_event
        ] # dim_process * T * size_batch
        #
        term_3 = tensor.sum(
            tensor.sum(
                (
                    (
                        numpy.float32(1.0) - tensor.exp(
                            -delta_over_seq * seq_time_to_end[
                                None, :, :
                            ]
                        )
                    ) * alpha_over_seq / delta_over_seq
                ),
                axis = 0
            ) * seq_mask,
            axis = 0
        ) # (size_batch, )
        # then we compute the 2nd term
        term_2 = tensor.sum(self.mu) * time_since_start_to_end
        lambda_over_seq = self.mu[:, None, None] + tensor.sum(
            (
                seq_mask_to_current[None,:,:,:]
                * (
                    alpha_over_seq[:,None,:,:] * tensor.exp(
                        -delta_over_seq[:,None,:,:]
                        * seq_time_to_current[None,:,:,:]
                    )
                )
            )
            , axis=2
        ) # dim_process * T * size_batch
        #
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=0
        ) # T * size_batch
        #
        # now we choose the right lambda for each step
        # by using seq_type_event : T * size_batch
        new_shape_0 = lambda_over_seq.shape[1]*lambda_over_seq.shape[2]
        new_shape_1 = lambda_over_seq.shape[0]
        #
        back_shape_0 = lambda_over_seq.shape[1]
        back_shape_1 = lambda_over_seq.shape[2]
        #
        lambda_target_over_seq = lambda_over_seq.transpose(
            (1,2,0)
        ).reshape(
            (
                new_shape_0, new_shape_1
            )
        )[
            tensor.arange(new_shape_0),
            seq_type_event.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
    #
    #
    def compute_lambda(
        self,
        seq_type_event,
        seq_sims_time_to_current,
        seq_sims_mask,
        seq_sims_mask_to_current
    ):
        '''
        use this function to compute intensity
        seq_type_event : T * size_batch -- for each data
        and each time step, tracks the type of event k_i
        these are only used for computing intensity estimation
        N is the # of MonteCarlo samples
        seq_sims_time_to_current : N * T * size_batch -- for each batch, and at each time step t, track t_i-t_i' for t_i'<t_i
        seq_sims_mask : N * size_batch
        seq_sims_mask_to_current : N * T * size_batch
        '''
        print "computing intensity ... "
        # first compute the 3rd term in loss
        alpha_over_seq = self.alpha[
            :, seq_type_event
        ] # dim_process * T * size_batch
        delta_over_seq = self.delta[
            :, seq_type_event
        ] # dim_process * T * size_batch
        #
        '''
        in this block, we compute intensity
        at sampled time
        '''
        #
        lambda_samples = self.mu[:,None,None] + tensor.sum(
            (
                seq_sims_mask_to_current[None,:,:,:] * (
                    alpha_over_seq[:,None,:,:] * tensor.exp(
                        -delta_over_seq[:,None,:,:] * seq_sims_time_to_current[None,:,:,:]
                    )
                )
            ), axis=2
        )
        # K * N * size_batch
        self.lambda_samples = lambda_samples * seq_sims_mask[None,:,:]
        self.num_of_samples = tensor.sum(seq_sims_mask)
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
#
#
# Note : _scale means : we use scaling parameter in transfer function
#