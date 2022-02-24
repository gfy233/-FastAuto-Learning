# -*- coding: utf-8 -*-
"""

Controllers for DP models

@author: hongyuan
"""

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
import DpModels
import DpOptimizers

#from scipy.optimize import minimize

dtype = theano.config.floatX


class ControlDotProcess(object):
    # This is a seq 2 seq model train_er
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_end : T * size_batch -- T - t_i
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
        self.seq_time_to_end = tensor.matrix(
            dtype=dtype, name='seq_time_to_end'
        )
        self.seq_time_to_current = tensor.tensor3(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_mask_to_current = tensor.tensor3(
            dtype=dtype, name='seq_mask_to_current'
        )
        #
        self.seq_sims_time_to_current = tensor.tensor3(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        self.seq_sims_mask_to_current = tensor.tensor3(
            dtype=dtype, name='seq_sims_mask_to_current'
        )
        #
        #
        self.hawkes_ctsm = DpModels.DotProcess(settings)
        #
        self.hawkes_ctsm.compute_loss(
            self.seq_time_to_end,
            self.seq_time_to_current,
            self.seq_type_event,
            self.time_since_start_to_end,
            self.seq_mask,
            self.seq_mask_to_current
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(
                adam_params=None
            )
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(
                adam_params=None
            )
        else:
            print "Choose a optimizer ! "
        #
        if 'learn_rate' in settings:
            print "learn rate is set to : ", settings['learn_rate']
            self.adam_optimizer.set_learn_rate(
                settings['learn_rate']
            )
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = range(3)
        )
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.time_since_start_to_end,
                self.seq_mask,
                self.seq_mask_to_current
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ],
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.time_since_start_to_end,
                self.seq_mask,
                self.seq_mask_to_current
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ]
        )
        if settings['predict_lambda']:
            print "compiling dev function for intensity computation ... "
            self.hawkes_ctsm.compute_lambda(
                self.seq_type_event,
                self.seq_sims_time_to_current,
                self.seq_sims_mask,
                self.seq_sims_mask_to_current
            )
            self.model_dev_lambda = theano.function(
                inputs = [
                    self.seq_type_event,
                    self.seq_sims_time_to_current,
                    self.seq_sims_mask,
                    self.seq_sims_mask_to_current
                ],
                outputs = [
                    self.hawkes_ctsm.lambda_samples,
                    self.hawkes_ctsm.num_of_samples
                ]
            )
        #
        #self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #