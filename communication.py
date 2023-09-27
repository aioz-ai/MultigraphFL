import os
from abc import ABC, abstractmethod

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_network, get_iterator, get_model, args_to_string, EXTENSIONS, logger_write_params, print_model, get_multi_network
import time
from utils.metrics import accuracy
import copy

class Network(ABC):
    def __init__(self, args):
        """
        Abstract class representing a network of worker collaborating to train a machine learning model,
        each worker has a local model and a local data iterator.
         Should implement `mix` to precise how the communication is done
        :param args: parameters defining the network
        """
        self.args = args
        self.device = args.device
        self.batch_size_train = args.bz_train
        self.batch_size_test = args.bz_test
        if args.multigraph:
            self.network = get_multi_network(args.network_name, args.architecture, args.experiment)
            self.n_workers = self.network[0].number_of_nodes()
        else:
            self.network = get_network(args.network_name, args.architecture, args.experiment)
            self.n_workers = self.network.number_of_nodes()
        self.local_steps = args.local_steps
        self.log_freq = args.log_freq
        self.fit_by_epoch = args.fit_by_epoch
        self.initial_lr = args.lr
        self.optimizer_name = args.optimizer
        self.lr_scheduler_name = args.decay
        self.test_ensemble = args.test_ensemble

        # create logger
        if args.save_logg_path == "":
            self.logger_path = os.path.join("loggs", args_to_string(args), args.architecture)
        else:
            self.logger_path = args.save_logg_path
        os.makedirs(self.logger_path, exist_ok=True)
        if not args.test:
            self.logger_write_param = logger_write_params(os.path.join(self.logger_path, 'log.txt'))
        else:
            self.logger_write_param = logger_write_params(os.path.join(self.logger_path, 'test.txt'))
        self.logger_write_param.write(args.__repr__())

        self.logger_write_param.write('>>>>>>>>>> start time: ' + str(time.asctime()))
        self.time_start = time.time()
        self.time_start_update = self.time_start

        self.logger = SummaryWriter(self.logger_path)

        self.round_idx = 0  # index of the current communication round

        self.train_dir = os.path.join("data", args.experiment, args.network_name, "train")
        self.test_dir = os.path.join("data", args.experiment, args.network_name, "test")

        self.train_path = os.path.join(self.train_dir, "train" + EXTENSIONS[args.experiment])
        self.test_path = os.path.join(self.test_dir, "test" + EXTENSIONS[args.experiment])

        print('- Loading: > %s < dataset from: %s'%(args.experiment, self.train_path))
        self.train_iterator = get_iterator(args.experiment, self.train_path, self.device, self.batch_size_test, numworkers=5)
        print('- Loading: > %s < dataset from: %s'%(args.experiment, self.test_path))
        self.test_iterator = get_iterator(args.experiment, self.test_path, self.device, self.batch_size_test, numworkers=5)

        self.workers_iterators = []
        self.local_function_weights = np.zeros(self.n_workers)
        train_data_size = 0
        print('>>>>>>>>>> Loading worker-datasets')
        for worker_id in range(self.n_workers):
            data_path = os.path.join(self.train_dir, str(worker_id) + EXTENSIONS[args.experiment])
            print('\t + Loading: > %s < dataset from: %s' % (args.experiment, data_path))
            self.workers_iterators.append(get_iterator(args.experiment, data_path, self.device, self.batch_size_train, numworkers=0))
            train_data_size += len(self.workers_iterators[-1])
            self.local_function_weights[worker_id] = len(self.workers_iterators[-1].dataset)

        self.epoch_size = int(train_data_size / self.n_workers)
        self.local_function_weights = self.local_function_weights / self.local_function_weights.sum()

        # create workers models
        if args.use_weighted_average:
            self.workers_models = [get_model(args.experiment, self.device, self.workers_iterators[w_i],
                                             optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                                             initial_lr=self.initial_lr, epoch_size=self.epoch_size,
                                             coeff=self.local_function_weights[w_i], test_ensemble=self.test_ensemble)
                                   for w_i in range(self.n_workers)]
        else:
            self.workers_models = [get_model(args.experiment, self.device, self.workers_iterators[w_i],
                                             optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                                             initial_lr=self.initial_lr, epoch_size=self.epoch_size, test_ensemble=self.test_ensemble)
                                   for w_i in range(self.n_workers)]
        if self.args.multigraph:
            self.workers_models_temp = [copy.deepcopy(i.net) for i in self.workers_models]
        # average model of all workers
        self.global_model = get_model(args.experiment,
                                      self.device,
                                      self.train_iterator,
                                      epoch_size=self.epoch_size)
        print_model(self.global_model.net, self.logger_write_param)

        # write initial performance
        if not args.test:
            self.write_logs()

    @abstractmethod
    def mix(self):
        pass

    def write_logs(self):
        """
        write train/test loss, train/tet accuracy for average model and local models
         and intra-workers parameters variance (consensus) adn save average model
        """
        if (self.round_idx - 1) == 0:
            return None
        print('>>>>>>>>>> Evaluating')
        print('\t - train set')
        start_time = time.time()
        train_loss, train_acc, _, _ = self.global_model.evaluate_iterator(self.train_iterator)
        end_time_train = time.time()
        print('\t - test set')
        test_loss, test_acc, _, _ = self.global_model.evaluate_iterator(self.test_iterator)
        end_time_test = time.time()
        self.logger.add_scalar("Train/Loss", train_loss, self.round_idx)
        self.logger.add_scalar("Train/Acc", train_acc, self.round_idx)
        self.logger.add_scalar("Test/Loss", test_loss, self.round_idx)
        self.logger.add_scalar("Test/Acc", test_acc, self.round_idx)
        self.logger.add_scalar("Train/Time", end_time_train - start_time, self.round_idx)
        self.logger.add_scalar("Test/Time", end_time_test - end_time_train, self.round_idx)
        # write parameter variance
        average_parameter = self.global_model.get_param_tensor()

        param_tensors_by_workers = torch.zeros((average_parameter.shape[0], self.n_workers))

        for ii, model in enumerate(self.workers_models):
            param_tensors_by_workers[:, ii] = model.get_param_tensor() - average_parameter

        consensus = (param_tensors_by_workers ** 2).mean()
        self.logger.add_scalar("Consensus", consensus, self.round_idx)

        self.logger_write_param.write(
            f'\t Round: {self.round_idx} |Train Loss: {train_loss:.3f} |Train Acc: {train_acc * 100:.2f}% |Eval-train Time: {end_time_train - start_time:.3f}')
        self.logger_write_param.write(
            f'\t -----: {self.round_idx} |Test  Loss: {test_loss:.3f} |Test  Acc: {test_acc * 100:.2f}% |Eval-test  Time: {end_time_test - end_time_train:.3f}')
        self.logger_write_param.write(f'\t -----: Time: {time.time() - self.time_start_update:.3f}')
        self.logger_write_param.write(f'\t -----: Total Time: {time.time() - self.time_start:.3f}')
        self.time_start_update = time.time()
        if not self.args.test and (self.round_idx - 1) % 800 == 0:
            self.save_models(self.round_idx)


    def save_models(self, round):
        round_path = os.path.join(self.logger_path, 'round_%s' % round)
        os.makedirs(round_path, exist_ok=True)
        path_global = round_path + '/model_global.pth'
        model_dict = {
            'round': round,
            'model_state': self.global_model.net.state_dict()
        }
        torch.save(model_dict, path_global)
        for i in range(self.n_workers):
            path_silo = round_path + '/model_silo_%s.pth' % i
            model_dict = {
                'epoch': round,
                'model_state': self.workers_models[i].net.state_dict()
            }
            torch.save(model_dict, path_silo)

    def load_models(self, round):
        self.round_idx = round
        round_path = os.path.join(self.logger_path, 'round_%s' % round)
        path_global = round_path + '/model_global.pth'
        print('loading %s' % path_global)
        model_data = torch.load(path_global)
        self.global_model.net.load_state_dict(model_data.get('model_state', model_data))
        for i in range(self.n_workers):
            path_silo = round_path + '/model_silo_%s.pth' % i
            print('loading %s' % path_silo)
            model_data = torch.load(path_silo)
            self.workers_models[i].net.load_state_dict(model_data.get('model_state', model_data))

class Peer2PeerNetwork(Network):
    def mix(self, k, write_results=True):
        """
        :param write_results:
        Mix local model parameters in a gossip fashion
        """
        # update workers
        if self.args.multigraph:
            local_coeff = 0.7
            s = k % len(self.network)
            previous_s = (k-1) % len(self.network)
        for worker_id, model in enumerate(self.workers_models):
            model.net.to(self.device)
            if self.fit_by_epoch:
                model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                   n_epochs=self.local_steps, verbose=0)
            else:
                model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.local_steps)

        # write logs
        if ((self.round_idx - 1) % self.log_freq == 0) and write_results:
            for param_idx, param in enumerate(self.global_model.net.parameters()):
                param.data.fill_(0.)
                for worker_model in self.workers_models:
                    param.data += (1 / self.n_workers) * list(worker_model.net.parameters())[param_idx].data.clone()
            self.write_logs()

        # mix models
        for param_idx, param in enumerate(self.global_model.net.parameters()):
            temp_workers_param_list = [torch.zeros(param.shape).to(self.device) for _ in range(self.n_workers)]
            if self.args.multigraph:
                temp_workers_param_list_previous = [torch.zeros(param.shape).to(self.device) for _ in range(self.n_workers)]
            for worker_id, model in enumerate(self.workers_models):
                if self.args.multigraph:
                    count = 0
                    for neighbour in self.network[s].neighbors(worker_id):
                        count += self.network[s].get_edge_data(worker_id, neighbour)["edge"]
                    for neighbour in self.network[s].neighbors(worker_id):
                        if k == 0:
                            if (worker_id == neighbour):
                                coeff = local_coeff
                            else:
                                coeff = (1 - local_coeff) / (count - 1)
                            temp_workers_param_list[worker_id] += \
                                    coeff * list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()
                            temp_workers_param_list_previous[neighbour] += \
                                    coeff * list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()
                        else:
                            if self.network[s].get_edge_data(worker_id, neighbour)["edge"] == 1:
                                if count == 1:
                                    coeff = 1.0
                                else:
                                    if (worker_id == neighbour):
                                        coeff = local_coeff
                                    else:
                                        coeff = (1 - local_coeff) / (count - 1)
                                if (worker_id == neighbour) or self.network[previous_s].get_edge_data(worker_id, neighbour)["edge"] == 1:
                                    temp_workers_param_list[worker_id] += \
                                        coeff * list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()
                                elif self.network[previous_s].get_edge_data(worker_id, neighbour)["edge"] == 0:
                                    temp_workers_param_list[worker_id] += \
                                        coeff * list(self.workers_models_temp[neighbour].parameters())[param_idx].data.clone()
                            elif self.network[previous_s].get_edge_data(worker_id, neighbour)["edge"] == 1:
                                temp_workers_param_list_previous[neighbour] = list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()
                else:
                    for neighbour in self.network.neighbors(worker_id):
                        coeff = self.network.get_edge_data(worker_id, neighbour)["weight"]
                        temp_workers_param_list[worker_id] += \
                            coeff * list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()

            for worker_id, model in enumerate(self.workers_models):
                for param_idx_, param_ in enumerate(model.net.parameters()):
                    if param_idx_ == param_idx:
                        param_.data = temp_workers_param_list[worker_id].clone()
            if self.args.multigraph:
                for worker_id, model in enumerate(self.workers_models_temp):
                    for param_idx_, param_ in enumerate(model.parameters()):
                        if param_idx_ == param_idx:
                            param_.data = temp_workers_param_list[worker_id].clone()
        self.round_idx += 1
