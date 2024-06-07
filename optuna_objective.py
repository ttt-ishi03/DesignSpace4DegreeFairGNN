import sys
from model_specific_utils import convert_data, create_controller, create_model
import numpy as np
import csv
import torch
import argparse
from copy import copy
from utils import ttest_p_value
import json


def csv_write(csv_file, row, mode='a'):
    if csv_file is None:
        return
    with open(csv_file, mode) as f:
        writer = csv.writer(f)
        writer.writerow(row)

class Objective:
    def __init__(self, args, data, csv_file:str = None, optuna=True):
        self.params_dict = vars(args)
        self.params_file = args.params_file
        self.data = data
        self.csv_file = csv_file
        self.cnt = 0
        self.exception_cnt = 0
        self.best_score = -1
        self.best_score_std = 0
        self.best_score_params = None
        self.best_score_lst = []
        self.best_score_acc = -1
        self.best_score_sp = float('inf')
        self.best_score_eo = float('inf')
        self.best_score_state_dict = None

        self.best_acc = -1
        self.best_acc_std = 0
        self.best_acc_params = None
        self.best_acc_lst = []
        self.best_acc_sp = float('inf')
        self.best_acc_eo = float('inf')
        self.best_acc_state_dict = None

        self.best_macf = -1
        self.best_macf_std = 0
        self.best_macf_params = None
        self.best_macf_lst = []
        self.hyper_params = set()
        self.best_macf_acc = -1
        self.best_macf_sp = float('inf')
        self.best_macf_eo = float('inf')
        self.best_macf_state_dict = None

        self.optuna = optuna

        self.naive_GNN_params = {
            'n_add_edge': 0,
            'n_add_node': 0,
            'low_degree_updater_num_layers': 0,
            'w_regularization_loss': 0,
            'w_sp_loss': 0,
            'w_b_loss': 0,
            'w_film_loss': 0,
            'w_missing_information_constraint': 0,
            'w_discriminator_loss': 0,
            'w_discriminator_tailgnn_loss': 0,
            'low_degree_finetune_lr': 0,
            'w_contrastive_loss': [0, 0],
        }
        self.params = {
            'link_prediction_lr': [1e-4, 1e-3, 1e-2],
            'link_prediction_decay': [1e-5, 1e-4, 1e-3],
            'link_predictor_hidden_channels': [32, 64, 128],
            'link_predictor_num_layers': [1, 2, 3],
            'link_predictor_out_channels': [32, 64, 128],
            'link_predictor_dropout': [0.3, 0.5, 0.7],

            'n_add_edge': [1, 2],
            'classification_lr': [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            'classification_decay': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            'discriminator_lr': [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], #
            'discriminator_decay': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3], #
            'w_b_loss': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_film_loss': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_sp_loss': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_node_generator_loss': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_discriminator_loss': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_discriminator_tailgnn_loss': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_contrastive_loss1': [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_contrastive_loss2': [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_missing_information_constraint': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'w_regularization_loss': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], #
            'dim_d': [8, 16, 32, 64, 128, 256], #
            'n_add_node': [1, 2], #
            'node_generator_hidden_channels': [32, 64, 128, 256], #
            'node_generator_num_layers': [1, 2, 3], #
            'node_generator_dropout': [0.3, 0.5, 0.7], #
            'low_degree_updater_hidden_channels': [32, 64, 128, 256], #
            'low_degree_updater_num_layers': [1, 2], #
            'low_degree_updater_dropout': [0.3, 0.5, 0.7], #
            'minor_classifier_in_channels': [32, 64, 128, 256],
            'minor_classifier_hidden_channels': [32, 64, 128, 256],
            'minor_classifier_num_layers': [1, 2, 3],
            'minor_classifier_dropout': [0.3, 0.5, 0.7],
            'discriminator_hidden_channels': [32, 64, 128, 256], #
            'discriminator_num_layers': [1, 2, 3], #
            'discriminator_dropout': [0.3, 0.5, 0.7], #

            'low_degree_finetune_lr': [1e-4, 1e-3, 1e-2],
            'low_degree_finetune_decay': [1e-5, 1e-4, 1e-3],
        }
        # self.params = {
        #     'classification': {
        #         'classification_lr': [1e-4, 3e-4, 1e-3, 3e-4, 1e-2, 3e-2, 1e-1],
        #         'classification_decay': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        #         'minor_classifier_hidden_channels': [16, 32, 64, 128, 256, 512],
        #         'minor_classifier_num_layers': [2, 3],
        #         'minor_classifier_dropout': [0.3, 0.5, 0.7],
        #     }
        # }
        self.method_params = {
            'no_scale_and_shift': {
                'w_film_loss': 0,
            },
            'no_structural_contrast_degfair': {},
            'no_modulation': {
                'dim_d': 32,
            },
            'no_sp_loss': {
                'w_sp_loss': 0,
            },
            'no_b_loss': {
                'w_b_loss': 0,
            },
            'random_miss': {},
            'no_miss': {},
            'no_localization': {},
            'no_discriminator_tailgnn': {
                'w_discriminator_tailgnn_loss': 0,
            },
            'no_forged_tail_node': {},
            'no_missing_information_constraint': {
                'w_missing_information_constraint': 0,
            },
            'no_add_node': {
                'w_node_generator_loss': 0,
                'n_add_node': 0,
                'node_generator_hidden_channels': 0,
                'node_generator_num_layers': 0,
                'node_generator_dropout': 0,
            },
            'no_add_edge': {},
            'no_discriminator': {
                'discriminator_lr': 0,
                'discriminator_decay': 0,
                'w_discriminator_loss': 0,
                'discriminator_hidden_channels': 0,
                'discriminator_num_layers': 0,
                'discriminator_dropout': 0,
            },
            'no_contrastive0': {
                'w_contrastive_loss1': 0,
            },
            'no_contrastive1': {
                'w_contrastive_loss2': 0,
            },
            'no_regularization': {
                'w_regularization_loss': 0,
            },
            'no_low_degree_updater': {
                'low_degree_updater_hidden_channels': 0,
                'low_degree_updater_num_layers': 0,
                'low_degree_updater_dropout': 0,
            },
        }
        self.default_excluded_params = {}
        for key in self.method_params.keys():
            if self.params_dict[key]:
                self.default_excluded_params.update(self.method_params[key])
        self.tuned_params = {}
        if self.params_file is not None:
            with open(self.params_file, 'r') as f:
                hyper_params = json.load(f)
            self.tuned_params.update(hyper_params['best_score_params'])

        for key in self.tuned_params.keys():
            self.hyper_params.add(key)

        self.tune_target = {}
        for key, value in self.params.items():
            if key in self.default_excluded_params.keys():
                self.params_dict[key] = self.default_excluded_params[key]
            else:
                if key in self.tuned_params.keys():
                    self.params_dict[key] = self.tuned_params[key]
                else:
                    self.tune_target[key] = value
                self.hyper_params.add(key)

        self.cur_param_idx = [0] * len(self.tune_target.keys())

    def set_optuna(self, optuna):
        if self.cnt == 0:
            self.optuna = optuna
        else:
            print('optuna cannot be changed after the first iteration', file=sys.stderr)
            exit(1)

    def get_max_n_trials(self) -> int:
        ret = 1
        for key, value in self.params.items():
            if key not in self.default_excluded_params.keys() and key not in self.tuned_params.keys():
                ret *= len(value)

        return ret

    def get_best_results(self) -> dict:
        return {
            'exception_cnt': self.exception_cnt,
            'best_score': self.best_score,
            'best_score_std': self.best_score_std,
            'best_score_acc': self.best_score_acc,
            'best_score_sp': self.best_score_sp,
            'best_score_eo': self.best_score_eo,
            'best_score_params': self.best_score_params,
            'best_score_state_dict': self.best_score_state_dict,
            'best_acc': self.best_acc,
            'best_acc_std': self.best_acc_std,
            'best_acc_params': self.best_acc_params,
            'best_acc_sp': self.best_acc_sp,
            'best_acc_eo': self.best_acc_eo,
            'best_acc_state_dict': self.best_acc_state_dict,
            'best_macf': self.best_macf,
            'best_macf_std': self.best_macf_std,
            'best_macf_params': self.best_macf_params,
            'best_macf_sp': self.best_macf_sp,
            'best_macf_eo': self.best_macf_eo,
            'best_macf_state_dict': self.best_macf_state_dict,
        }

    def get_minimal_params(self) -> dict:
        return list(self.hyper_params)

    def print_best(self) -> None:
        print(f'best_score: {self.best_score}')
        print(f'best_score_std: {self.best_score_std}')
        print(f'best_score_params: {self.best_score_params}')
        print(f'best_acc: {self.best_acc}')
        print(f'best_acc_std: {self.best_acc_std}')
        print(f'best_acc_params: {self.best_acc_params}')
        print(f'best_macf: {self.best_macf}')
        print(f'best_macf_std: {self.best_macf_std}')
        print(f'best_macf_params: {self.best_macf_params}')

    def iterate(self):
        for i, v in enumerate(self.cur_param_idx):
            if v < len(list(self.tune_target.values())[i]) - 1:
                self.cur_param_idx[i] += 1
                break
            else:
                self.cur_param_idx[i] = 0

        print()
        print('iteration:', self.cnt)
        for i, (key, value) in enumerate(self.tune_target.items()):
            self.params_dict[key] = value[self.cur_param_idx[i]]
            print(f'{key}: {value[self.cur_param_idx[i]]}')

    def __call__(self, trial=None):
        if self.optuna:
            for key, value in self.tune_target.items():
                self.params_dict[key] = trial.suggest_categorical(key, value)
        else:
            self.iterate()

        score_lst = []
        sp_lst = []
        eo_lst = []
        acc_lst = []
        macf_lst = []

        args = argparse.Namespace(**self.params_dict)

        model = create_model(args, self.data)
        controller = create_controller(model, args)

        try:
            controller.train(
                data = self.data,
                n_epoch = [
                    args.link_prediction_n_epoch,
                    args.classification_n_epoch,
                    args.low_degree_finetune_n_epoch,
                ],
                early_stopping_patience = 100,
            )

            acc, macf, _, out1, _, _, _, _, _, _, _ = controller.validate(self.data)
        except Exception as e:
            print(e)
            self.exception_cnt += 1
            acc = 0
            macf = 0
            out1 = {
                'mu_sp': 0, 'mu_eo': 0,
                'head_acc': 0, 'tail_acc': 0,
                'head_macf': 0, 'tail_macf': 0,
                'diff_acc': 0, 'diff_macf': 0, 'diff_weif': 0,
            }
        results = {
            'acc': float(acc), 'macf': macf, 'sp': out1['mu_sp'], 'eo': out1['mu_eo'],
            'head_acc': out1['head_acc'], 'tail_acc': out1['tail_acc'],
            'head_macf': out1['head_macf'], 'tail_macf': out1['tail_macf'],
            'delta_f1': np.abs(out1['diff_macf'])/out1['head_macf'],
            'params': self.params_dict,
        }

        # if the scoring function is changed,
        # the controller.classification_validate should be modified
        score = acc - out1['mu_sp'] / 16 - out1['mu_eo'] / 16
        score_lst.append(score)
        sp_lst.append(results['sp'])
        eo_lst.append(results['eo'])
        acc_lst.append(acc)
        macf_lst.append(results['macf'])


        if self.cnt == 0:
            csv_write(self.csv_file, results.keys())
        csv_write(self.csv_file, [results[k] for k in results.keys()])

        if np.mean(score_lst) > self.best_score:
            self.best_score = np.mean(score_lst)
            self.best_score_std = np.std(score_lst)
            self.best_score_params = copy(self.params_dict)
            self.best_score_lst = copy(score_lst)
            self.best_score_acc = np.mean(acc_lst)
            self.best_score_sp = np.mean(sp_lst)
            self.best_score_eo = np.mean(eo_lst)
            self.best_score_state_dict = controller.get_best_model()
        if np.mean(acc_lst) > self.best_acc:
            self.best_acc = np.mean(acc_lst)
            self.best_acc_std = np.std(acc_lst)
            self.best_acc_params = copy(self.params_dict)
            self.best_acc_lst = copy(acc_lst)
            self.best_acc_sp = np.mean(sp_lst)
            self.best_acc_eo = np.mean(eo_lst)
            self.best_acc_state_dict = controller.get_best_model()
        if np.mean(macf_lst) > self.best_macf:
            self.best_macf = np.mean(macf_lst)
            self.best_macf_std = np.std(macf_lst)
            self.best_macf_params = copy(self.params_dict)
            self.best_macf_lst = copy(macf_lst)
            self.best_macf_acc = np.mean(acc_lst)
            self.best_macf_sp = np.mean(sp_lst)
            self.best_macf_eo = np.mean(eo_lst)
            self.best_macf_state_dict = controller.get_best_model()

        self.cnt += 1
        return np.mean(score_lst)
        # return (acc + macf + weif) / 3 - (np.abs(out1['diff_acc']) + np.abs(out1['diff_macf']) + np.abs(out1['diff_weif'])) / 30