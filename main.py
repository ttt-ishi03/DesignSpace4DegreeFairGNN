import os
import torch
import numpy as np
# import scipy.sparse as sp
import datetime
import pytz
import json

import datasets
from model_specific_utils import convert_data, create_controller, create_model
import argparse
# from collections import Counter
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import f1_score
import scipy.stats as st
# import math
import optuna
from utils import fix_type2json

from optuna_objective import Objective
from figure_utils import save_test_results_fig
from utils import read_hyper_params_from_csv, set_seed
import warnings

# TODO: fix the type of adj and delete the following line
warnings.simplefilter('ignore', UserWarning)


def compute_CI(out_list, name=None, log_file=None):
    # ci = 1.96 * st.sem(out_list) / math.sqrt(len(out_list))
    if None in out_list:
        return None, None
    mu = np.mean(out_list)
    std = st.sem(out_list)
    log = name + ' Mean: {:.4f} '.format(mu) + \
            'Std: {:.4f}'.format(std)
    print(log)
    return mu, std


# general arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nba', help='dataset')
parser.add_argument('--optuna', action='store_true', help='whether to use optuna or not. if this arg is given, some hyperparameters will be ignored')
parser.add_argument('--optuna_n_trials', type=int, default=100, help='number of trials for optuna')
parser.add_argument('--optuna_timeout', type=int, default=None, help='optuna timeout')
parser.add_argument('--params_file', type=str, default=None, help='hyperparameter file. the specified params in this file take precedence over other args')
parser.add_argument('--line_csv', type=int, default=100, help='if params_file is csv, read hyper_params from this number of line in params_file')
parser.add_argument('--low_degree_ratio', type=int, default=20, help='low degree ratio')
parser.add_argument('--drange', type=int, default=None, help='degree range of figure')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--delete_results', action='store_true', help='whether to save the results')
parser.add_argument('--memo2filename', type=str, default='', help='string to append to the end of a file name')
parser.add_argument('--memo', type=str, default='', help='string to write in the memo.txt')
parser.add_argument('--dir_name', type=str, default=None, help='directory name to save the results')

GENERAL_ARGS = [
    'dataset',
    'optuna',
    'optuna_n_trials',
    'optuna_timeout',
    'params_file',
    'line_csv',
    'low_degree_ratio',
    'drange',
    'seed',
    'gpu',
    'delete_results',
    'memo2filename',
]

# ablation study arguments
# TODO: 排除する機能を指定する引数ではなく，追加する機能を指定する引数に変更
parser.add_argument('--no_degfair_gnn', action='store_true', help='whether to use DegFairGNN or not')
parser.add_argument('--no_tail_gnn', action='store_true', help='whether to use Tail-GNN or not')
# degfair gnn
parser.add_argument('--no_scale_and_shift', action='store_true', help='whether to use scale and shift or not')  # no film
parser.add_argument('--no_structural_contrast_degfair', action='store_true', help='whether to use structural contrastive loss or not')
parser.add_argument('--no_modulation', action='store_true', help='whether to use modulation or not')
parser.add_argument('--no_sp_loss', action='store_true', help='whether to use sp loss or not')  # TODO: implement
parser.add_argument('--no_b_loss', action='store_true', help='whether to use b loss or not')  # TODO: implement
# tail gnn
parser.add_argument('--random_miss', action='store_true', help='whether to use random missing information or not')
parser.add_argument('--no_miss', action='store_true', help='whether to use missing information  or not')
parser.add_argument('--no_localization', action='store_true', help='whether to use localization or not')
parser.add_argument('--no_discriminator_tailgnn', action='store_true', help='whether to use discriminator of Tail-GNN or not')
parser.add_argument('--no_forged_tail_node', action='store_true', help='whether to use forged tail node or not')
parser.add_argument('--no_missing_information_constraint', action='store_true', help='whether to use missing information constraint or not')  # TODO: implement
# others
parser.add_argument('--no_add_node', action='store_true', help='whether to add node or not')  # TODO: implement
parser.add_argument('--no_add_edge', action='store_true', help='whether to add edge or not')  # TODO: implement
parser.add_argument('--no_discriminator', action='store_true', help='whether to use discriminator or not')  # TODO: implement
parser.add_argument('--no_contrastive0', action='store_true', help='whether to use contrastive0 loss or not')  # TODO: implement
parser.add_argument('--no_contrastive1', action='store_true', help='whether to use contrastive1 loss or not')  # TODO: implement
parser.add_argument('--no_regularization', action='store_true', help='whether to use regularization loss or not')  # TODO: implement
parser.add_argument('--no_low_degree_updater', action='store_true', help='whether to use low degree updater or not')  # TODO: implement
parser.add_argument('--no_low_degree_finetune', action='store_true', help='whether to use low degree finetune or not')  # TODO: implement

HIGH_PRIORITY_ARGS = [
    'no_degfair_gnn',
    'no_tail_gnn',
    'no_scale_and_shift',
    'no_structural_contrast_degfair',
    'no_modulation',
    'no_sp_loss',
    'no_b_loss',
    'random_miss',
    'no_miss',
    'no_localization',
    'no_discriminator_tailgnn',
    'no_forged_tail_node',
    'no_missing_information_constraint',
    'no_add_node',
    'no_add_edge',
    'no_discriminator',
    'no_contrastive0',
    'no_contrastive1',
    'no_regularization',
    'no_low_degree_updater',
    'no_low_degree_finetune',
]

# TODO: early stoppingのエポック数の閾値を設定する引数を追加

# TODO: 並び替え
# training arguments
parser.add_argument('--link_prediction_n_epoch', type=int, default=1024, help='number of iteration for link prediction')
parser.add_argument('--classification_n_epoch', type=int, default=1024, help='number of iteration for classification')
parser.add_argument('--low_degree_finetune_n_epoch', type=int, default=1024, help='number of iteration for discriminator')
parser.add_argument('--link_prediction_lr', type=float, default=1e-2, help='learning rate for link predictor')
parser.add_argument('--link_prediction_decay', type=float, default=1e-4, help='weight decay for link predictor')
parser.add_argument('--classification_lr', type=float, default=1e-2, help='learning rate for classifier')
parser.add_argument('--classification_decay', type=float, default=1e-4, help='weight decay for classifier')
parser.add_argument('--discriminator_lr', type=float, default=1e-2, help='learning rate for discriminator')
parser.add_argument('--discriminator_decay', type=float, default=1e-4, help='weight decay for discriminator')
parser.add_argument('--low_degree_finetune_lr', type=float, default=1e-2, help='learning rate for low degree updater')
parser.add_argument('--low_degree_finetune_decay', type=float, default=1e-4, help='weight decay for low degree updater')
parser.add_argument('--w_b_loss', type=float, default=1e-04, help='weight constraint')
parser.add_argument('--w_film_loss', type=float, default=1e-04, help='weight FILM')
parser.add_argument('--w_sp_loss', type=float, default=1e-04, help='weight fair')
parser.add_argument('--w_node_generator_loss', type=float, default=1e-04, help='weight for node generator loss')
parser.add_argument('--w_discriminator_loss', type=float, default=1e-04, help='weight for discriminator loss')
parser.add_argument('--w_discriminator_tailgnn_loss', type=float, default=1e-4, help='weight for TailGNN discriminator loss')
parser.add_argument('--w_contrastive_loss1', type=float, default=1e-4, help='weights for contrastive loss')
parser.add_argument('--w_contrastive_loss2', type=float, default=1e-4, help='weights for contrastive loss')
parser.add_argument('--w_missing_information_constraint', type=float, default=1e-4, help='weights for TailGNN missing information constraint loss')
parser.add_argument('--w_regularization_loss', type=float, default=1e-4, help='weight for regularization loss')

# model arguments
parser.add_argument('--model', type=str, default='DegFairGNN', help='model name (DegFarGNN, GCN, GAT, GraphSAGE, GloGNN)')
parser.add_argument('--base', type=int, default=1, help='1: GCN, 2: GAT, 3: Sage')
parser.add_argument('--discriminator', type=str, default='MLP', help='discriminator(MLP/GCN)')
parser.add_argument('--dim_d', type=int, default=32, help='degree mat dimension')
parser.add_argument('--omega', type=float, default=0.1, help='weight bias')
parser.add_argument('--k', type=float, default=1, help='ratio split head and tail group')
parser.add_argument('--n_add_edge', type=int, default=1, help='how many edges to add to low degree node when model is NodeGenGNN or AllGNN')
parser.add_argument('--n_add_node', type=int, default=0, help='how many nodes to add to low degree node when model == NodeGenGNN')

parser.add_argument('--link_predictor_hidden_channels', type=int, default=32, help='hidden channel of link predictor')
parser.add_argument('--link_predictor_num_layers', type=int, default=2, help='number of layers of link predictor')
parser.add_argument('--link_predictor_out_channels', type=int, default=32, help='output channel of link predictor')
parser.add_argument('--link_predictor_dropout', type=float, default=0.5, help='dropout rate of link predictor')

parser.add_argument('--node_generator_hidden_channels', type=int, default=32, help='hidden channel of node generator')
parser.add_argument('--node_generator_num_layers', type=int, default=2, help='number of layers of node generator')
parser.add_argument('--node_generator_dropout', type=float, default=0.5, help='dropout rate of node generator')

parser.add_argument('--low_degree_updater_hidden_channels', type=int, default=32, help='hidden channel of low degree updater')
parser.add_argument('--low_degree_updater_num_layers', type=int, default=2, help='number of layers of low degree updater')
parser.add_argument('--low_degree_updater_dropout', type=float, default=0.5, help='dropout rate of low degree updater')

parser.add_argument('--minor_classifier_in_channels', type=int, default=32, help='input channel of minor classifier')
parser.add_argument('--minor_classifier_hidden_channels', type=int, default=32, help='hidden channel of minor classifier')
parser.add_argument('--minor_classifier_num_layers', type=int, default=2, help='number of layers of minor classifier')
parser.add_argument('--minor_classifier_dropout', type=float, default=0.5, help='dropout rate of minor classifier')
parser.add_argument('--minor_classifier_intermediate_layer', type=int, default=0, help='number of intermediate layers of minor classifier')

parser.add_argument('--discriminator_hidden_channels', type=int, default=32, help='hidden channel of discriminator')
parser.add_argument('--discriminator_num_layers', type=int, default=2, help='number of layers of discriminator')
parser.add_argument('--discriminator_dropout', type=float, default=0.5, help='dropout rate of discriminator')

init_args = parser.parse_args()

cuda = torch.cuda.is_available()

# TODO: generalized degreeの機能削除
# TODO: generalized degreeの機能削除に伴ったリファクタリング．変数名_1, 変数名_2みたいなものを消す

def main():
    args_dict = vars(init_args)
    if init_args.no_degfair_gnn:
        args_dict['no_scale_and_shift'] = True
        args_dict['no_structural_contrast_degfair'] = True
        args_dict['no_modulation'] = True
        args_dict['no_sp_loss'] = True
        args_dict['no_b_loss'] = True
    if init_args.no_tail_gnn:
        args_dict['random_miss'] = True
        args_dict['no_miss'] = True
        args_dict['no_localization'] = True
        args_dict['no_discriminator_tailgnn'] = True
        args_dict['no_forged_tail_node'] = True
        args_dict['no_missing_information_constraint'] = True
    args = argparse.Namespace(**args_dict)
    print(str(args))

    modelname = args.model
    if args.model == "DegFairGNN":
        if args.base == 1:
            modelname = "DegFairGCN"
        elif args.base == 2:
            modelname = "DegFairGAT"
        elif args.base == 3:
            modelname = "DegFairSAGE"
    elif args.model == "AllDegFairGNN":
        if args.base == 1:
            modelname = "AllDegFairGCN"
        elif args.base == 2:
            modelname = "AllDegFairGAT"
        elif args.base == 3:
            modelname = "AllDegFairSAGE"

    print('model: ', modelname)
    print('dataset: ', args.dataset)

    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    current_time = now.strftime("%Y%m%d%H%M%S")
    if cuda:
        torch.cuda.set_device(args.gpu)

    num = 5
    np.random.seed(args.seed)
    seed = np.random.choice(100, num, replace=False)

    if args.dir_name is not None:
        experiment_root = os.path.join('./experiment', args.dataset, args.dir_name)
    else:
        experiment_root = os.path.join('./experiment', args.dataset, f'{modelname}{args.memo2filename}_{current_time}{args.gpu}')
    
    if not os.path.exists(experiment_root) and not args.delete_results:
        print(f'making directory {experiment_root}')
        os.makedirs(experiment_root)

    f_acc = list()
    f_macf = list()
    f_weif = list()
    SP_1 = list()
    EO_1 = list()
    h_acc_1 = list()
    t_acc_1 = list()
    diff_acc_1 = list()
    h_macf_1 = list()
    t_macf_1 = list()
    diff_macf_1 = list()
    h_weif_1 = list()
    t_weif_1 = list()
    diff_weif_1 = list()
    delta_f1 = list()

    if not args.delete_results and args.memo != '':
        memo_txt = os.path.join(experiment_root, 'memo.txt')
        with open(memo_txt, 'w') as f:
            f.write(args.memo)

    for i in range(num):
        print('================================================================')
        print(f'{i+1}  /  {num}')
        print('Seed: ', seed[i])

        set_seed(seed[i])
        data = datasets.get_dataset(args.dataset, norm=False, ratio=args.low_degree_ratio)
        data.to_tensor()
        print('num of nodes:', data.feat.shape[0])
        print('embedding size:', data.feat.shape[1])
        print('num of edges:', int((sum(data.degree)[0]/2).item()))
        print('num of class', len(np.unique(data.labels)))
        print('dataset prepared')
        data = convert_data(data, args.model, args.base, ratio=args.low_degree_ratio)
        experiment_seed_root = os.path.join(experiment_root, f'seed{seed[i]}')
        if not os.path.exists(experiment_seed_root) and not args.delete_results:
            print(f'making directory {experiment_seed_root}')
            os.makedirs(experiment_seed_root)

        if args.optuna:
            print('starting Optuna')
            print('note: some hyper-params will be ignored even if args of the hyper-params are set')
            if args.params_file is not None and '???' in args.params_file:
                print('params_file include "???", it will be replaced with seed')
                args_dict = vars(args)
                args_dict['params_file'] = args.params_file.replace('???', str(seed[i]))
                args = argparse.Namespace(**args_dict)

            log_file = os.path.join(experiment_seed_root, f'{modelname}_optuna_log.csv')
            output_file_whole_args = os.path.join(experiment_seed_root, f'{modelname}_best_params.json')
            output_file_minimal_args = os.path.join(experiment_seed_root, f'{modelname}_best_params_minimal.json')

            objective = Objective(args, data, csv_file=log_file)
            objective_n_trials = objective.get_max_n_trials()
            # if n_trials is larger than the maximum number of trials
            # it will run grid-search
            if args.optuna_n_trials is not None and args.optuna_n_trials > objective_n_trials:
                objective.set_optuna(False)
                args.optuna_n_trials = max(objective_n_trials, 1)
                for _ in range(args.optuna_n_trials):
                    objective()
            else:
                study = optuna.create_study(direction='maximize')
                if args.optuna_timeout is None:
                    study.optimize(objective, n_trials=args.optuna_n_trials)
                else:
                    study.optimize(objective, timeout=args.optuna_timeout)

            if not args.delete_results:
                best_results = objective.get_best_results()
                # save state_dict
                pth_path = {
                    'best_score_state_dict': {
                        'link_predictor': None,
                        'classifier': None,
                        'low_degree_specific_classifier': None,
                    },
                    'best_acc_state_dict': {
                        'link_predictor': None,
                        'classifier': None,
                        'low_degree_specific_classifier': None,
                    },
                    'best_macf_state_dict': {
                        'link_predictor': None,
                        'classifier': None,
                        'low_degree_specific_classifier': None,
                    },
                }
                for k in pth_path.keys():
                    if k in best_results:
                        for m in pth_path[k].keys():
                            if best_results[k][m] is not None:
                                path = os.path.join(experiment_seed_root, f'{modelname}_{m}_{k}.pth')
                                torch.save(best_results[k][m], path)
                                pth_path[k][m] = path
                list_key = [
                    'exception_cnt',
                    'best_score', 'best_score_std', 'best_score_acc', 'best_score_sp', 'best_score_eo', 'best_score_params',
                    'best_acc', 'best_acc_std', 'best_acc_params', 'best_acc_sp', 'best_acc_eo',
                    'best_macf', 'best_macf_std', 'best_macf_params', 'best_macf_sp', 'best_macf_eo',
                ]
                best_params = {k: best_results[k] for k in list_key}
                optimal_params = objective.get_minimal_params()
                best_params.update(pth_path)
                with open(output_file_whole_args, 'w') as f:
                    json.dump(best_params, f, indent=4)
                with open(output_file_minimal_args, 'w') as f:
                    best_params['best_score_params'] = {k: v for k, v in best_params['best_score_params'].items() if k in optimal_params or k in pth_path.keys()}
                    best_params['best_acc_params'] = {k: v for k, v in best_params['best_acc_params'].items() if k in optimal_params or k in pth_path.keys()}
                    best_params['best_macf_params'] = {k: v for k, v in best_params['best_macf_params'].items() if k in optimal_params or k in pth_path.keys()}
                    json.dump(best_params, f, indent=4)

            print(i, 'Optuna done !\n')

        # load hyper parameters
        params_file = None
        if args.optuna:
            params_file = output_file_minimal_args
        elif args.params_file is not None:
            params_file = args.params_file

        if params_file is not None:
            print(f'loading hyper-params from {params_file}')
            with open(params_file, 'r') as f:
                hyper_params = json.load(f)

            params_key = 'best_score_params'
            if  not args.delete_results:
                with open(os.path.join(experiment_seed_root,
                            f'{modelname}{args.memo2filename}_current_params_minimal.json'), 'w') as f:
                    json.dump(hyper_params[params_key], f, indent=4)
            print(f'hyper-params: {hyper_params[params_key]}')
            args_dict = vars(args)
            for k, v in hyper_params[params_key].items():
                if k not in HIGH_PRIORITY_ARGS + GENERAL_ARGS:
                    args_dict[k] = v
            args = argparse.Namespace(**args_dict)
            if not args.delete_results:
                with open(os.path.join(experiment_seed_root,
                            f'{modelname}{args.memo2filename}_current_params.json'), 'w') as f:
                    json.dump(args_dict, f, indent=4)

        # SP_2 = list()
        # EO_2 = list()
        # h_acc_2 = list()
        # t_acc_2 = list()

        model_dic = create_model(args, data)
        controller = create_controller(model_dic, args)
        # TODO: 引数で指定したパスからモデルのweightを読み込む機能を追加
        if args.optuna:
            controller.load_state_dict(**best_results['best_score_state_dict'])
        else:
            # TODO: lrとdecayはここで設定．controllerの定義で設定するべきではない
            controller.train(
                data = data,
                n_epoch = [
                    args.link_prediction_n_epoch,
                    args.classification_n_epoch,
                    args.low_degree_finetune_n_epoch,
                ],
                early_stopping_patience = 100,
            )

        dranges = {'citeseer': 16, 'chameleon': 30, 'arxiv-year': 30, 'emnlp': 16, 'squirrel': 30}
        if args.drange is None:
            drange = dranges[args.dataset]
        else:
            drange = args.drange
        acc, macf, weif, out1, out2, deg_acc, deg_macf, deg_weif, preds, labels, degree = controller.test(data, drange)

        if not args.delete_results:
            save_test_results_fig(
                labels,
                preds,
                n_class=data.n_class,
                recall = {
                    'low': out1['recall0'],
                    'high': out1['recall1'],
                    'mid': out1['recall_other']
                },
                degree=degree,
                seed=seed[i],
                output_path=experiment_seed_root,
                dataset=args.dataset,
                ratio=args.low_degree_ratio,
            )

        output_file = os.path.join(experiment_seed_root, f'{modelname}{args.memo2filename}_result.json')
        result = {'acc': acc, 'macf': macf, 'out1': out1, 'deg_acc': deg_acc, 'deg_macf': deg_macf, 'deg_weif': deg_weif}
        if not args.delete_results:
            with open(output_file, 'w') as f:
                result = fix_type2json(result)
                json.dump(result, f, indent=4)

        f_acc.append(acc)
        f_macf.append(macf)
        f_weif.append(weif)
        SP_1.append(out1['mu_sp'])
        EO_1.append(out1['mu_eo'])
        h_acc_1.append(out1['head_acc'])
        t_acc_1.append(out1['tail_acc'])
        diff_acc_1.append(out1['diff_acc'])
        h_macf_1.append(out1['head_macf'])
        t_macf_1.append(out1['tail_macf'])
        diff_macf_1.append(out1['diff_macf'])
        h_weif_1.append(out1['head_weif'])
        t_weif_1.append(out1['tail_weif'])
        diff_weif_1.append(out1['diff_weif'])
        delta_f1.append(abs(out1['diff_macf'])/out1['head_macf'])

        del model_dic
        del controller

    print('--------------------------------------------------------------')
    ar = dict()
    ar['acc_mu'], ar['acc_std'] = compute_CI(f_acc, name='Acc')
    ar['macf_mu'], ar['macf_std'] = compute_CI(f_macf, name='Macf')
    ar['weif_mu'], ar['weif_std'] = compute_CI(f_weif, name='Weif')

    ar['head_acc_1_mu'], ar['head_acc_1_std'] = compute_CI(h_acc_1, name='Head Acc')
    ar['tail_acc_1_mu'], ar['tail_acc_1_std'] = compute_CI(t_acc_1, name='Tail Acc')
    ar['diff_acc_1_mu'], ar['diff_acc_1_std'] = compute_CI(diff_acc_1, name='Diff Acc')
    ar['head_macf_1_mu'], ar['head_macf_1_std'] = compute_CI(h_macf_1, name='Head Macf')
    ar['tail_macf_1_mu'], ar['tail_macf_1_std'] = compute_CI(t_macf_1, name='Tail Macf')
    ar['diff_macf_1_mu'], ar['diff_macf_1_std'] = compute_CI(diff_macf_1, name='Diff Macf')
    ar['head_weif_1_mu'], ar['head_weif_1_std'] = compute_CI(h_weif_1, name='Head Weif')
    ar['tail_weif_1_mu'], ar['tail_weif_1_std'] = compute_CI(t_weif_1, name='Tail Weif')
    ar['diff_weif_1_mu'], ar['diff_weif_1_std'] = compute_CI(diff_weif_1, name='Diff Weif')
    ar['delta_f1_mu'], ar['delta_f1_std'] = compute_CI(delta_f1, name='Delta F1')
    ar['SP_1_mu'], ar['SP_1_std'] = compute_CI(SP_1, name='SP')
    ar['EO_1_mu'], ar['EO_1_std'] = compute_CI(EO_1, name='EO')
    ar['args'] = args

    output_file = os.path.join(experiment_root, f'{modelname}{args.memo2filename}_all_result_seed_{args.seed}.json')
    if not args.delete_results:
        with open(output_file, 'w') as f:
            ar = fix_type2json(ar)
            json.dump(ar, f, indent=4)

    if not args.delete_results:
        print(f'saved result to "{experiment_root}"')

    print('Done!')

if __name__ == "__main__":
    main()
