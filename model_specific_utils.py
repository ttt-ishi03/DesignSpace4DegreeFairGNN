import sys
import copy
import numpy as np
import datasets
import scipy.sparse as sp
import torch
from torch_geometric.nn.models import GCN
from torch_geometric.nn import summary
from models import DFair_GCN, DFair_GAT, DFair_Sage #DFairGNN_2
from models import AllGNN, LinkPredictor
from models import MLP, RandomModel
from models import GCNWithIntermediateOutput, GATWithIntermediateOutput, GraphSAGEWithIntermediateOutput
from models import RandomDegreeDiscriminator
from models import DiscriminatorTailGNN

from controllers import DegFairGNNController, AllGNNController, RandomModelController


def convert_data(data, model_type, model_base, ratio=20):
    data.degree_label = copy.deepcopy(data.labels).detach()
    degree = np.array(data.degree.cpu().detach()).flatten()
    p = np.percentile(degree, [ratio, 100-ratio])
    data.degree_label[degree <= p[0]] = 0
    data.degree_label[p[0] < degree] = 1
    data.degree_label[p[1] < degree] = 2
    data.adj = datasets.convert_sparse_tensor(sp.csc_matrix(data.adj))
    data.adj = data.adj.coalesce()
    data.adj_ = datasets.convert_sparse_tensor(sp.csc_matrix(data.adj_))
    data.adj_ = data.adj_.coalesce()
    if model_base == 2:
        data.edge = data.adj.indices()
        data.edge_ = data.adj_.indices()
    if model_type == "DegFairGNN":
        pass
    elif model_type in ["DegreeSpecificGAT"]:
        data.adj = data.adj.indices()
        data.adj_ = data.adj_.indices()
    elif model_type in ["DegreeDiscriminator", "RandomDegreeDiscriminator"]:
        print("this model don't estimate the class of the dataset, but estimate the degree group of each nodes in the dataset")
        degree = np.array(data.degree.cpu().detach()).flatten()
        p = np.percentile(degree, [ratio, 100-ratio])
        data.labels[degree <= p[0]] = 0
        data.labels[p[0] < degree] = 1
        data.labels[p[1] < degree] = 2
    elif model_type in ['GCN', 'GAT', 'GraphSAGE']:
        data.adj = data.adj.to_sparse_csr()
        data.adj_ = data.adj_.to_sparse_csr()
    elif model_type in ["MLP", "AllGCN", "AllGAT", "AllSAGE", "DegreeFairGNN"]:
        pass
    else:
        print('model invalid', file=sys.stderr)
        exit(1)
    
    if torch.cuda.is_available():
        data.to_cuda()

    return data


def create_model(args, data):
    # check the shape of data
    in_feat = data.feat.shape[1]
    in_class = data.labels.max().item() + 1

    model_dic = {}

    # if args.model == "DegFairGNN":
    #     if args.base == 1:
    #         model_dic['main'] = DFair_GCN(args, in_feat, in_class, data.max_degree)
    #     elif args.base == 2:
    #         model_dic['main'] = DFair_GAT(args, in_feat, in_class, data.max_degree)
    #     elif args.base == 3:
    #         model_dic['main'] = DFair_Sage(args, in_feat, in_class, data.max_degree)
    #     else:
    #         ValueError('model invalid')
    if args.model in ["AllGCN", "AllGAT", "AllSAGE", "DegreeFairGNN"]:
        if args.edge_adding_4_low and args.n_add_edge > 0 and args.link_predictor_num_layers > 0 and args.link_prediction_lr > 0:
            link_predictor_base = MLP(
                in_channels = in_feat,
                hidden_channels = args.link_predictor_hidden_channels,
                num_layers = args.link_predictor_num_layers,
                out_channels = args.link_predictor_out_channels,
                dropout = args.link_predictor_dropout,
            )
            model_dic['link_predictor'] = LinkPredictor(
                base_model = link_predictor_base,
                similarity_based = True,
            )
        else:
            model_dic['link_predictor'] = None

        if args.node_adding_4_low and args.n_add_node > 0 and args.node_generator_num_layers > 0:
            node_generator = GCN(
                in_channels = args.minor_classifier_in_channels,
                hidden_channels = args.node_generator_hidden_channels,
                num_layers = args.node_generator_num_layers,
                out_channels = args.minor_classifier_in_channels,
                dropout = args.node_generator_dropout,
            )
        else:
            node_generator = None

        if args.low_degree_additional_layer and args.low_degree_updater_num_layers > 0:
            low_degree_updater = GCN(
                in_channels = args.minor_classifier_in_channels,
                hidden_channels = args.low_degree_updater_hidden_channels,
                num_layers = args.low_degree_updater_num_layers,
                out_channels = args.minor_classifier_in_channels,
                dropout = args.low_degree_updater_dropout,
            )
        else:
            low_degree_updater = None

        kwargs_basic_gnn = {
            'in_channels': args.minor_classifier_in_channels,
            'hidden_channels': args.minor_classifier_hidden_channels,
            'num_layers': args.minor_classifier_num_layers,
            'out_channels': in_class,
            'dropout': args.minor_classifier_dropout,
            'intermediate_layer': args.minor_classifier_intermediate_layer,
        }
        kwargs_degfair_gnn = {
            'in_channels': args.minor_classifier_in_channels,
            'hidden_channels': args.minor_classifier_hidden_channels,
            'out_channels': in_class,
            'dropout': args.minor_classifier_dropout,
            'base': args.base,
            'dataset': args.dataset,
            'dim_d': args.dim_d,
            'omega': args.omega,
            'k': args.k,
            'max_degree': data.max_degree,
            'structural_contrast': args.structural_contrast_on_degfair,
            'modulation': args.degree_injection,
            'random_miss': False,
            'miss': args.missing_info,
            'localization': args.missing_info,
        }
        if args.model == "AllGCN":
            minor_classifier = GCNWithIntermediateOutput(**kwargs_basic_gnn)
        elif args.model == "AllGAT":
            minor_classifier = GATWithIntermediateOutput(**kwargs_basic_gnn)
        elif args.model == "AllSAGE":
            minor_classifier = GraphSAGEWithIntermediateOutput(**kwargs_basic_gnn)
        elif args.model == "DegreeFairGNN":
            if args.base == 1:
                minor_classifier = DFair_GCN(**kwargs_degfair_gnn)
            elif args.base == 2:
                minor_classifier = DFair_GAT(**kwargs_degfair_gnn)
            elif args.base == 3:
                minor_classifier = DFair_Sage(**kwargs_degfair_gnn)
        minor_classifier.cuda()

        # for AllGNN
        # n_dummy_node = 3
        # dummy_feat = torch.zeros(n_dummy_node, kwargs_basic_gnn['in_channels']).cuda()
        # dummy_adj = torch.zeros(n_dummy_node, n_dummy_node).cuda()
        # dummy_adj[0, 1] = 1
        # dummy_adj = dummy_adj.to_sparse_coo()
        # print(summary(minor_classifier, dummy_feat, dummy_adj))

        model_dic['classifier'] = AllGNN(
            in_channels = in_feat,
            hidden_channels = args.minor_classifier_in_channels,
            classifier = minor_classifier,
            node_generator = node_generator,
            low_degree_updater = low_degree_updater,
            n_add_node = args.n_add_node,
            is_input_layer_discriminator_enabled = args.degree_disc,
            is_input_layer_contrastive_enabled = False,
        )
        kwargs = {
            'in_channels': args.minor_classifier_in_channels,
            'hidden_channels': args.discriminator_hidden_channels,
            'num_layers': args.discriminator_num_layers,
            'out_channels': 3,  # low, high or middle
            'dropout': args.discriminator_dropout,
        }
        if args.degree_disc:
            if args.discriminator == 'MLP':
                model_dic['discriminator'] = MLP(**kwargs)
            elif args.discriminator == 'GCN':
                model_dic['discriminator'] = GCN(**kwargs)
        else:
            model_dic['discriminator'] = None

        if args.augmented_low_degree_disc and args.edge_removing:
            model_dic['discriminator_tailgnn'] = DiscriminatorTailGNN(in_class)
        else:
            model_dic['discriminator_tailgnn'] = None
    # elif args.model == "DegreeDiscriminator":
    #     model_dic['main'] = MLP(in_feat, args.dim, args.dim2, 3, args.dropout)
    elif args.model == "RandomDegreeDiscriminator":
        model_dic['main'] = RandomDegreeDiscriminator()
    elif args.model == "MLP":
        model_dic['main'] = MLP(in_feat, args.dim, args.dim2, in_class)
    elif args.model == "Random":
        model_dic['main'] = RandomModel(in_class)
    else:
        print('model invalid', file=sys.stderr)
        exit(1)
    
    if torch.cuda.is_available():
        for key in model_dic.keys():
            if model_dic[key] is not None:
                model_dic[key].cuda()

    return model_dic


def create_controller(model, args):
    if args.model in ["DegFairGNN"]:
        controller = DegFairGNNController(model, args.loss, args.lr, args.decay, args.w_f, args.w_b)
    elif args.model in ["AllGCN", "AllGAT", "AllSAGE", "DegreeFairGNN"]:
        controller = AllGNNController(
            link_predictor = model['link_predictor'],
            discriminator = model['discriminator'],
            discriminator_tailgnn = model['discriminator_tailgnn'],
            classifier = model['classifier'],
            link_prediction_lr = args.link_prediction_lr,
            link_prediction_decay = args.link_prediction_decay,
            classification_lr = args.classification_lr,
            classification_decay = args.classification_decay,
            discriminator_lr = args.discriminator_lr,
            discriminator_decay = args.discriminator_decay,
            low_degree_finetune_lr = args.low_degree_finetune_lr,
            low_degree_finetune_decay = args.low_degree_finetune_decay,
            w_sp_loss = args.w_sp_loss,
            w_b_loss = args.w_b_loss,
            w_film_loss = args.w_film_loss,
            w_node_generator_loss = args.w_node_generator_loss,
            w_contrastive_loss = [args.w_contrastive_loss1, args.w_contrastive_loss2],
            w_discriminator_loss = args.w_discriminator_loss,
            w_discriminator_tailgnn_loss = args.w_discriminator_tailgnn_loss,
            w_missing_information_constraint = args.w_missing_information_constraint,
            w_regularization_loss = args.w_regularization_loss,
            n_add_edge = args.n_add_edge,
            n_add_node = args.n_add_node,
            low_degree_additional_layer = args.low_degree_additional_layer and (args.low_degree_updater_num_layers > 0),
            scale_and_shift = args.structural_contrast_on_degfair or args.degree_injection,
            forged_tail_node = args.edge_removing,
            sp_loss = args.mean_cl,
            b_loss = args.structural_contrast_on_degfair or args.degree_injection,
            missing_information_constraint = args.missing_info,
            add_edge = args.edge_adding_4_low,
            add_node = args.node_adding_4_low,
            degree_discriminator = args.degree_disc,
            contrastive0 = False,
            contrastive1 = args.pairwise_degree_cl,
            regularization = False,
            low_degree_finetune = args.low_degree_finetune,
        )
    elif args.model in ['Random']:
        controller = RandomModelController(model)
    elif args.model in ['RandomDegreeDiscriminator']:
        controller = RandomModelController(model)
    else:
        print('model invalid', file=sys.stderr)
        exit(1)
    
    return controller
