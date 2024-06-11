import sys, os
import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import copy
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_accuracy, binary_precision, binary_recall, binary_f1_score, multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score

from typing import List, Callable, Union
from utils import normalize_output, create_tail_adj

EPS = 1e-4

class Controller(object):
    def __init__(self, model_dic, task_loss, lr, decay):
        self.model = model_dic['main']
        self.lr = lr
        self.decay = decay
        self.best_model_state_dict = None
        self.need_pretrain = False
        self.need_aftrain = False
        self.task_loss_name = task_loss
        # if task_loss is None:
        #     self.task_loss = nn.CrossEntropyLoss()
        # else:
        #     self.task_loss = task_loss

        if task_loss == 'cross_entropy_loss':
            self.task_loss = nn.CrossEntropyLoss()
        elif task_loss == 'degree_weighted_cross_entropy_loss':
            self.task_loss = utils.degree_weighted_cross_entropy_loss

    def train(self, data, n_epoch, flexible_epoch, pretrain=False, aftrain=False):
        if pretrain:
            if self.need_pretrain:
                print('Pre-Training ...')
            else:
                return
        elif aftrain:
            if self.need_aftrain:
                print('After-Training ...')
            else:
                return
        else:
            print('Training ...')
        if pretrain:
            self.optim = optim.Adam(self.model.parameters(),
                        lr=self.pretrain_lr, weight_decay=self.decay)
        else:
            self.optim = optim.Adam(self.model.parameters(),
                        lr=self.lr, weight_decay=self.decay)

        for key in self.loss_lst.keys():
            self.loss_lst[key] = []
        min_val_loss = float('inf')

        i = 0
        best_epoch = 0
        while 1:
            if flexible_epoch and i >= best_epoch + 100 or not flexible_epoch and i >= n_epoch:
                break
            self.model.train()
            self.optim.zero_grad()
            if pretrain:
                loss = self.inference_and_calc_loss_for_pretrain(data, save_loss=True)
            else:
                loss = self.inference_and_calc_loss(data, save_loss=True)
            loss.backward()
            self.optim.step()
            if i%25 == 0:
                loss_terms = list(self.loss_lst.keys())
                loss_terms = [term for term in loss_terms if self.loss_lst[term] != [] and self.loss_lst[term][-1] is not None]
                print('epoch {} {} = ('.format(i, loss_terms), end='')
                for term in loss_terms:
                    if term != loss_terms[-1]:
                        print('{:.2f}, '.format(self.loss_lst[term][-1]), end='')
                    else:
                        print('{:.2f})'.format(self.loss_lst[term][-1]))

            val_loss = self.validation(data, pretrain)
            if val_loss < min_val_loss:
                best_epoch = i
                min_val_loss = val_loss
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            i += 1

        print('best epoch {}: validation loss = ({:.2f})'.format(best_epoch, min_val_loss))
        self.model.load_state_dict(self.best_model_state_dict)
        if pretrain:
            self.post_process_after_pretrain()
        return self.loss_lst

    def validation(self, data, pretrain=False, aftrain=False):
        self.model.eval()
        with torch.no_grad():
            if pretrain:
                loss = self.inference_and_calc_val_score_for_pretrain(data)
            if aftrain:
                loss = self.inference_and_calc_val_score_for_aftrain(data)
            else:
                loss = self.inference_and_calc_val_score(data)
        return loss

    def test(self, data, degree_range, enable_deg2):
        print('Testing ...')

        degree = np.array(data.degree.cpu(), dtype='uint16').flatten()
        mask = data.idx_test
        self.model.eval()

        with torch.no_grad():
            # accuracy
            output = self.get_output_for_test(data)
            acc = utils.accuracy(output[mask], data.labels[mask])
            acc = acc.cpu()

            preds = output.max(1)[1].cpu().detach()
            macf = f1_score(data.labels[mask].cpu(), preds[mask], average='macro')
            weif = f1_score(data.labels[mask].cpu(), preds[mask], average='weighted')

            degree_acc, degree_macf, degree_weif = utils.metrics_by_degree(preds, data, data.group1, degree_range)

            out1 = utils.evaluate_fairness(preds, data, data.group1, embed=output[mask])
            out2 = None
            if enable_deg2:
                out2 = utils.evaluate_fairness(preds, data, data.group2)
            print('Accuracy={:.4f}'.format(acc))
            print('Macro-F1={:.4f}'.format(macf))

        return acc*100, macf*100, weif*100, out1, out2, degree_acc, degree_macf, degree_weif, preds[mask], data.labels[mask], degree[mask]

    def calc_task_loss(self, output, label, degree_label=None):
        if self.task_loss_name == 'cross_entropy_loss':
            task_loss = self.task_loss(output, label)
        elif self.task_loss_name == 'degree_weighted_cross_entropy_loss':
            if degree_label is None:
                print('degree_label is required', file=sys.stderr)
                exit(1)
            task_loss = self.task_loss(output, label, degree_label)
        return task_loss

    def post_process_after_pretrain(self):
        pass


def _get_link(adj, n_target_edge):
    n_node = adj.shape[0]
    n_edge = adj.indices().shape[1]

    random_indices = torch.randperm(n_edge)[:n_target_edge]
    pos_link = adj.indices()[:, random_indices]
    neg_link = torch.randint(0, n_node, [n_target_edge, 2]).cuda()
    is_not_self_loop = neg_link[:, 0] != neg_link[:, 1]
    neg_link = neg_link[is_not_self_loop]
    _, inverse, counts = torch.unique(torch.concatenate([neg_link, adj.indices().T], axis=0), dim=0, return_inverse=True, return_counts=True)
    n_neg_link = neg_link.shape[0]
    neg_link = neg_link[counts[inverse[:n_neg_link]] == 1].T
    n_neg_link = neg_link.shape[1]

    return pos_link, neg_link


class AllGNNController(object):
    """
    for AllGNN
    """
    def __init__(
            self,
            link_predictor: nn.Module,
            discriminator: nn.Module,
            discriminator_tailgnn: nn.Module,
            classifier: nn.Module,
            link_prediction_lr: float,
            link_prediction_decay: float,
            classification_lr: float,
            classification_decay: float,
            discriminator_lr: float,
            discriminator_decay: float,
            low_degree_finetune_lr: float,
            low_degree_finetune_decay: float,
            w_node_generator_loss: float,
            w_contrastive_loss: List[float],
            w_discriminator_loss: float,
            w_discriminator_tailgnn_loss: float,
            w_regularization_loss: float,
            w_missing_information_constraint: float,
            w_sp_loss: float,
            w_b_loss: float,
            w_film_loss: float,
            n_add_edge: int = 1,
            n_add_node: int = 1,
            link_prediction_loss: Callable = F.binary_cross_entropy,
            node_generation_loss: Callable = utils.node_generation_loss,
            contrastive_loss: Union[Callable, List[Callable]] = None,
            discriminator_loss: Callable = nn.CrossEntropyLoss(),
            discriminator_tailgnn_loss: Callable = nn.BCELoss(),
            regularization_loss: Callable = torch.norm,
            low_degree_finetune_loss: Callable = utils.low_degree_specific_loss,
            task_loss: Callable = nn.CrossEntropyLoss(),
            writer: SummaryWriter = None,
            low_degree_additional_layer: bool = True,
            scale_and_shift: bool = False,
            forged_tail_node: bool = False,
            sp_loss: bool = False,
            b_loss: bool = False,
            missing_information_constraint: bool = False,
            add_edge: bool = False,
            add_node: bool = False,
            degree_discriminator: bool = False,
            contrastive0: bool = False,
            contrastive1: bool = False,
            regularization: bool = False,
            low_degree_finetune: bool = False,
        ):
        self.link_predictor = link_predictor
        self.discriminator = discriminator
        self.discriminator_tailgnn = discriminator_tailgnn
        self.classifier = classifier
        self.low_degree_specific_classifier = None
        self.link_prediction_lr = link_prediction_lr
        self.link_prediction_decay = link_prediction_decay
        self.classification_lr = classification_lr
        self.classification_decay = classification_decay
        self.low_degree_finetune_lr = low_degree_finetune_lr
        self.low_degree_finetune_decay = low_degree_finetune_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.w_node_generator_loss = w_node_generator_loss
        self.w_contrastive_loss = w_contrastive_loss
        self.w_discriminator_loss = w_discriminator_loss
        self.w_discriminator_tailgnn_loss = w_discriminator_tailgnn_loss
        self.w_regularization_loss = w_regularization_loss
        self.w_sp_loss = w_sp_loss
        self.w_b_loss = w_b_loss
        self.w_film_loss = w_film_loss
        self.w_missing_information_constraint = w_missing_information_constraint
        self.n_add_edge = n_add_edge
        self.n_add_node = n_add_node
        self.writer = writer

        self.link_prediction_loss = link_prediction_loss
        self.node_generation_loss = node_generation_loss
        self.discriminator_loss = discriminator_loss
        self.discriminator_tailgnn_loss = discriminator_tailgnn_loss
        self.regularization_loss = regularization_loss
        self.low_degree_finetune_loss = low_degree_finetune_loss
        self.task_loss = task_loss

        self.scale_and_shift = scale_and_shift
        self.forged_tail_node = forged_tail_node
        self.sp_loss = sp_loss
        self.b_loss = b_loss
        self.missing_information_constraint = missing_information_constraint
        self.add_edge = add_edge
        self.add_node = add_node
        self.degree_disc = degree_discriminator
        self.contrastive0 = contrastive0
        self.contrastive1 = contrastive1
        self.regularization = regularization
        self.low_degree_finetune = low_degree_finetune

        self.contrastive_loss = [None, None]
        if contrastive_loss is None:
            self.contrastive_loss[0] = utils.degree_contrastive_loss
            self.contrastive_loss[1] = utils.degree_contrastive_loss
        else:
            if isinstance(contrastive_loss, Callable):
                self.contrastive_loss[0] = contrastive_loss
                self.contrastive_loss[1] = contrastive_loss
            else:
                if contrastive_loss[0] is None:
                    self.contrastive_loss[0] = utils.degree_contrastive_loss
                else:
                    self.contrastive_loss[0] = contrastive_loss[0]
                if contrastive_loss[1] is None:
                    self.contrastive_loss[1] = utils.degree_contrastive_loss
                else:
                    self.contrastive_loss[1] = contrastive_loss[1]
        
        if writer is None:
            self.writer = SummaryWriter(log_dir='./logs')
        else:
            self.writer = writer

        self.is_edge_addition_enabled = (n_add_edge > 0 and link_predictor is not None and link_prediction_lr != 0 and self.add_edge)
        self.is_node_addition_enabled = (n_add_node > 0 and node_generation_loss is not None and w_node_generator_loss != 0 and self.add_node)
        self.is_contrastive_learning_enabled = [
            (self.w_contrastive_loss[0] != 0) and self.contrastive0,
            (self.w_contrastive_loss[1] != 0) and self.contrastive1,
        ]
        self.is_ldu_enabled = low_degree_additional_layer
        self.is_regularization_enabled = (self.w_regularization_loss != 0 and low_degree_additional_layer) and not self.regularization
        self.is_adversarial_learning_enabled = (self.w_discriminator_loss != 0 and self.discriminator_lr != 0 and self.discriminator is not None) and self.degree_disc
        self.is_low_degree_finetuning_enabled = (self.low_degree_finetune_lr > 0) and self.low_degree_finetune
        self.is_sp_loss_enabled = (self.w_sp_loss != 0) and self.sp_loss
        self.is_b_loss_enabled = (self.w_b_loss != 0) and self.b_loss
        self.is_film_loss_enabled = (self.w_film_loss != 0) and self.scale_and_shift
        self.is_missing_information_constraint_enabled = (self.w_missing_information_constraint != 0) and self.missing_information_constraint
        self.is_discriminator_tailgnn_enabled = (self.discriminator_tailgnn is not None and self.discriminator_tailgnn_loss is not None and self.w_discriminator_tailgnn_loss != 0) and self.forged_tail_node

    def load_state_dict(self, link_predictor, classifier, low_degree_specific_classifier):
        if self.link_predictor is not None:
            self.link_predictor.load_state_dict(link_predictor)
            self.best_link_predictor_state_dict = copy.deepcopy(self.link_predictor.state_dict())
        if self.classifier is not None:
            self.classifier.load_state_dict(classifier)
            self.classifier.set_hook()
            self.best_classifier_state_dict = copy.deepcopy(classifier)
        if low_degree_specific_classifier is not None:
            self.low_degree_specific_classifier = copy.deepcopy(self.classifier)
            self.low_degree_specific_classifier.load_state_dict(low_degree_specific_classifier)
            self.low_degree_specific_classifier.set_hook()
            self.best_low_degree_specific_classifier_state_dict = copy.deepcopy(low_degree_specific_classifier)

    def link_prediction_train(
            self,
            data,
            n_epoch: int,
            early_stopping_patience: int = None,
            ratio: float = 0.3,
        ):
        print('Link Prediction Training ...')
        optimizer = optim.Adam(self.link_predictor.parameters(),
                    lr=self.link_prediction_lr, weight_decay=self.link_prediction_decay)

        n_edge = data.adj.indices().shape[1]
        n_target_edge = int(n_edge*ratio)

        best_score = -float('inf')
        best_epoch = 0

        pbar = tqdm.tqdm(range(n_epoch))

        for i in pbar:
            self.link_predictor.train()
            optimizer.zero_grad()

            pbar.set_description('Epoch {}'.format(i))

            pos_link, neg_link = _get_link(data.adj, n_target_edge)

            pos_score = self.link_predictor(data.feat, data.adj, pos_link[0], pos_link[1])
            neg_score = self.link_predictor(data.feat, data.adj, neg_link[0], neg_link[1])

            label = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cuda()

            # Clip the score to avoid numerical instability
            link_score = (torch.cat([pos_score, neg_score]) + EPS) / (1 + 2*EPS)
            link_prediction_loss = self.link_prediction_loss(link_score, label)

            link_prediction_loss.backward()
            optimizer.step()

            score = self.link_prediction_validate(data, n_target_edge, i+1)
            self.writer.add_scalar('link_prediction_validation_score', score, i)
            if score > best_score:
                best_score = score
                best_epoch = i
                self.best_link_predictor_state_dict = copy.deepcopy(self.link_predictor.state_dict())
                self.writer.add_scalar('link_prediction_best_validation_score', best_score, i)
            pbar.set_postfix(
                link_prediction_loss = link_prediction_loss.item(),
                validation_score = score,
                best_score = best_score,
                best_epoch = best_epoch,
            )
            if early_stopping_patience is not None and i - best_epoch > early_stopping_patience:
                print('Early Stopping at epoch {}'.format(i))
                break
        return best_score

    def classification_train(
            self,
            data,
            n_epoch: int,
            early_stopping_patience: int = None,
        ):
        print('Classification Training ...')
        self.classifier.set_degree(data.low_degree_group, data.degree)
        optimizer = optim.Adam(self.classifier.parameters(),
                    lr=self.classification_lr, weight_decay=self.classification_decay)
        if self.is_adversarial_learning_enabled:
            optimizer_discr = optim.Adam(self.discriminator.parameters(),
                    lr=self.discriminator_lr, weight_decay=self.discriminator_decay)
        if self.is_discriminator_tailgnn_enabled:
            optimizer_discr_tailgnn = optim.Adam(self.discriminator_tailgnn.parameters(),
                    lr=self.discriminator_lr, weight_decay=self.discriminator_decay)

        mask = data.idx_train

        pbar = tqdm.tqdm(range(n_epoch))
        best_score = -float('inf')
        best_epoch = 0
        postfix = {}
        kwargs = {
            'd': data.degree,
            'idx': data.idx_train,
            'edge': data.edge_,
        }
        for i in pbar:
            pbar.set_description('Epoch {}'.format(i))
            optimizer.zero_grad()
            self.classifier.train()

            kwargs['head'] = True
            output, features = self.classifier(data.feat, data.adj_, **kwargs)

            discriminator_tailgnn_loss = 0
            if self.forged_tail_node:
                kwargs['head'] = False
                self.classifier.set_degree(data.low_degree_group, data.tail_adj_degree)
                output_t, _ = self.classifier(data.feat, data.tail_adj, **kwargs)
                assert not torch.isinf(output_t).any()
                if self.is_discriminator_tailgnn_enabled:
                    prob_t = self.discriminator_tailgnn(output_t)
                    assert not torch.isnan(prob_t).any()
                    discriminator_tailgnn_loss += self.discriminator_tailgnn_loss(prob_t, torch.zeros_like(prob_t))
                    postfix['Ld_tailgnn'] = discriminator_tailgnn_loss.item()
                    self.writer.add_scalar('discriminator_tailgnn_loss', discriminator_tailgnn_loss.item(), i)

            if 'b' not in features.keys():
                self.is_b_loss_enabled = False
            if 'film' not in features.keys():
                self.is_film_loss_enabled = False
            if 'relation_output' not in features.keys():
                self.is_missing_information_constraint_enabled = False

            # Task Loss
            task_loss = self.task_loss(output[mask], data.labels[mask])
            if self.forged_tail_node:
                task_loss += self.task_loss(output_t[mask], data.labels[mask])
                task_loss = task_loss / 2
            postfix['Lt'] = task_loss.item()
            self.writer.add_scalar('task_loss', task_loss.item(), i)

            # DegFair loss
            sp_loss = 0
            b_loss = 0
            film_loss = 0
            missing_information_constraint_loss = 0
            if self.is_sp_loss_enabled:
                sp_loss = self._calc_sp_loss(output, data, mask)
                postfix['Lsp'] = sp_loss.item()
                self.writer.add_scalar('sp_loss', sp_loss.item(), i)
            if self.is_b_loss_enabled:
                b_loss += features['b']
                postfix['b'] = features['b'].item()
                self.writer.add_scalar('b', features['b'].item(), i)
            if self.is_film_loss_enabled:
                film_loss += features['film']
                postfix['film'] = features['film'].item()
                self.writer.add_scalar('film', features['film'].item(), i)
            if self.is_missing_information_constraint_enabled:
                missing_information_constraint_loss += normalize_output(features['relation_output'], mask)
                postfix['Lmic'] = missing_information_constraint_loss.item()
                self.writer.add_scalar('missing_information_constraint_loss', missing_information_constraint_loss.item(), i)

            # Discriminator loss
            discriminate_loss = 0
            if self.is_adversarial_learning_enabled and 'discr_target' in features.keys():
                self.discriminator.eval()
                discr_output = self.discriminator(features['discr_target'], data.adj_)
                for j in range(self.n_class):
                    label_mask = (data.labels[mask] == j)
                    discriminate_loss += self.discriminator_loss(discr_output[mask][label_mask], data.degree_label[mask][label_mask])
                postfix['Ld'] = discriminate_loss.item()
                self.writer.add_scalar('discriminate_loss', discriminate_loss.item(), i)

            # Node Generator Loss
            node_generation_loss = 0
            regularization_loss = 0
            if self.is_node_addition_enabled:
                node_generation_loss = self.node_generation_loss(features['truth_emb'], features['predicted_emb'])
                postfix['Lng'] = node_generation_loss.item()
                self.writer.add_scalar('node_generation_loss', node_generation_loss.item(), i)
                if self.is_regularization_enabled:
                    regularization_loss += self.regularization_loss(features['predicted_emb'])

            # Contrastive Loss(1)
            contrastive_loss = [0, 0]
            if self.is_contrastive_learning_enabled[0] and features['contrastive_target'][0] is not None:
                contrastive_loss[0] += self.contrastive_loss[0](features['contrastive_target'][0][mask], data.labels[mask], data.degree[mask])
                postfix['Lc1'] = contrastive_loss[0].item()
                self.writer.add_scalar('contrastive_loss1', contrastive_loss[0].item(), i)

            # regularization loss
            if self.is_regularization_enabled:
                regularization_loss += self.regularization_loss(features['regularization_target'][mask])
                postfix['Lr'] = regularization_loss.item()
                self.writer.add_scalar('regularization_loss', regularization_loss.item(), i)

            # Contrastive Loss(2)
            if self.is_contrastive_learning_enabled[1] and features['contrastive_target'][1] is not None:
                contrastive_loss[1] += self.contrastive_loss[1](features['contrastive_target'][1][mask], data.labels[mask], data.degree[mask])
                postfix['Lc2'] = contrastive_loss[1].item()
                self.writer.add_scalar('contrastive_loss2', contrastive_loss[1].item(), i)

            loss = task_loss \
                    + (self.w_sp_loss * sp_loss) \
                    + (self.w_b_loss * b_loss) \
                    + (self.w_film_loss * film_loss) \
                    + (self.w_contrastive_loss[0] * contrastive_loss[0]) \
                    + (self.w_contrastive_loss[1] * contrastive_loss[1]) \
                    + (self.w_node_generator_loss * node_generation_loss) \
                    + (self.w_regularization_loss * regularization_loss) \
                    + (self.w_missing_information_constraint * missing_information_constraint_loss) \
                    - (self.w_discriminator_loss * discriminate_loss) \
                    - (self.w_discriminator_tailgnn_loss * discriminator_tailgnn_loss) \

            loss.backward(retain_graph=True)
            optimizer.step()
            postfix['L'] = loss.item()
            self.writer.add_scalar('total_loss', loss.item(), i)

            # Discriminator Train
            if self.is_adversarial_learning_enabled and 'discr_target' in features.keys():
                self.discriminator.train()
                optimizer_discr.zero_grad()
                discr_output = self.discriminator(features['discr_target'].detach(), data.adj_)
                for j in range(self.n_class):
                    label_mask = (data.labels[mask] == j)
                    discriminate_loss += self.discriminator_loss(discr_output[mask][label_mask], data.degree_label[mask][label_mask])
                discriminate_loss.backward(retain_graph=True)
                optimizer_discr.step()

            # Discriminator TailGNN Train
            if self.is_discriminator_tailgnn_enabled:
                self.discriminator_tailgnn.train()
                optimizer_discr_tailgnn.zero_grad()

                prob_h = self.discriminator_tailgnn(output)
                prob_t = self.discriminator_tailgnn(output_t)
                discriminator_tailgnn_loss = self.discriminator_tailgnn_loss(prob_h, torch.ones_like(prob_h))
                discriminator_tailgnn_loss += self.discriminator_tailgnn_loss(prob_t, torch.zeros_like(prob_t))
                discriminator_tailgnn_loss.backward()
                optimizer_discr_tailgnn.step()

            score = self.classification_validate(data, i)
            postfix['score'] = score
            self.writer.add_scalar('classification_validation_score', score, i)
            if best_score < score:
                best_score = score
                best_epoch = i
                self.best_classifier_state_dict = copy.deepcopy(self.classifier.state_dict())
                postfix['best_score'] = best_score
                postfix['best_ep'] = best_epoch
                self.writer.add_scalar('classification_validation_best_score', best_score, i)

            pbar.set_postfix(**postfix)
            if early_stopping_patience is not None and i - best_epoch > early_stopping_patience:
                print('Early Stopping at epoch {}'.format(i))
                break

        return best_score

    def finetune(
            self,
            data,
            n_epoch: int = 500,
            early_stopping_patience: int = None,
        ):
        print('Low Degree Finetuning ...')
        self.low_degree_specific_classifier = copy.deepcopy(self.classifier)
        self.low_degree_specific_classifier.set_hook()
        optimizer = optim.Adam(self.low_degree_specific_classifier.parameters(),
                    lr=self.low_degree_finetune_lr, weight_decay=self.low_degree_finetune_decay)
        mask = data.idx_train

        pbar = tqdm.tqdm(range(n_epoch))
        best_score = -float('inf')
        best_epoch = 0
        postfix = {}
        kwargs = {
            'd': data.degree,
            'idx': data.idx_train,
            'edge': data.edge_,
            'head': False,
        }
        for i in pbar:
            pbar.set_description('Epoch {}'.format(i))
            optimizer.zero_grad()
            self.low_degree_specific_classifier.train()
            output, _ = self.low_degree_specific_classifier(data.feat, data.adj_, **kwargs)

            assert not torch.isnan(output).any()

            # Task Loss
            # low_degree_finetune_loss = self.low_degree_finetune_loss(output[mask], data.labels[mask], data.degree_label[mask])
            # low_degree_finetune_loss = self.task_loss(output[mask][data.degree_label[mask] == 0], data.labels[mask][data.degree_label[mask] == 0])
            low_degree_finetune_loss = self.task_loss(output[mask], data.labels[mask])
            postfix['LDF_loss'] = low_degree_finetune_loss.item()
            self.writer.add_scalar('low_degree_finetune_loss', low_degree_finetune_loss.item(), i)

            low_degree_finetune_loss.backward()
            optimizer.step()

            score = self.classification_validate(data, i)
            postfix['val_score'] = score
            self.writer.add_scalar('low_degree_finetune_validation_score', score, i)
            if best_score < score:
                best_score = score
                best_epoch = i
                self.best_low_degree_specific_classifier_state_dict = copy.deepcopy(self.low_degree_specific_classifier.state_dict())
                postfix['best_val_score'] = best_score
                postfix['best_epoch'] = best_epoch
                self.writer.add_scalar('low_degree_finetune_validation_best_score', best_score, i)
            
            pbar.set_postfix(**postfix)
            if early_stopping_patience is not None and i - best_epoch > early_stopping_patience:
                print('Early Stopping at epoch {}'.format(i))
                break

        return best_score

    def train(
            self,
            data,
            n_epoch: Union[int, List[int]] = 500,
            early_stopping_patience: Union[int, List[int]] = 100,
        ):
        self.n_class = len(torch.unique(data.labels))

        if isinstance(n_epoch, int):
            n_epoch_link_prediction = n_epoch
            n_epoch_classification = n_epoch
            n_epoch_low_degree_finetune = n_epoch
        else:
            n_epoch_link_prediction = n_epoch[0]
            n_epoch_classification = n_epoch[1]
            n_epoch_low_degree_finetune = n_epoch[2]

        if isinstance(early_stopping_patience, int):
            early_stopping_patience_link_prediction = early_stopping_patience
            early_stopping_patience_classification = early_stopping_patience
            early_stopping_patience_low_degree_finetune = early_stopping_patience
        else:
            early_stopping_patience_link_prediction = early_stopping_patience[0]
            early_stopping_patience_classification = early_stopping_patience[1]
            early_stopping_patience_low_degree_finetune = early_stopping_patience[2]


        if self.is_edge_addition_enabled:
            self.link_prediction_train(data, n_epoch_link_prediction, early_stopping_patience_link_prediction)

            # add edge
            if self.is_edge_addition_enabled:
                self.best_link_predictor = copy.deepcopy(self.link_predictor)
                self.best_link_predictor.load_state_dict(self.best_link_predictor_state_dict)
                self.best_link_predictor.eval()
                with torch.no_grad():
                    score = self.best_link_predictor(data.feat, data.adj, data.low_degree_group, torch.arange(data.feat.shape[0]), all_pair=True)
                    data.adj_ = utils.add_edge(data.adj, data.low_degree_group, score, self.n_add_edge)
                    data.edge_ = data.adj_.indices()

        data.tail_adj, data.tail_adj_degree = create_tail_adj(data.adj_, data.degree, data.idx_train, 5, self.classifier.rng)

        self.classification_train(data, n_epoch_classification, early_stopping_patience_classification)

        if self.is_low_degree_finetuning_enabled:
            self.finetune(data, n_epoch_low_degree_finetune, early_stopping_patience_low_degree_finetune)

    def link_prediction_validate(self, data, n_target_edge, epoch):
        self.link_predictor.eval()
        with torch.no_grad():
            pos_link, neg_link = _get_link(data.adj, n_target_edge)
            pos_score = self.link_predictor(data.feat, data.adj, pos_link[0], pos_link[1])
            neg_score = self.link_predictor(data.feat, data.adj, neg_link[0], neg_link[1])
            label = [1]*len(pos_score) + [0]*len(neg_score)

            label = torch.tensor(label).cuda()
            # link_prediction_loss = self.link_prediction_loss(torch.cat([pos_score, neg_score]), label)
            preds = torch.cat([torch.where(pos_score >= 0.5, 1, 0), torch.where(neg_score >= 0.5, 1, 0)])
            acc = binary_accuracy(preds, label)
            precision = binary_precision(preds, label)
            recall = binary_recall(preds, label)
            f1 = binary_f1_score(preds, label)

            self.writer.add_scalar('link_prediction_validation_accuracy', acc, epoch)
            self.writer.add_scalar('link_prediction_validation_precision', precision, epoch)
            self.writer.add_scalar('link_prediction_validation_recall', recall, epoch)
            self.writer.add_scalar('link_prediction_validation_f1', f1, epoch)

        return f1.item()

    def classification_validate(self, data, epoch):
        acc, out1 = self.validate_light(data)

        self.writer.add_scalar('classification_validation_accuracy', acc, epoch)
        self.writer.add_scalar('classification_validation_dsp', out1['mu_sp'], epoch)
        self.writer.add_scalar('classification_validation_deo', out1['mu_eo'], epoch)

        return acc - out1['mu_sp'] / 16 - out1['mu_eo'] / 16


    def validate_light(self, data, best_model=False):
        mask = data.idx_val
        if best_model:
            self.best_classifier = copy.deepcopy(self.classifier)
            self.best_classifier.set_hook()
            self.best_classifier.load_state_dict(self.best_classifier_state_dict)
            self.best_classifier.eval()
        else:
            self.classifier.eval()
        with torch.no_grad():
            if best_model:
                self.best_classifier.set_degree(data.low_degree_group, data.degree)
                kwargs = {
                    'd': data.degree,
                    'idx': data.idx_val,
                    'edge': data.edge_,
                    'head': False,
                }
                output, _ = self.best_classifier(data.feat, data.adj_, **kwargs)
            else:
                self.classifier.set_degree(data.low_degree_group, data.degree)
                kwargs = {
                    'd': data.degree,
                    'idx': data.idx_val,
                    'edge': data.edge_,
                    'head': False,
                }
                output, _ = self.classifier(data.feat, data.adj_, **kwargs)
            if self.is_low_degree_finetuning_enabled and self.low_degree_specific_classifier is not None:
                if best_model:
                    self.best_low_degree_specific_classifier = copy.deepcopy(self.low_degree_specific_classifier)
                    self.best_low_degree_specific_classifier.load_state_dict(self.best_low_degree_specific_classifier_state_dict)
                    self.best_low_degree_specific_classifier.set_hook()
                    self.best_low_degree_specific_classifier.eval()
                    self.best_low_degree_specific_classifier.set_degree(data.low_degree_group, data.degree)
                    low_degree_output, _ = self.best_low_degree_specific_classifier(data.feat, data.adj_, **kwargs)
                else:
                    self.low_degree_specific_classifier.eval()
                    self.low_degree_specific_classifier.set_degree(data.low_degree_group, data.degree)
                    low_degree_output, _ = self.low_degree_specific_classifier(data.feat, data.adj_, **kwargs)
                output[data.low_degree_group] = low_degree_output[data.low_degree_group]
            preds = output.max(1)[1].cpu().detach()

            acc = utils.accuracy(output[mask], data.labels[mask])
            acc = acc.cpu()

            out1 = utils.evaluate_fairness_light(preds, data, data.group1)

        return float(acc.detach().cpu())*100, out1

    def validate(self, data):
        mask = data.idx_val
        return self._test(data, mask)

    def test(self, data, degree_range=30):
        print('Testing ...')
        mask = data.idx_test
        return self._test(data, mask, degree_range)

    def _test(self, data, mask, degree_range=30):
        degree = np.array(data.degree.cpu(), dtype='uint16').flatten()
        self.best_classifier = copy.deepcopy(self.classifier)
        self.best_classifier.load_state_dict(self.best_classifier_state_dict)
        self.best_classifier.set_hook()
        self.best_classifier.eval()
        with torch.no_grad():
            if self.is_edge_addition_enabled:
                self.best_link_predictor = copy.deepcopy(self.link_predictor)
                self.best_link_predictor.load_state_dict(self.best_link_predictor_state_dict)
                self.best_link_predictor.eval()
                score = self.best_link_predictor(data.feat, data.adj, data.low_degree_group, torch.arange(data.feat.shape[0]), all_pair=True)
                data.adj_ = utils.add_edge(data.adj, data.low_degree_group, score, self.n_add_edge)
                data.edge_ = data.adj_.indices()
            self.best_classifier.set_degree(data.low_degree_group, data.degree)
            kwargs = {
                'd': data.degree,
                'idx': data.idx_test,
                'edge': data.edge_,
                'head': False,
            }
            output, _ = self.best_classifier(data.feat, data.adj_, **kwargs)
            if self.is_low_degree_finetuning_enabled:
                self.best_low_degree_specific_classifier = copy.deepcopy(self.low_degree_specific_classifier)
                self.best_low_degree_specific_classifier.load_state_dict(self.best_low_degree_specific_classifier_state_dict)
                self.best_low_degree_specific_classifier.set_hook()
                self.best_low_degree_specific_classifier.eval()
                self.best_low_degree_specific_classifier.set_degree(data.low_degree_group, data.degree)
                low_degree_output, _ = self.best_low_degree_specific_classifier(data.feat, data.adj_, **kwargs)
                output[data.low_degree_group] = low_degree_output[data.low_degree_group]
            preds = output.max(1)[1].cpu().detach()
            # acc = multiclass_accuracy(preds[mask], data.labels[mask], self.n_class).item()
            # precision = multiclass_precision(preds[mask], data.labels[mask], self.n_class, average='macro').item()
            # recall = multiclass_recall(preds[mask], data.labels[mask], self.n_class, average='macro').item()
            # f1 = multiclass_f1_score(preds[mask], data.labels[mask], self.n_class, average='macro').item()

            acc = utils.accuracy(output[mask], data.labels[mask])
            acc = acc.cpu()

            macf = f1_score(data.labels[mask].cpu(), preds[mask], average='macro')
            weif = f1_score(data.labels[mask].cpu(), preds[mask], average='weighted')

            degree_acc, degree_macf, degree_weif = utils.metrics_by_degree(preds, data, data.group1, degree_range)

            out1 = utils.evaluate_fairness(preds, data, data.group1, embed=output[mask])
            out2 = None
            print('Accuracy={:.4f}'.format(acc))
            print('Macro-F1={:.4f}'.format(macf))

        return acc*100, macf*100, weif*100, out1, out2, degree_acc, degree_macf, degree_weif, preds[mask], data.labels[mask], degree[mask]


    def _calc_sp_loss(self, output, data, mask):
        assert not torch.isnan(output).any()

        train_output = output[mask]
        mean = torch.mean(data.degree[mask].float())

        idx_low = torch.where(data.degree[mask] < mean)[0]
        idx_high = torch.where(data.degree[mask] >= mean)[0]

        low_embed = torch.mean(train_output[idx_low], dim=0)
        high_embed = torch.mean(train_output[idx_high], dim=0)


        sp_loss = F.mse_loss(low_embed, high_embed)

        low_eo_embed = torch.zeros(data.labels.max()+1).cuda()
        high_eo_embed = torch.zeros(data.labels.max()+1).cuda()


        train_label = data.labels[mask]
        for i in range(data.labels.max()+1):
            idx_lc = torch.where(train_label[idx_low] == i)[0]
            idx_hc = torch.where(train_label[idx_high] == i)[0]

            mean_l = torch.mean(train_output[idx_lc], dim=0)
            mean_h = torch.mean(train_output[idx_hc], dim=0)

            low_eo_embed[i] = mean_l[i]
            high_eo_embed[i] = mean_h[i]
        # eo_loss = F.mse_loss(low_eo_embed, high_eo_embed)

        return sp_loss

    def get_best_model(self):
        if self.is_edge_addition_enabled and hasattr(self, 'best_link_predictor_state_dict'):
            best_link_predictor_state_dict = self.best_link_predictor_state_dict
        else:
            best_link_predictor_state_dict = None

        if hasattr(self, 'best_classifier_state_dict'):
            best_classifier_state_dict = self.best_classifier_state_dict
        else:
            best_classifier_state_dict = None

        if self.is_low_degree_finetuning_enabled and hasattr(self, 'best_low_degree_specific_classifier_state_dict'):
            best_low_degree_specific_classifier_state_dict = self.best_low_degree_specific_classifier_state_dict
        else:
            best_low_degree_specific_classifier_state_dict = None
        ret = {
            'link_predictor': best_link_predictor_state_dict,
            'classifier': best_classifier_state_dict,
            'low_degree_specific_classifier': best_low_degree_specific_classifier_state_dict,
        }
        return ret


class DegFairGNNController(Controller):
    def __init__(self, model_dic, loss, lr, decay, w_f, w_b):
        super().__init__(model_dic, loss, lr, decay)
        self.w_f = w_f
        self.w_b = w_b
        self.loss_lst = {'L_cls': [],'sp_loss': [], 'b': [], 'film': []}

    def inference_and_calc_loss(self, data, train=True, save_loss=False):
        if train:
            mask = data.idx_train
        else:
            mask = data.idx_val

        output, b, film = self.model(data.feat, data.adj, data.degree, mask, data.edge)
        L_cls, sp_loss = self.calc_L_cls_sp_loss(output, data, mask)

        if save_loss:
            self.loss_lst['L_cls'].append(L_cls)
            self.loss_lst['sp_loss'].append(sp_loss)
            self.loss_lst['b'].append(b)
            self.loss_lst['film'].append(film)

        # no L3
        # b = 0
        loss = L_cls + (self.w_f * sp_loss) + (self.w_b * b) + (self.w_b * film)

        return loss

    def inference_and_calc_val_score(self, data):
        return self.inference_and_calc_loss(data, train=False)

    def get_output_for_test(self, data):
        output, _, _ = self.model(data.feat, data.adj, data.degree, data.idx_test, data.edge)
        return output

    def calc_L_cls_sp_loss(self, output, data, mask):
        assert not torch.isnan(output).any()

        train_output = output[mask]
        mean = torch.mean(data.degree[mask].float())

        idx_low = torch.where(data.degree[mask] < mean)[0]
        idx_high = torch.where(data.degree[mask] >= mean)[0]

        low_embed = torch.mean(train_output[idx_low], dim=0)
        high_embed = torch.mean(train_output[idx_high], dim=0)


        sp_loss = F.mse_loss(low_embed, high_embed)

        low_eo_embed = torch.zeros(data.labels.max()+1).cuda()
        high_eo_embed = torch.zeros(data.labels.max()+1).cuda()


        train_label = data.labels[mask]
        for i in range(data.labels.max()+1):
            idx_lc = torch.where(train_label[idx_low] == i)[0]
            idx_hc = torch.where(train_label[idx_high] == i)[0]

            mean_l = torch.mean(train_output[idx_lc], dim=0)
            mean_h = torch.mean(train_output[idx_hc], dim=0)

            low_eo_embed[i] = mean_l[i]
            high_eo_embed[i] = mean_h[i]
        # eo_loss = F.mse_loss(low_eo_embed, high_eo_embed)

        L_cls = self.calc_task_loss(output[mask], data.labels[mask], data.degree_label[mask])

        return L_cls, sp_loss


class RandomModelController(object):
    def __init__(self, model):
        self.model = model['main']
        self.need_pretrain = False

    def train(self, data, n_epoch, flexible_epoch, pretrain=False, aftrain=False):
        return []

    def test(self, data, degree_range, enable_deg2):
        print('Testing ...')
        degree = np.array(data.degree.cpu(), dtype='uint16').flatten()

        mask = data.idx_test
        # accuracy
        output = self.model(data.feat, data.adj)
        acc = utils.accuracy(output[mask], data.labels[mask])
        acc = acc.cpu()

        preds = output.max(1)[1].cpu().detach()
        macf = f1_score(data.labels[mask].cpu(), preds[mask], average='macro')
        weif = f1_score(data.labels[mask].cpu(), preds[mask], average='weighted')

        degree_acc, degree_macf, degree_weif = utils.metrics_by_degree(preds, data, data.group1, degree_range)

        out1 = utils.evaluate_fairness(preds, data, data.group1, embed=output[mask])
        out2 = None
        if enable_deg2:
            out2 = utils.evaluate_fairness(preds, data, data.group2)
        print('Accuracy={:.4f}'.format(acc))
        print('Macro-F1={:.4f}'.format(macf))

        return acc*100, macf*10, weif*100, out1, out2, degree_acc, degree_macf, degree_weif, preds[mask], data.labels[mask], degree[mask]
