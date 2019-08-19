#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.base import KGEModel
from utils.math_utils import householder_reflection, householder_rotation
import pdb


class O2MEKGEModel(KGEModel):
    """
    Euclidean one to many knowledge graph embedding model
    """

    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, p_norm,
                 dropout, entity_embedding_multiple, relation_embedding_multiple, nsiblings, rho):
        super(O2MEKGEModel, self).__init__(model_name, nentity, nrelation, hidden_dim, gamma, p_norm, dropout)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * entity_embedding_multiple
        self.relation_dim = hidden_dim * relation_embedding_multiple

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        # TODO: try xavier
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        #added num of siblings, rho
        self.nsiblings = nsiblings
        self.nsiblings_vec = torch.tensor([i for i in range(-nsiblings, nsiblings+1, 1)]).view(1,-1,1).type(torch.FloatTensor).cuda()
        self.rho = rho

        if model_name == 'One2ManyTransE' and relation_embedding_multiple - 2 * entity_embedding_multiple != 0:
            raise ValueError('One2ManyTransE should double relationship embeddings (center and sibling)')

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'One2ManyTransE': self.One2ManyTransE
        }

        if self.model_name in model_func:
            head = F.dropout(head, self.dropout, training=self.training)
            relation = F.dropout(relation, self.dropout, training=self.training)
            tail = F.dropout(tail, self.dropout, training=self.training)
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=self.p_norm, dim=2)
        return score

    def One2ManyTransE(self, head, relation, tail, mode):
        '''
        Euclidean one-to-many model using two translations
        '''
        center, sibling = torch.chunk(relation, 2, dim=2)
        if mode == 'head-batch':

            head_pred_v = (tail - center) + torch.matmul(self.nsiblings_vec, sibling)
            head_T_head = torch.matmul(head.unsqueeze(2), head.unsqueeze(-1)).squeeze(-1)
            head_pred_v_T_head_pred_v =  torch.matmul(head_pred_v.unsqueeze(2), head_pred_v.unsqueeze(-1)).squeeze(-1).squeeze(-1).unsqueeze(1)
            head_T_head_pred_v = torch.matmul(head, head_pred_v.view(-1, self.entity_dim, 2*self.nsiblings +1))

            scores_l_2 = -self.rho * (head_T_head + head_pred_v_T_head_pred_v - 2 * head_T_head_pred_v)

        else:
            #pdb.set_trace()
            # sibling, head, center (B,1,d)
            # nsiblings_vec (1, 2N+1, 1)
            # tail (B, Ne, d)

            tail_pred_v = (head + center) + torch.matmul(self.nsiblings_vec, sibling) #(B, 2N+1, d)
            tail_T_tail = torch.matmul(tail.unsqueeze(2), tail.unsqueeze(-1)).squeeze(-1) #(B, Ne, 1)
            tail_pred_v_T_tail_pred_v = torch.matmul(tail_pred_v.unsqueeze(2), tail_pred_v.unsqueeze(-1)).squeeze(-1).squeeze(-1).unsqueeze(1) #(B, 1, 2N+1)
            tail_T_tail_pred_v =  torch.matmul(tail, tail_pred_v.view(-1, self.entity_dim, 2*self.nsiblings +1)) #(B, Ne, 2N+1)

            scores_l_2 = -self.rho*(tail_T_tail+ tail_pred_v_T_tail_pred_v -2*tail_T_tail_pred_v)
        return self.gamma.item() + (1/self.rho) * torch.logsumexp(scores_l_2, dim=-1)




