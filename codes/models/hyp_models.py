#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import KGEModel

from manifolds import ManifoldParameter, Poincare
from utils.math_utils import householder_reflection, householder_rotation


class HKGEModel(KGEModel):
    """
    Hyperbolic knowledge graph embedding model
    """

    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, p_norm,
                 dropout, entity_embedding_multiple, relation_embedding_multiple):
        super(HKGEModel, self).__init__(model_name, nentity, nrelation, hidden_dim, gamma, p_norm, dropout)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.manifold = Poincare()
        self.entity_dim = hidden_dim * entity_embedding_multiple
        self.relation_dim = hidden_dim
        self.entity_embedding = ManifoldParameter(manifold=self.manifold,
                                                  data=torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = ManifoldParameter(manifold=self.manifold,
                                                    data=torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name in ['RotationH'] and relation_embedding_multiple - 3 * entity_embedding_multiple != 0:
            raise ValueError('RotationE should triple relationship embeddings (center and two reflections)')

        if model_name in ['ReflectionH'] and relation_embedding_multiple - 2 * entity_embedding_multiple != 0:
            raise ValueError('ReflectionE should double relationship embeddings (center and one reflection)')


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
        'DistMult': self.DistMult,
        'ComplEx': self.ComplEx,
        'RotatE': self.RotatE,
        'pRotatE': self.pRotatE,
        'ReflectionE': self.ReflectionE,
        'RotationE': self.RotationE,
        'TranslationH': self.TranslationH,
        'ReflectionH': self.ReflectionH,
        'RotationH': self.RotationH,
    }

    if self.model_name in model_func:
        head = F.dropout(head, self.dropout, training=self.training)
        relation = F.dropout(relation, self.dropout, training=self.training)
        tail = F.dropout(tail, self.dropout, training=self.training)
        score = model_func[self.model_name](head, relation, tail, mode)
    else:
        raise ValueError('model %s not supported' % self.model_name)

    return score


def TranslationH(self, head, relation, tail, mode):
    '''
    Hyperbolic translation model
    '''
    if mode == 'head-batch':
        score = head + (relation - tail)
    else:
        score = (head + relation) - tail

    score = self.gamma.item() - torch.norm(score, p=self.p_norm, dim=2)
    return score


def RotationH(self, head, relation, tail, mode):
    '''
    Hyperbolic rotation model with real numbers using two Householder reflections
    '''
    center, v1, v2 = torch.chunk(relation, 3, dim=2)
    prediction = householder_rotation(head - center, v1, v2) + center
    score = self.gamma.item() - torch.norm(prediction - tail, p=self.p_norm, dim=2)
    return score


def ReflectionH(self, head, relation, tail, mode):
    '''
    Hyperbolic reflection model using one Householder reflection
    '''
    center, v = torch.chunk(relation, 2, dim=2)
    prediction = householder_reflection(head - center, v) + center
    score = self.gamma.item() - torch.norm(prediction - tail, p=self.p_norm, dim=2)
    return score
