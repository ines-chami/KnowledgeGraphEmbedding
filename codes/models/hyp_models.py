#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import KGEModel

from manifolds import ManifoldParameter, Poincare
from utils.euc_utils import householder_reflection, householder_rotation


class KGEModelH(KGEModel):
    """
    Hyperbolic knowledge graph embedding model
    """

    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, p_norm,
                 dropout, entity_embedding_multiple, relation_embedding_multiple):
        super(KGEModelH, self).__init__(model_name, nentity, nrelation, hidden_dim, gamma, p_norm, dropout)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.manifold = Poincare()

        # Initialize entity hyperbolic embeddings
        self.entity_embedding = ManifoldParameter(manifold=self.manifold,
                                                  data=torch.zeros(nentity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Initialize relation center as hyperbolic embeddings
        self.relation_center = ManifoldParameter(manifold=self.manifold,
                                                 data=torch.zeros(nrelation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_center,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Reflection and rotation parameters are Euclidean vector
        self.relation_transforms = nn.Parameter(
            data=torch.zeros(nrelation, hidden_dim * (relation_embedding_multiple - 1)))
        nn.init.xavier_uniform(self.relation_transforms.data)

        if model_name in ['RotationH'] and (relation_embedding_multiple != 3 or entity_embedding_multiple != 1):
            raise ValueError('RotationE should triple relationship embeddings (center and two reflections)')

        if model_name in ['ReflectionH'] and (relation_embedding_multiple != 2 or entity_embedding_multiple != 1):
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

            relation_center = torch.index_select(
                self.relation_center,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            relation_transforms = torch.index_select(
                self.relation_transforms,
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

            relation_center = torch.index_select(
                self.relation_center,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            relation_transforms = torch.index_select(
                self.relation_transforms,
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

            relation_center = torch.index_select(
                self.relation_center,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            relation_transforms = torch.index_select(
                self.relation_transforms,
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
            'TranslationH': self.TranslationH,
            'ReflectionH': self.ReflectionH,
            'RotationH': self.RotationH,
        }

        if self.model_name in model_func:
            head = F.dropout(head, self.dropout, training=self.training)
            relation_center = F.dropout(relation_center, self.dropout, training=self.training)
            relation_center = F.dropout(relation_center, self.dropout, training=self.training)
            relation_transforms = F.dropout(relation_transforms, self.dropout, training=self.training)
            score = model_func[self.model_name](head, relation_center, relation_transforms, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TranslationH(self, head, relation_center, relation_transforms, tail, mode):
        '''
        Hyperbolic translation model
        '''
        if mode == 'head-batch':
            score = self.manifold.distance(self.manifold.add(-relation_center, tail), head)
        else:
            score = self.manifold.distance(self.manifold.add(head, relation_center), tail)
        return self.gamma.item() - score

    def RotationH(self, head, relation_center, relation_transforms, tail, mode):
        '''
        Hyperbolic rotation model with real numbers using two Householder reflections
        '''
        raise NotImplementedError

    def ReflectionH(self, head, relation_center, relation_transforms, tail, mode):
        '''
        Hyperbolic reflection model using one Householder reflection
        '''
        raise NotImplementedError
