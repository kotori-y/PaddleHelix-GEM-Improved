#!/usr/bin/python                                                                                                
# -*-coding:utf-8-*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
gnn network
"""

import paddle
import paddle.nn as nn
import pgl

from pahelix.networks.basic_block import MLP


class DownstreamModel(nn.Layer):
    """
    Docstring for DownstreamModel,it is an supervised 
    GNN model which predicts the tasks shown in num_tasks and so on.
    """
    def __init__(self, model_config, compound_encoder, use_gem=False):
        super(DownstreamModel, self).__init__()
        self.task_type = model_config['task_type']
        self.num_tasks = model_config['num_tasks']

        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(compound_encoder.graph_dim)
        self.mlp = MLP(
            model_config['layer_num'],
            in_size=compound_encoder.graph_dim,
            hidden_size=model_config['hidden_size'],
            out_size=self.num_tasks,
            act=model_config['act'],
            dropout_rate=model_config['dropout_rate'])
        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()

        self.use_gem = use_gem

    def forward(self, atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs):
        """
        Define the forward function,set the parameter layer options.compound_encoder 
        creates a graph data holders that attributes and features in the graph.
        Returns:
            pred: the model prediction.
        """
        if not self.use_gem:
            _, _, graph_repr = \
                self.compound_encoder(atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs)
        else:
            _, _, graph_repr = \
                self.compound_encoder(atom_bond_graphs, bond_angle_graphs)

        graph_repr = self.norm(graph_repr)
        pred = self.mlp(graph_repr)
        if self.task_type == 'class':
            pred = self.out_act(pred)
        return pred


class DownstreamLocalModel(nn.Layer):
    """
    Docstring for DownstreamModel,it is a supervised
    GNN model which predicts the tasks shown in num_tasks and so on.
    """

    def __init__(self, model_config, compound_encoder, use_gem=False):
        super(DownstreamLocalModel, self).__init__()

        self.task_type = model_config['task_type']
        self.num_tasks = model_config['num_tasks']

        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(compound_encoder.graph_dim)

        if self.atom_task:
            self.atom_mlp = MLP(
                model_config['layer_num'],
                in_size=compound_encoder.graph_dim,
                hidden_size=model_config['hidden_size'],
                out_size=self.num_tasks,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate']
            )

        if self.bond_task:
            self.bond_mlp = MLP(
                model_config['layer_num'],
                in_size=compound_encoder.graph_dim,
                hidden_size=model_config['hidden_size'],
                out_size=self.num_tasks,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate']
            )

        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()

        self.use_gem = use_gem

    def forward(self, atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs):
        """
        Define the forward function,set the parameter layer options.compound_encoder
        creates a graph data holders that attributes and features in the graph.
        Returns:
            pred: the model prediction.
        """
        if not self.use_gem:
            node_repr, edge_repr, _ = \
                self.compound_encoder(atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs)
        else:
            node_repr, edge_repr, _ = \
                self.compound_encoder(atom_bond_graphs, bond_angle_graphs)

        node_repr = self.norm(node_repr)
        edge_repr = self.norm(edge_repr)

        atom_pred = self.mlp(node_repr)
        if self.task_type == 'class':
            atom_pred = self.out_act(atom_pred)

        edge_pred = self.mlp(edge_repr)
        if self.task_type == 'class':
            edge_pred = self.out_act(edge_pred)

        return atom_pred, edge_pred
