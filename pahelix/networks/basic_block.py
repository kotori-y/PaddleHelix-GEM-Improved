#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
Some frequently used basic blocks
"""


import paddle
import paddle.nn as nn


class Activation(nn.Layer):
    """
    Activation
    """
    def __init__(self, act_type, **params):
        super(Activation, self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(**params)
        else:
            raise ValueError(act_type)
     
    def forward(self, x):
        """tbd"""
        return self.act(x)


class MLP(nn.Layer):
    """
    MLP
    """
    def __init__(self, layer_num, in_size, hidden_size, out_size, act, dropout_rate):
        super(MLP, self).__init__()

        layers = []
        for layer_id in range(layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            elif layer_id < layer_num - 1:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            else:
                layers.append(nn.Linear(hidden_size, out_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
                x(tensor): (-1, dim).
            """
        return self.mlp(x)


class MLPwoLastAct(nn.Layer):
    """
    MLPwoLastAct
    """
    def __init__(self, layer_num, in_size, hidden_size, out_size, act, dropout_rate=0):
        super(MLPwoLastAct, self).__init__()

        layers = []

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        for layer_id in range(layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
            elif layer_id < layer_num - 1:
                layers.append(nn.Linear(hidden_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, out_size))
                layers.append(Activation(act))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, dim).
        """
        return self.mlp(x)


class RBF(nn.Layer):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = paddle.reshape(paddle.to_tensor(centers, dtype=dtype), [1, -1])
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = paddle.reshape(x, [-1, 1])
        return paddle.exp(-self.gamma * paddle.square(x - self.centers))
        
    

