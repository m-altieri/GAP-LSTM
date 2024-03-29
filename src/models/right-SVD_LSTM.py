# -*- coding: utf-8 -*-
"""SVD_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l7iD34_SqsaTEsHpgMMT9YxLK-5bjBNe
"""

import tensorflow as tf
import numpy as np
import logging

class SVD_LSTM(tf.keras.Model):
  """
  """
  def __init__(self, nodes, features, prediction_steps, **kwargs):
    super(SVD_LSTM, self).__init__()

    self.logger = logging.getLogger(__name__)
    self.logger.info(__name__ + ' initializing.')

    self.nodes = nodes
    self.features = features
    self.P = prediction_steps
    self.order = kwargs.get('order', 3)

    self.cell = tf.keras.layers.LSTMCell(self.nodes * self.features)
    if kwargs.get('has_dense', False):
      self.has_dense = True
      self.dense = tf.keras.layers.Dense(self.nodes)
    self.logger.info(__name__ + ' initialized.')

  def call(self, inputs):
    B, H, N, F = inputs.shape

    s, u, v = tf.linalg.svd(inputs) # s è [B,H,Z], u è [B,H,N,Z], v è [B,H,Z,F], dove Z = min(N,F)

    self.logger.critical('001 s: ' + str(s.shape) + ', u: ' + str(u.shape) + ', v: ' + str(v.shape))
    s = tf.tensor_scatter_nd_update(tensor=s, 
                          indices=[[B,H,o] for o in range(self.order, s.shape[-1]) for b in range(B) for h in range(H)],
                          updates=[0. for o in range((s.shape[-1] - self.order) * B * H)]
    )
    s = tf.linalg.diag(s)
    inputs_approx = tf.linalg.matmul(u, s)
    inputs_approx = tf.linalg.matmul(inputs_approx, v, transpose_b=True)
    inputs_approx = tf.reshape(inputs_approx, [B,H,N*F])

    preds = []
    carry = [tf.zeros((B, N * F)), tf.zeros((B, N * F))]

    for h in range(H):
      memory, carry = self.cell(inputs_approx[:,h,:], carry) # Memory: [B, NF], Carry: [2, [B, NF]]; memory e carry[0] sono identici

    for p in range(self.P):
      memory, carry = self.cell(memory, carry) # Memory: [B, NF], Carry: [2, [B, NF]]; memory e carry[0] sono identici
      preds.append(memory) # [p, [B, NF]]
      
    res = tf.transpose(preds, perm=[1,0,2]) # [B, P, N]

    if self.has_dense:
      res = self.dense(res) # [B, P, N]
    return res