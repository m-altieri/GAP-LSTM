# -*- coding: utf-8 -*-
"""CNN_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19HVZMYpmwa1alRNOG6yHIj6NjhiTArbc
"""

import tensorflow as tf
import numpy as np
import logging

class CNN_LSTM(tf.keras.Model):
  """
  Implementazione di CNN-LSTM.
  I dati di input sono in formato [Batch, History steps, Nodes, Features].
  Prima vengono concatenate tutte le feature per ogni nodo, in modo da avere [Batch, History steps, Nodes*Features],
  poi viene applicato un layer Conv1D lungo le feature, che produce N filtri da H step.
  La matrice di output [Batch, History steps, Nodes] viene data alla LSTM.
  Alla fine della LSTM, essa continua per ulteriori P step, gli hidden state dei quali saranno le predizioni.

  Se invece viene passato il kwarg has_dense=True, gli hidden state passano prima da un layer dense, che resituisce le predizioni.
  Con has_dense=True dovrebbe andare meglio di circa il 10%.
  """
  def __init__(self, nodes, features, prediction_steps, **kwargs):
    super(CNN_LSTM, self).__init__()

    self.logger = logging.getLogger(__name__)
    self.logger.info(__name__ + ' initializing.')

    self.nodes = nodes
    self.features = features
    self.P = prediction_steps

    self.CNN = tf.keras.layers.Conv1D(filters=self.nodes, kernel_size=1)
    self.cell = tf.keras.layers.LSTMCell(self.nodes)
    if kwargs.get('has_dense', False):
      self.has_dense = True
      self.dense = tf.keras.layers.Dense(self.nodes)
    self.logger.info(__name__ + ' initialized.')

  def call(self, inputs):
    B, H, N, F = inputs.shape
    inputs = tf.reshape(inputs, (B, H, N * F)) # [B,H,NF]

    inputs = self.CNN(inputs) # [B, H, N], perchè N filtri

    preds = []
    carry = [tf.zeros((B, N)), tf.zeros((B, N))] # Initial states, matrici di 0 da [B, N]

    for h in range(H):
      memory, carry = self.cell(inputs[:,h,:], carry) # Memory: [B, N], Carry: [2, [B, N]]; memory e carry[0] sono identici

    for p in range(self.P):
      memory, carry = self.cell(memory, carry) # Memory: [B, N], Carry: [2, [B, N]]; memory e carry[0] sono identici
      preds.append(memory) # [p, [B, N]]
      
    res = tf.transpose(preds, perm=[1,0,2]) # [B, P, N]

    if self.has_dense:
      res = self.dense(res) # [B, P, N]
    return res