# -*- coding: utf-8 -*-
"""VAR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OyJxJZw2xn3FEn654SsDbHvzjFvMpAnl
"""

import statsmodels.tsa.api
import numpy as np
import logging

class VAR:
  """
  Usa solo la variabile target, per tutti i nodi insieme.
  Le feature non-target vengono scartate.
  Il modello viene addestrato per tutti i nodi contemporaneamente, quindi la matrice è [H x N]
  """
  def __init__(self, prediction_steps, order):
    self.logger = logging.getLogger(__name__)
    self.logger.info(__name__ + ' initializing.')

    self.model = None
    self.fitted_on = []
    self.single_node_model = False
    self.order = order
    self.P = prediction_steps
    self.compiled = False

    self.logger.info(__name__ + ' initialized.')

  def fit(self, x, y, **kwargs):
    if len(self.fitted_on) != 0:
      self.fitted_on = np.concatenate((self.fitted_on, x), axis=0)
    else:
      self.fitted_on = x

    self.model = statsmodels.tsa.api.VAR(np.reshape(self.fitted_on, (-1, x.shape[2], x.shape[3]))[...,0])
    self.model = self.model.fit(self.order)

  def predict(self, x, **kwargs):
    return self.model.forecast(np.reshape(x, (-1, x.shape[2], x.shape[3]))[...,0], self.P)
  
  def compile(self, **kwargs):
    self.compiled = True