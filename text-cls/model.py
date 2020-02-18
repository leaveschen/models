# -*- coding:utf-8 -*-
# ------ Import ------ #

from __future__ import absolute_import
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

# ------ Global Parameters ------ #


# ------ Class & Function ------ #

class TextNN(nn.Module):
	def __init__(self,
			vocab_size,
			num_classes,
			embed_dim=128,
			hidden_size=128,
			quantize=False):
		super(TextNN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.hidden = nn.Linear(embed_dim, hidden_size)
		self.relu = nn.ReLU()
		self.fc = nn.Linear(hidden_size, num_classes)

		self.quantize = quantize
		self.quant = QuantStub()
		self.dequant = DeQuantStub()
		return

	def forward(self, x):
		embedded = self.embedding(x)
		if self.quantize:
			embedded = self.quant(embedded)
		embedded = torch.mean(embedded, dim=1)
		hidden = self.hidden(embedded)
		hidden = self.relu(hidden)
		logits = self.fc(hidden)
		if self.quantize:
			logits = self.dequant(logits)
		return logits


class TextCNN(nn.Module):
	def __init__(self,
			vocab_size,
			num_classes,
			embed_dim=64,
			num_filters=64,
			kernel_size=5,
			use_2d=False,
			quantize=False):
		super(TextCNN, self).__init__()
		self.use_2d = use_2d
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		if self.use_2d:
			self.conv = nn.Conv2d(embed_dim, num_filters, (kernel_size, 1))
		else:
			self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
		self.pool = nn.AdaptiveMaxPool1d(1)
		self.fc = nn.Linear(num_filters, num_classes)

		self.quantize = quantize
		self.quant = QuantStub()
		self.dequant = DeQuantStub()
		return

	def forward(self, x):
		""" forward operation
			args:
				x - input tensor with shape=[batch, seq]
		"""
		embedded = self.embedding(x).transpose(1, 2) # [batch, in_channel, seq]
		#if self.quantize:
		#	embedded = self.quant(embedded)
		if self.use_2d:
			embedded = embedded.unsqueeze(-1)
		conved = self.conv(embedded)
		if self.use_2d:
			conved = conved.squeeze(-1)
		pooled = torch.squeeze(self.pool(conved), dim=-1)
		if self.quantize:
			pooled = self.quant(pooled)
		logits = self.fc(pooled)
		if self.quantize:
			logits = self.dequant(logits)
		return logits


# ------ Main Process ------ #

if __name__ == "__main__":

	pass





