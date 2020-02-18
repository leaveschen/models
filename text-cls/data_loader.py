# -*- coding:utf-8 -*-
# ------ Import ------ #

from __future__ import absolute_import
import os, sys
import pickle
import random

import torch
from torch.utils.data import Dataset, DataLoader

# ------ Global Parameters ------ #


# ------ Class & Function ------ #

class RawDataset():
	def __init__(self, data_dir):
		with open(os.path.join(data_dir, 'data.pickle'), 'rb') as fin:
			data = pickle.load(fin)
			self.x_train = data['x_train']
			self.y_train = data['y_train']
			#self.x_test = data['x_test']
			#self.y_test = data['y_test']
			self.x_valid = data['x_valid']
			self.y_valid = data['y_valid']
		with open(os.path.join(data_dir, 'maps.pickle'), 'rb') as fin:
			maps = pickle.load(fin)
			self.label_map = maps['label_map']
			self.vocab = maps['vocab']
		return


class ClsDataset():
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.num_samples = len(self.x)
		return

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		sample = {
				'x':torch.tensor(self.x[idx], dtype=torch.long),
				'y':torch.tensor(self.y[idx], dtype=torch.long)
				}
		return sample


class ClsDataLoader():
	def __init__(self, data_dir, batch_size=2):
		rds = RawDataset(data_dir)
		self.label_map = rds.label_map
		self.vocab = rds.vocab

		train_ds = ClsDataset(rds.x_train, rds.y_train)
		#test_ds = ClsDataset(rds.x_test, rds.y_test)
		valid_ds = ClsDataset(rds.x_valid, rds.y_valid)

		self.train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
		#self.test = DataLoader(test_ds, batch_size=batch_size)
		self.valid = DataLoader(valid_ds, batch_size=batch_size)
		return

	@property
	def vocab_size(self):
		return len(self.vocab)

	@property
	def num_classes(self):
		return len(self.label_map)


# ------ Main Process ------ #

if __name__ == "__main__":

	pass





