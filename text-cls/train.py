# -*- coding:utf-8 -*-
# ------ Import ------ #

import os, sys
import time
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader import ClsDataLoader
from model import TextCNN, TextNN

# ------ Global Parameters ------ #

loader = ClsDataLoader('data', batch_size=64)

# ------ Class & Function ------ #

def calc_acc(logits, labels):
	correct = (logits.argmax(1) == labels).sum().item()
	total = labels.size(0)
	return (correct, total)

def evaluate(model, loader):
	correct = total = 0
	start_time = time.time()
	with torch.no_grad():
		for batch in loader:
			inputs = batch['x']
			labels = batch['y']
			logits = model(inputs)
			(c, t) = calc_acc(logits, labels)
			correct += c
			total += t
	elapse = time.time() - start_time
	print('>>> evaluate result: {}/{}, acc={}, time elapse: {}'\
			.format(correct, total, correct/total, elapse))
	return correct/total

def train_iter(model, loader, criterion, optimizer):
	total_loss = 0
	tqdm_train = tqdm(loader.train)
	for batch in tqdm_train:
		optimizer.zero_grad()
		inputs = batch['x']
		labels = batch['y']
		logits = model(inputs)
		loss = criterion(logits, labels)
		total_loss += loss.item()
		loss.backward()
		optimizer.step()
		tqdm_train.set_description('>>> training loss: {:.3f}'.format(loss.item()))
	print('>>> training loss: {}'.format(total_loss))

	print('>>> [evaluate on training]')
	evaluate(model, loader.train)
	print('>>> [evaluate on validation]')
	val_acc = evaluate(model, loader.valid)
	return val_acc

def train(epochs):
	vocab_size = loader.vocab_size
	num_classes = loader.num_classes

	model = TextCNN(vocab_size, num_classes)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	for epoch in range(epochs):
		print('-'*40 + ' epoch {} '.format(epoch) + '-'*40)
		train_iter(model, loader, criterion, optimizer)
		print()
	torch.save(model.state_dict(), 'cnn.state_dict.pth')
	return

# ------ Main Process ------ #

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=2, help='training epochs')
	args = parser.parse_args()
	train(args.epochs)
	#train()





