# -*- coding:utf-8 -*-
# ------ Import ------ #

import os, sys
import pickle
import argparse

# ------ Global Parameters ------ #

enable_truncate = True

# ------ Class & Function ------ #

def load_vocab(vocab_f):
	with open(vocab_f, encoding='utf8') as fin:
		x = fin.readlines()
		vocab = { v.strip('\n'):i for i,v in enumerate(x) }
	return vocab

def build_vocab(train_f):
	vocab = {'<PAD>':0}
	with open(train_f, encoding='utf8') as fin:
		for line in fin:
			_, text = line.strip().split('\t')
			for token in list(text):
				if token not in vocab:
					vocab[token] = len(vocab)
	return vocab

def load_corpus(f, vocab, label_map=None, max_len=100):
	if label_map is None:
		label_map_temp = {}

	labels = []
	texts = []
	with open(f, encoding='utf8') as fin:
		for line in fin:
			line = line.strip().split('\t')
			if len(line) != 2:
				continue
			label, text = line
			if label_map is None and label not in label_map_temp:
			#if label not in label_map_temp:
				label_map_temp[label] = len(label_map_temp)
			label_id = label_map[label] if label_map else label_map_temp[label]
			labels.append(label_id)

			if enable_truncate:
				text = text[:max_len]

			text = [ vocab[t] if t in vocab else 0 for t in text ]

			if enable_truncate and len(text) < max_len:
				text = text + [0] * (max_len - len(text))

			texts.append(text)
	return labels, texts, label_map_temp if label_map is None else label_map

def main(
		train_file,
		valid_file,
		test_file,
		vocab_file,
		out_dir,
		max_len,
		):
	# build vocabulary first
	vocab = load_vocab(vocab_file) if vocab_file is not None else build_vocab(train_file)

	y_train, x_train, label_map = load_corpus(train_file, vocab, max_len=max_len)
	y_valid, x_valid, _ = load_corpus(valid_file, vocab, label_map, max_len=max_len)
	if test_file is not None:
		y_test, x_test, _ = load_corpus(test_file, vocab, label_map, max_len=max_len)

	data = {'y_train':y_train, 'x_train':x_train,
			'y_valid':y_valid, 'x_valid':x_valid}
	if test_file is not None:
		data['y_test'] = y_test
		data['x_test'] = x_test
	maps = {'label_map':label_map, 'vocab':vocab}

	# save result
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	with open(os.path.join(out_dir, 'data.pickle'), 'wb') as fout:
		pickle.dump(data, fout)
	with open(os.path.join(out_dir, 'maps.pickle'), 'wb') as fout:
		pickle.dump(maps, fout)

	return

# ------ Main Process ------ #

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_file', required=True, help='training data file')
	parser.add_argument('--valid_file', required=True, help='validation data file')
	parser.add_argument('--test_file', help='testing data file')
	parser.add_argument('--vocab_file', help='vocabulary data file')
	parser.add_argument('--out_dir', default='data', help='output directory')
	parser.add_argument('--max_len', default=100, help='truncate max len of input sequence')
	args = parser.parse_args()

	main(
			args.train_file,
			args.valid_file,
			args.test_file,
			args.vocab_file,
			args.out_dir,
			args.max_len,
			)





