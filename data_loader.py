#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils import data

from utils import *

class DataFolder(data.Dataset):
	"""Load Data for Iterator. """
	def __init__(self, support, support_t, u_indices, v_indices, labels,
					   u_features_side, v_features_side, class_cnt):
		"""Initializes image paths and preprocessing module."""

		self.support = support.reshape(support.shape[0], class_cnt, -1)
		self.support_t = support_t.reshape(support_t.shape[0], class_cnt, -1)
		self.u_indices = u_indices
		self.v_indices = v_indices
		self.labels = labels
		self.u_features_side = u_features_side
		self.v_features_side = v_features_side

	def __getitem__(self, index):
		"""Reads an Data and Neg Sample from a file and returns."""
		u_index = self.u_indices[index]
		v_index = self.v_indices[index]
		label = self.labels[index]

		support = self.support[u_index]
		support_t = self.support_t[v_index]
		u_features_side = self.u_features_side[u_index]
		v_features_side = self.v_features_side[v_index]

		return u_index, v_index, label, support, support_t, u_features_side, v_features_side

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.labels)


def get_loader(data_type, batch_size, shuffle=True, num_workers=2):
	"""Builds and returns Dataloader."""
	SYM = True
	DATASET = data_type
	datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'

	u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
		val_labels, val_u_indices, val_v_indices, test_labels, \
		test_u_indices, test_v_indices, class_values = create_trainvaltest_split(data_type, datasplit_path=datasplit_path)

	num_users, num_items = adj_train.shape

	print("Normalizing feature vectors...")
	u_features_side = normalize_features(u_features)
	v_features_side = normalize_features(v_features)

	u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

	# 943x41, 1682x41
	u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
	v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

	num_side_features = u_features_side.shape[1]

	# node id's for node input features
	id_csr_u = sp.identity(num_users, format='csr')
	id_csr_v = sp.identity(num_items, format='csr')

	u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

	# global normalization
	support = []
	support_t = []
	adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)

	for i in range(len(class_values)):
		# build individual binary rating matrices (supports) for each rating
		support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)
		support_unnormalized_transpose = support_unnormalized.T
		support.append(support_unnormalized)
		support_t.append(support_unnormalized_transpose)

	support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
	support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

	num_support = len(support)
	support = sp.hstack(support, format='csr')
	support_t = sp.hstack(support_t, format='csr')

	'''
	# Collect all user and item nodes for test set
	test_u = list(set(test_u_indices))
	test_v = list(set(test_v_indices))
	test_u_dict = {n: i for i, n in enumerate(test_u)}
	test_v_dict = {n: i for i, n in enumerate(test_v)}

	test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
	test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])

	test_support = support[np.array(test_u)]
	test_support_t = support_t[np.array(test_v)]

	# features as side info
	test_u_features_side = u_features_side[np.array(test_u)]
	test_v_features_side = v_features_side[np.array(test_v)]

	# Collect all user and item nodes for validation set
	val_u = list(set(val_u_indices))
	val_v = list(set(val_v_indices))
	val_u_dict = {n: i for i, n in enumerate(val_u)}
	val_v_dict = {n: i for i, n in enumerate(val_v)}

	val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
	val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

	val_support = support[np.array(val_u)]
	val_support_t = support_t[np.array(val_v)]

	val_u_features_side = u_features_side[np.array(val_u)]
	val_v_features_side = v_features_side[np.array(val_v)]

	# Collect all user and item nodes for train set
	train_u = list(set(train_u_indices))
	train_v = list(set(train_v_indices))
	train_u_dict = {n: i for i, n in enumerate(train_u)}
	train_v_dict = {n: i for i, n in enumerate(train_v)}

	train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
	train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

	train_support = support[np.array(train_u)]
	train_support_t = support_t[np.array(train_v)]

	train_u_features_side = u_features_side[np.array(train_u)]
	train_v_features_side = v_features_side[np.array(train_v)]


	test_support = test_support.toarray()
	test_support_t = test_support_t.toarray()

	val_support = val_support.toarray()
	val_support_t = val_support_t.toarray()

	train_support = train_support.toarray()
	train_support_t = train_support_t.toarray()
	'''

	support = support.toarray()
	support_t=support_t.toarray()
	
	u_features = u_features.toarray()
	v_features = v_features.toarray()

	num_features = u_features.shape[1]


	train_dataset = DataFolder(support, support_t,
							   train_u_indices, train_v_indices, train_labels,
							   u_features_side, v_features_side, len(class_values))
	valid_dataset = DataFolder(support, support_t,
							   val_u_indices, val_v_indices, val_labels,
							   u_features_side, v_features_side, len(class_values))
	test_dataset = DataFolder(support, support_t,
							  test_u_indices, test_v_indices, test_labels,
							  u_features_side, v_features_side, len(class_values))

	train_loader = data.DataLoader(dataset=train_dataset,
		 						   batch_size=batch_size,
								   shuffle=shuffle,
								   num_workers=num_workers)
	valid_loader = data.DataLoader(dataset=valid_dataset,
		 						   batch_size=batch_size,
								   shuffle=False,
								   num_workers=num_workers)
	test_loader = data.DataLoader(dataset=test_dataset,
		 						   batch_size=batch_size,
								   shuffle=False,
								   num_workers=num_workers)

	return train_loader, valid_loader, test_loader,\
	 	   u_features, v_features, \
		   class_values, num_users, num_items,\
		   u_features_side.shape[1], num_support
