#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils import data

from utils import load_official_trainvaltest_split, adj2support

class DataFolder(data.Dataset):
	"""Load Data for Iterator. """
	def __init__(self, data_type, mode):
		"""Initializes image paths and preprocessing module."""
		u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
		    val_labels, val_u_indices, val_v_indices, test_labels, \
		    test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(data_type)

		self.u_features = u_features.todense()
		self.v_features = v_features.todense()
		self.adj_train = adj_train.todense()
		self.class_values = class_values

		if mode == 'train':
			self.u_indices = train_u_indices
			self.v_indices = train_v_indices
			self.labels = train_labels
		elif mode == 'valid':
			self.u_indices = val_u_indices
			self.v_indices = val_v_indices
			self.labels = val_labels
		elif mode == 'test':
			self.u_indices = test_u_indices
			self.v_indices = test_v_indices
			self.labels = test_labels

	def get_features(self):
		self.adj_matrix = adj2support(self.adj_train, len(self.class_values), self.u_features.shape[0], self.v_features.shape[0])
		return self.u_features, self.v_features, self.adj_matrix, self.class_values

	def __getitem__(self, index):
		"""Reads an Data and Neg Sample from a file and returns."""
		u_indices = self.u_indices[index]
		v_indices = self.v_indices[index]
		labels = self.labels[index]

		return u_indices, v_indices, labels

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.labels)


def get_loader(data_type, mode, batch_size, shuffle=True, num_workers=2):
	"""Builds and returns Dataloader."""

	dataset = DataFolder(data_type, mode)
	u_features, v_features, adj, classes = dataset.get_features()

	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers)

	return data_loader, u_features, v_features, adj, classes
