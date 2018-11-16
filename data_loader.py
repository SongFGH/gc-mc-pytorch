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

def get_loader(data_type):
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

	u_features = u_features.toarray()
	v_features = v_features.toarray()

	num_features = u_features.shape[1]

	return num_users, num_items, len(class_values), num_side_features, num_features, \
		   u_features, v_features, u_features_side, v_features_side, \
