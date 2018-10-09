#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils import data


class DataFolder(data.Dataset):
	"""Load Data for Iterator. """
	def __init__(self, data_path, neg_path, du_path, di_path, neg_cnt):
		"""Initializes image paths and preprocessing module."""
		self.data = torch.tensor(pd.read_pickle(data_path).values)
		self.neg_list = torch.tensor(np.load(neg_path))
		self.degree_user = torch.load(du_path)
		self.degree_item = torch.load(di_path)
		self.neg_cnt = neg_cnt

	def __getitem__(self, index):
		"""Reads an Data and Neg Sample from a file and returns."""
		src = self.data[index]
		usr = int(src[0])-1
		itm = int(src[1])-1

		neg = random.sample(list(self.neg_list[usr]), self.neg_cnt)
		neg = self.neg_list[usr]

		du = torch.abs(self.degree_user[usr]).float()
		di = torch.abs(self.degree_item[itm]).float()

		return src, neg, 1/torch.sqrt(du*di)

	def __len__(self):
		"""Returns the total number of font files."""
		return self.data.size(0)


def get_loader(data_path, neg_path, du_path, di_path,
			   neg_cnt, batch_size, shuffle=True, num_workers=2):
	"""Builds and returns Dataloader."""

	dataset = DataFolder(data_path, neg_path, du_path, di_path, neg_cnt)

	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers)
	return data_loader
