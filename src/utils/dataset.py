import json
import os
import transformers
import numpy as np

from collections import defaultdict

import logging
logging.basicConfig(level = logging.DEBUG)


class Dataset():
	"""
	Class to generate and split datasets
	"""
	folderpath: str
	splits: dict

	def __init__(self, folderpath = "../data/hatexplain/"):
		"""
		Initialize Dataset class

		Args:
			folderpath (str): Folder at which the data is stored
		"""
		self.folderpath = folderpath
		self.split_dataset()
		self.tokenize_posts()


	def load_posts(self) -> dict:
		"""
		Load dataset from dataset.json
		
		Returns:
			dataset (dict): A dictionary containing post IDs as keys and post
				content (a dicionary) as values
		"""
		filepath = os.path.join(self.folderpath, "dataset.json")
		with open(filepath, "r") as fp:
			dataset = json.load(fp)

		return dataset


	def load_split_indices(self) -> dict:
		"""
		Load post IDs by dataset splits from post_id_divisions.json

		Returns:
			split_indices (dict): A dictionary containing dataset split names as
				keys and post IDs as values
		"""
		filepath = os.path.join(self.folderpath, "post_id_divisions.json")
		with open(filepath, "r") as fp:
			split_indices = json.load(fp)

		return split_indices


	def print_dataset_info(self):
		"""
		Print the number of samples by dataset split
		"""
		for key in self.splits.keys():
  			print(F"# of samples in {key}:\t{len(self.splits[key])}")
			  
		for i in ["y_train", "y_test", "y_val"]:
			counts = np.unique(self.splits[i], return_counts = True)
			pos_pct = counts[1][0]/sum(counts[1])
			print(F"{i}: {counts}\t| positive label %: {pos_pct}")


	def split_dataset(self) -> dict:
		"""
		Split the dataset yielding a dictionary containing the following:
			- X_train: Texts from posts in train set
			- y_train: Toxicity labels for posts in train set
			- X_val: Texts from posts in validation set
			- y_val: Toxicity labels for posts in validation set
			- X_test: Texts from posts in test set
			- X_test_rationales: Annotator rationales for test set
			- X_test_post_tokens: Post tokens for test set
			- X_test_annotators: Annotators for test set
			- y_test: Toxicity labels for posts in test set
		"""
		
		# get datset and split indices
		posts = self.load_posts()
		split_indices = self.load_split_indices()

		# set up default dictionary
		self.splits = defaultdict(list)

		# set threshold for positive label
		pos_threshold = 0.5

		# set up labeling index; forcing dataset into a binary classification task
		label_index = {"normal": 0, "offensive": 1, "hatespeech": 1}

		# loop through each split
		for split_name, post_ids in split_indices.items():
			logging.debug(F"[split_dataset] processing {split_name}...")

			# loop through each post id
			for post_id in post_ids:

				# get content of the post
				post_content = posts[post_id]

				# majority vote on labels
				label_sum = 0
				for annotator in post_content["annotators"]:
					label_sum += label_index[annotator["label"]]
				pos_pct = label_sum/len(post_content["annotators"])
				label = 1 if pos_pct > pos_threshold else 0

				# concatenate tokens using space
				text = " ".join(post_content["post_tokens"])

				self.splits[F"X_{split_name}"].append(text)
				self.splits[F"y_{split_name}"].append(label)

				# get rationale and token list for test set only for probing
				if split_name == "test":
					self.splits[F"X_{split_name}_rationales"].append(post_content["rationales"])
					self.splits[F"X_{split_name}_post_tokens"].append(post_content["post_tokens"])
					self.splits[F"X_{split_name}_annotators"].append(post_content["annotators"])
	

	def tokenize_posts(
			self, 
			max_length = 200, 
			tokenizer_name = 'bert-base-uncased',
			splits_to_process = ["X_train", "X_test", "X_val"]):
		"""
		Tokenized posts using a pretrained tokenizer from Huggingface

		Args:
			max_length (int): Maximum length of the post (truncate if longer)
			tokenizer_name (str):  Name of pretrained tokenizer
			splits_to_process (list[str]): List of split names to tokenize
		"""

		tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

		for split_name in splits_to_process:
			logging.debug(F"[tokenize_posts] processing {split_name}...")
			
			result = tokenizer(
				self.splits[split_name],
				max_length = max_length,
				truncation = True,
				padding = 'max_length',
				return_tensors = 'tf',
			)
			self.splits[F"{split_name}_tokenized"] = [
				result.input_ids,
				result.token_type_ids,
				result.attention_mask	
			]

	def get_split(self, split_name: str) -> np.array:
		"""
		"""
		if split_name not in self.splits.keys():
			raise RuntimeError(F"Invalid split_name: {split_name}")
		
		return self.splits[split_name]