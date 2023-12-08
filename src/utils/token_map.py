import transformers
import pandas as pd
import numpy as np
import os

import logging
logging.basicConfig(level = logging.DEBUG)


class TokenMap():
    """
    Class to build post token to bert subword token map
    """
    df: pd.DataFrame
    folderpath: str


    def __init__(self, folderpath = "../data/token_maps/"):
        """
        Initalize the class object

        Args:
            folderpath (str): Full folderpath where all token maps are stored
        """
        self.df = pd.DataFrame()
        self.folderpath = folderpath


    def save(self, filename: str):
        """
        Save token map dataframe as a csv

        Args:
            filename (str): Filename to be used for the csv
        """
        filepath = os.path.join(self.folderpath, filename)
        self.df.to_csv(filepath, index = False)


    def load(self, filename: str):
        """
        Load csv as a token map dataframe

        Args:
            filename (str): Filename of the csv to be loaded
        """
        filepath = os.path.join(self.folderpath, filename)
        self.df = pd.read_csv(filepath)
        
    
    def _get_post_content(self, post_tokens: list[str]) -> str:
        """
        Concatenate post tokens from original dataset into one string

        Args:
            post_tokens (list[str]): List of tokens in a post

        Returns:
            post_content (str): A string of concatenated tokens
        """
        post_content = " ".join(post_tokens)

        return post_content
    

    def _get_targets(self, annotators: list[dict]) -> str:
        """
        Retrieve a list of unique targets per post

        Args:
            annotators (list[dict]): List of annotations for a post such as
                the following:
                [
                    {
                        "label": "hatespeech",
                        "annotator_id": 4,
                        "target": ["African"]
                    },
                    {
                        "label": "hatespeech",
                        "annotator_id": 3,
                        "target": ["Women"]
                    },
                    {
                        "label": "offensive",
                        "annotator_id": 5,
                        "target": ["African"]
                    }
                ]
        
        Returns:
            targets (str): A string of unique targets (e.g. "African, Women")
        """
        targets = list()
        for annotator in annotators:
            targets.extend(annotator["target"])
        targets = ", ".join(sorted(list(set(targets) - set(["None"]))))

        return targets
    

    def _get_majority_rationales(
            self, 
            num_post_tokens: int, 
            rationales: list[list[int]]) -> np.array:
        """
        Apply majority voting on rationales for a post

        Args:
            num_post_tokens (int): Number of post tokens in a post
            rationales (list[list[int]]): A list of list of integers 
                representing a list of annotator's rationales for each token
                in a post, such as the following:
                [
                    [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ]

        Returns:
            rationales (np.array): An array of rationales for a post after
                majority voting
        """
        # assign rationale indicator as -1 for those without rationale
        if len(rationales) == 0:
            rationales = np.ones(num_post_tokens) * -1

        # otherwise, apply majority vote
        else:
            rationales = (np.mean(rationales, axis = 0) > 0.5) * 1

        return rationales


    def build(
        self,
        pretrained_model_name: str,
        X_test_post_tokens: list[list[str]],
        X_test_rationales: list[list[int]],
        X_test_annotators: list[list[dict]],
        y_test: list[int]):
        """
        Build the mapping from post tokens to BERT subword tokens

        Args:
            pretrained_model_name (str): Name of pretrained tokenizer to use
            X_test_post_tokens (list[list[str]]): A list of list of post tokens
                from the test set
            X_test_rationales (list[list[int]]): A list of list of rationales
                from the test set
            X_test_annotators (list[list[dict]]): A list of list of annotations
                from the test set
            y_test (list[int]): A list of labels from the test set
        """
        # set up tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name
        )

        # set up empty arrays for final dataframe
        records = []

        # loop through each test set sample
        for test_set_index in range(len(y_test)):
            logging.debug(F"[generate_mapping] test_set_index={test_set_index}")

            post_tokens = X_test_post_tokens[test_set_index]
            annotators = X_test_annotators[test_set_index]
            rationales = X_test_rationales[test_set_index]
            label = y_test[test_set_index]

            # generate post-level information
            post_content = self._get_post_content(post_tokens)
            targets = self._get_targets(annotators)
            rationales = self._get_majority_rationales(
                num_post_tokens = len(post_tokens),
                rationales = rationales
            )

            # set up bert subword token index at 1 since [cls] is at index 0
            bert_token_start_idx = 1 

            # tokenize one post token to one or more subword BERT tokens at a time
            for post_token_idx, post_token in enumerate(post_tokens):

                logging.debug(
                    F"post_token_idx={post_token_idx}, post_token={post_token}"
                )

                bert_input_ids = tokenizer(post_token, add_special_tokens = False).input_ids
                bert_token_end_idx = bert_token_start_idx + len(bert_input_ids)

                # check to make sure that start and end indices are not the same
                if bert_token_start_idx == bert_token_end_idx:
                    continue

                record = {
                    "test_set_index": test_set_index,
                    "post_token_text": post_token,
                    "post_token_index": post_token_idx,
                    "bert_token_start_index": bert_token_start_idx,
                    "bert_token_end_index": bert_token_end_idx,
                    "post_content": post_content,
                    "rationales": rationales[post_token_idx],
                    "label": label,
                    "targets": targets,
                }
                records.append(record)

                bert_token_start_idx = bert_token_end_idx

        # combine all records
        self.df = pd.DataFrame.from_records(records)
    

