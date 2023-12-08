from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from token_map import TokenMap

import os
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level = logging.DEBUG)



class Probe():
    """
    Class to build and evaluate probing classifiers
    """
    model_index: int
    folderpath: str
    token_map: TokenMap

    def __init__(
        self, 
        model_index: int, 
        token_map: TokenMap,
        folderpath: str = "../data/probe_results/"):
        """
        Initialize the class object

        Args:
            model_index (int): Index of the model
            token_map (TokenMap): A TokenMap object associated with the model
            folderpath (str): Full folderpath where results are saved
        """
        
        self.model_index = model_index
        self.token_map = token_map
        self.folderpath = folderpath


    def _get_probe_masks(self, words: list[str]) -> dict:
        """
        Set up dataframe masks based on a list of words

        Args:
            words (list[str]): List of words to be used in setting up masks
        
        Returns:
            probe_masks (dict): A dictionary containing dataframe masks for 
                positive and negative samples for each task
        """
        probe_masks = {
            "mention_vs_non_mention": {
                "mask_pos": (self.token_map.df["post_token_text"].isin(words)),
                "mask_neg": ~(self.token_map.df["post_token_text"].isin(words)),
            },
            "mention_in_toxic_post_vs_non_mention_in_toxic_post": {
                "mask_pos": (
                    (self.token_map.df["post_token_text"].isin(words)) 
                    & (self.token_map.df["label"] == 1)
                ),
                "mask_neg": (
                    ~(self.token_map.df["post_token_text"].isin(words)) 
                    & (self.token_map.df["label"] == 1)
                ),
            },
            "mention_in_non_toxic_post_vs_non_mention_in_non_toxic_post": {
                "mask_pos": (
                    (self.token_map.df["post_token_text"].isin(words)) 
                    & (self.token_map.df["label"] == 0)
                ),
                "mask_neg": (
                    ~(self.token_map.df["post_token_text"].isin(words)) 
                    & (self.token_map.df["label"] == 0)
                ),
            },
            "toxic_mention_in_toxic_post_vs_non_toxic_mention_in_non_toxic_post": {
                "mask_pos": (
                    self.token_map.df["post_token_text"].isin(words) 
                    & (self.token_map.df["label"] == 1) 
                    & (self.token_map.df["rationales"] == 1)
                ),
                "mask_neg": (
                    self.token_map.df["post_token_text"].isin(words)) 
                    & (self.token_map.df["label"] == 0)
                    & (self.token_map.df["rationales"] <= 0),
            },
            "toxic_mention_in_toxic_post_vs_non_toxic_mention_in_toxic_post": {
                "mask_pos": (
                    self.token_map.df["post_token_text"].isin(words)
                    & (self.token_map.df["label"] == 1) 
                    & (self.token_map.df["rationales"] == 1)
                ),
                "mask_neg": (
                    self.token_map.df["post_token_text"].isin(words) 
                    & (self.token_map.df["label"] == 1) 
                    & (self.token_map.df["rationales"] <= 0)
                ),
            }
        }

        return probe_masks
    

    def _get_probe_dataset(self, probe_masks: dict, random_state: int):
        """
        Generate probe dataset using probe masks and a random state

        Args:
            prob_masks (dict): A dictionary with keys "mask_pos" and "mask_neg"
                along with the corresponding dataframe masks
            random_state (int): Random state to use for downsampling
        """
        # get column and row masks
        mask_pos, mask_neg = probe_masks["mask_pos"], probe_masks["mask_neg"]
        keep_cols = ["test_set_index", "bert_token_start_index", "bert_token_end_index"]

        # get positive samples
        df_pos = self.token_map.df.loc[mask_pos, keep_cols].copy()
        df_pos["label"] = 1

        # get negative samples
        df_neg = self.token_map.df.loc[mask_neg, keep_cols].copy()
        df_neg["label"] = 0

        logging.debug(F"df_pos: {df_pos.shape}, df_neg: {df_neg.shape}")

        # downsample more prevalent class
        max_records = min(df_pos.shape[0], df_neg.shape[0])
        if df_pos.shape[0] > max_records:
            df_pos = df_pos.sample(max_records, random_state = random_state)
        if df_neg.shape[0] > max_records:
            df_neg = df_neg.sample(max_records, random_state = random_state)

        logging.debug(F"df_pos: {df_pos.shape}, df_neg: {df_neg.shape}")

        # combine positive and negative samples
        df_probe_dataset = pd.concat([df_pos, df_neg]).reset_index()

        return df_probe_dataset


    def _get_hidden_state_for_post_token(
        self,
        layer_index: int,
        test_set_index: int,
        bert_token_start_index: int,
        bert_token_end_index: int) -> np.array:
        """
        Retrieve hidden state for a particular layer, a test set index and a 
        post token. Apply simple average for post tokens with multiple BERT
        subword tokens.

        Args:
            layer_index (int): Index of the layer from which to retrieve the
                hidden state
            test_set_index (int): Index of the test set sample from which to
                retrieve the hidden state
            bert_token_start_index (int): Start index of the BERT token
            bert_token_end_index (int): End index of the BERT token
        
        Returns:
            X (np.array): An array with extracted hidden states (one hidden
                state for each post token)
        """

        # load the hidden states
        filepath_prefix = F"../data/internal_states/model_{self.model_index}_internal_states"
        filepath = F"{filepath_prefix}_test_set_{test_set_index}_hidden_states.npy"
        hidden_states = np.load(filepath)

        # generate corresponding BERT token indices
        bert_token_indices = np.arange(bert_token_start_index, bert_token_end_index)

        # get contextual embeddings for BERT subword tokens
        X = hidden_states[layer_index][0][bert_token_indices]

        # take the average of all BERT subword tokens for the post token
        X = np.mean(X, axis = 0)

        return X
    

    def _split_probe_dataset(
            self,
            X: np.array,
            y: np.array,
            group: np.array) -> tuple:
        """
        Split probe dataset into train and test sets
        
        Args:
            X (np.array): An array with extracted hidden states (one hidden
                state for each post token)
            y (np.array): An array with labels
            group (np.array): An array with test set indices
        
        Returns:
            X_train (np.array): Features for train set
            X_test (np.array): Features for test set
            y_train (np.array): Labels for train set
            y_test (np.array): Labels for test set
        """
        # reason to use stratified group k fold is to avoid having tokens from
        # the same post assigned to both train and test sets
        cv = StratifiedGroupKFold(n_splits = 2, shuffle = True)
        train_idx, test_idx = next(cv.split(X, y, group))

        # ensure that the same post doesn't get assigned across train and test
        assert len(set(group[train_idx]).intersection(set(group[test_idx]))) == 0

        X_train, X_test, y_train, y_test = (
            X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        )

        return X_train, X_test, y_train, y_test


    def _train_probe_classifier(
            self,
            X_train: np.array,
            y_train: np.array,
            random_state: int) -> Pipeline:
        """
        Train a probing classifier using train sets and a random state

        Args:
            X_train (np.array): Features for train set
            y_train (np.array): Labels for train set
            random_state (int): Random state to use while training

        Returns:
            pipe (Pipeline): A trained pipeline object containing both scaler
                and classifier
        """
        # scale values based on training set
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # train probing classifier
        model = LogisticRegression(
            random_state = random_state, 
            class_weight = "balanced", 
            max_iter = 10000
        )
        
        pipe = Pipeline([('scaler', scaler), ('clf', model)])
        pipe.fit(X_train, y_train)

        return pipe


    def _evaluate_probe_classifier(
            self,
            pipe: Pipeline,
            X_test: np.array,
            y_test: np.array) -> float:
        """
        Evaluate a trained probing classifier

        Args:
            pipe (Pipeline): A trained pipeline containing a scaler and a 
                classifier
            X_test (np.array): Features for test set
            y_test (np.array): Labels for test set
        
        Returns:
            f1 (float): F1 score
        """
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        return f1

        
    def _get_hidden_states(self, df_probe_dataset: pd.DataFrame, layer_index: int):
        """
        Retrieve hidden states for all probe samples

        Args:
            df_probe_dataset (pd.DataFrame): A dataframe containing the probe 
                dataset along with the following columns: test_set_index, 
                bert_token_start_index and bert_token_end_index
            layer_index (int): Index of the layer from which to retrieve the 
                hidden states
        
        Returns:
            X (np.array): Hidden states for all probe samples
        """

        X = list()

        for probe_sample in df_probe_dataset.to_dict("records"):
            X_arr = self._get_hidden_state_for_post_token(
                layer_index = layer_index,
                test_set_index = probe_sample["test_set_index"],
                bert_token_start_index = probe_sample["bert_token_start_index"],
                bert_token_end_index = probe_sample["bert_token_end_index"],
            )
            X.append(X_arr)

        X = np.concatenate([X])

        return X


    def _save_results(self, df_result: pd.DataFrame, filename: str):
        """
        Save probing results to a csv

        Args:
            df_result (pd.DataFrame): A dataframe containing the probing
                results
            filename (str): Filename of the csv file to be saved
        """
        filepath = os.path.join(self.folderpath, filename)
        df_result.to_csv(filepath, index = False)


    def _is_completed(self, filename: str) -> bool:
        """
        Check if a probe has been completed based on a saved csv file

        Args:
            filename (str): Filename of the csv file to be checked
        
        Returns:
            is_completed (bool): Whether the probe has been completed
        """
        filepath = os.path.join(self.folderpath, filename)
        is_completed = os.path.exists(filepath)
        
        return is_completed


    def run(
        self,
        topic: str,
        words: list[str],
        random_states: list[int] = np.arange(10),
        layer_indices: list[int] = np.arange(13)):
        """
        Run probe for a topic and a list of key words. The results are saved
        in the directory (self.folderpath)

        Args:
            topic (str): The topic to be probed
            words (list[str]): A list of keywords for the topic
            random_states (list[int]): A list of random states to run the probe
            layer_indices (list[int]): A list of layer indices to run the probe
        """
        # get masks for each probe
        probe_masks = self._get_probe_masks(words = words)

        # loop through each task
        for probe_name, probe_masks in probe_masks.items():

            # loop through each random states
            for random_state in random_states:

                # get task dataset
                df_probe_dataset = self._get_probe_dataset(probe_masks, random_state)

                # generate results filepath
                probe_id = F"{topic}_{probe_name}"
                filename = (
                    F"probe_id={probe_id}_model_index={self.model_index}_"
                    F"random_state={random_state}.csv"
                )
                # check if it exists, if it does, skip it
                if self._is_completed(filename):
                    continue

                df_results = []

                # loop through each layer
                for layer_index in layer_indices:

                    # get dataset
                    X = self._get_hidden_states(df_probe_dataset, layer_index)
                    y = df_probe_dataset["label"]
                    group = df_probe_dataset["test_set_index"]

                    # split dataset
                    X_train, X_test, y_train, y_test = (
                        self._split_probe_dataset(X, y, group)
                    )

                    # train probing classifier
                    pipe = self._train_probe_classifier(X_train, y_train, random_state)

                    # evaluate probing classifier
                    f1 = self._evaluate_probe_classifier(pipe, X_test, y_test)
                
                    # save results
                    result = {
                        "probe_id": probe_id,
                        "model_name": F"model_{self.model_index}",
                        "layer_index": layer_index,
                        "f1": f1
                    }
                    df_results.append(result)

                    # delete X to save memory
                    del X

                # combine results
                df_results = pd.DataFrame(df_results)

                # save results
                self._save_results(df_results, filename)


def run_probes(model_index: int, token_map_filename: str):
    """
    Run probes for a model using a specified token map

    Args:
        model_index (int): Index of the model
        token_map_filename (str): Filename of the token map in csv
    """
    token_map = TokenMap()
    token_map.load(filename = token_map_filename)
    probe = Probe(model_index, token_map)

    topics_words = {
        "Profanity": [
            "ass", "assclown", "asses", "asshole", "assholes", "deadass", 
            "dumbass", 'shit','fucking', 'fuck', 'fucked', 'fuckin', 'mofuckas', 
            'motherfucking', 'fuckers',  'fuckoff', 'fuckwad', 'fucks', 'fucker', 
            'motherfuckers', 'clusterfuck', 'fuckwits', 'dumbfucks', 
            'motherfucker', 'bullshit', 'shitskin', 'bullshits', 'shitlings', 
            'shithole', 'dogshit', 'shitskins', 'shits', 'shitty',
            'shittier', 'shitholes', 'shitter', 'shitting','shithead',
        ],
        "African": [
            "africa","african","africans","afro","black","blacks","colored",
            "coon","coonin","coons","coony","ghetto","ghettos","harlem",
            "housenigger","kafir","mammy","moneynigger","mooncricket","negress",
            "negresses","negro","negroes","nig","nigero","nigga","niggah",
            "niggas","nigger","niggerdick","niggerish","niggers",
            "niggershitbullshit","niggerz","niggra","niggress","niglet",
            "niglets","nigress","nigs","pickaninny","sheboon","spade"
        ],
        "Women": [
            "babygirl","biatch","biches","bitch","bitched","bitches",
            "bitching","chicks","cunt","cunts","dame","daughter","daughters",
            "daughty","divorcee","estrogen","fanny","female","femaleness",
            "females","feminazi","feminine","feminism","feminist","feminists",
            "feminizing","femme","fems","girl","girlfriend","girls","hag","ho",
            "hoe","hoes","ladies","lady","loosewomen","lulu","mamma",
            "misogenous","misogynist","misogynistic","misogyny","mother",
            "mothers","princess","prostitute","prostitutes","queen","sis",
            "sister","sisters","slut","sluts","slutshaming","slutty","sow",
            "sows","thot","whore","whoredom","whores","wife","witch","witches",
            "woman","women","womens","womensmarch","yentas"
        ],
        "LGBTQ": [
            "agender","bulldykes","dyke","dykes","dykey","fag","faggot",
            "faggotry","faggots","faggoty","faggy","fags","fgt","furfaggotry",
            "gay","gays","girlfags","h0m0","homo","homophobe","homophobes",
            "homophobia","homophobic","homosexual","homosexuality","homosexuals",
            "jewfags","lesbian","lesbianism","lesbians","lesbophobia","lgbt",
            "lgbtq","lgbtqwtf","newfag","pansy","queer","queerbaiting","queers",
            "stormfags"
        ],
        "Jewish": [
            "antisemitism","chabad","goy","goyem","goyim","hebrew","hebrews",
            "heeb","heebhole","holocaust","hymiewood","israel","israeli",
            "israelite","israels","israid","istandwithisrael","jew","jewa",
            "jewdea","jewess","jewfags","jewfilth","jewish","jewishprivilege",
            "jews","judaism","kik","kike","kikes","kikescum","kikey",
            "namethejew","rabbi","schlomo","semite","semites","semitic",
            "semitism","shekel","shekels","shoahed","synagogue","talmud",
            "yid","yids","zionism","zionist"
        ],
        "Islam": [
            "allah","banislam","burqa","halal","hamas","hezbollah","islam",
            "islamaphobic","islami","islamic","islamist","islamistheproblem",
            "islamists","islamofascism","islamophobe","islamophobia",
            "islamophobic","jihad","jihadi","jihadis","jihadists","kislam",
            "mecca","medina","mohamed","mohammad","mohammed","mohammedean",
            "moslem","moslems","mosque","mosques","muhammad","muhammedans",
            "mulla","muslim","muslime","muslimes","muslims","musloid","muslum",
            "mussie","mussleman","mussorgsky","muzrat","muzrats","muzzie",
            "muzzies","muzzles","muzzrat","muzzrats","stopislam","umma","zakat"
        ]
    }

    for topic, words in topics_words.items():
        probe.run(topic, words)




