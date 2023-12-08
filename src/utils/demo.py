import numpy as np
import tensorflow as tf
import transformers
import re
import os
import torch

from utils.dataset import Dataset
from utils.model import Model
from utils.token_map import TokenMap
from utils.probe import Probe
from sklearn.metrics import classification_report

import logging
logging.basicConfig(level = logging.DEBUG)


def finetune_model(
        model_index: int,
        X_train: np.array, 
        y_train: np.array, 
        X_val: np.array, 
        y_val: np.array):
    """
    Iterate through all combinations of configs to fine-tune model
    Args:
        model_index (int): Index of the model
        X_train (np.array): Array of BERT-tokenized post tokens from 
            training set (X_train_tokenized)
        y_train (np.array): Array of labels from training set (y_train)
        X_val (np.array): Array of BERT-tokenized post tokens from 
            training set (X_val_tokenized)
        y_val (np.array): Array of labels from training set (y_val)
    """

    optimizer_names = [
       "sgd", "rmsprop", "adadelta", "adagrad", 
       "adam", "adamax", "adafactor", "nadam", "ftrl", "adamw"
    ]
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]

    for optimizer_name in optimizer_names:
        for learning_rate in learning_rates:
            # clear session before training a new model
            tf.keras.backend.clear_session()

            # build a model
            model = Model(
                model_index = model_index,
                trainable = (model_index == 2),
                optimizer_name = optimizer_name,
                learning_rate = learning_rate,
            )
                
            # finetune model
            model_hist = model.finetune(
                X_train = X_train,
                y_train = y_train,
                X_val = X_val,
                y_val = y_val,
            )


def evaluate_tf_model(
      filename: str,
      X_test: np.array,
      y_test: np.array):
    """
    After running all experiments, pick the best iterations for models 1 and 2 
    based on validation scores and then run evaluate using test set

    Args:
        filename (str): Filename to the saved model weights
        X_test (np.array): Features for test set
        y_test (np.array): Labels for test set
    """
    tf.keras.backend.clear_session()

    model = Model(weights_filename = filename)
    model.evaluate(X_test = X_test, y_test = y_test)


def evaluate_pt_model(
        X_test: np.array,
        y_test: np.array,
        pretrained_model_name: str = "Hate-speech-CNERG/bert-base-uncased-hatexplain"):
    """
    After running all experiments, pick the best iterations for model 3 
    based on validation scores and then run evaluate using test set

    Args:
        X_test (np.array): Features for test set
        y_test (np.array): Labels for test set
        pretrained_model_name (str): Name of pretrained model
    """

    pipe = transformers.pipeline(
        "text-classification",
        model = pretrained_model_name)
    
    y_pred = np.array([
        int(prediction["label"] != "normal") for prediction in pipe(X_test)
    ])

    if isinstance(y_test, list):
        y_test = np.array(y_test)

    print(
        classification_report(
            y_true = y_test,
            y_pred = y_pred
        )
    )


def save_tf_hidden_states(
        model_index: int,
        weights_filename: str, 
        dataset: Dataset,
        folderpath: str = "../data/hidden_states/"):
    """
    Save hidden states as numpy arrays for models built using tensorflow under
    the folderpath directory.

    Args:
        model_index (int): Index of the model
        weights_filename (str): Filename of the saved model weights
        dataset (Dataset): A dataset object
        folderpath (str): Folderpath to save the hidden states
    """
    # load models
    model = Model(
        weights_filename = weights_filename,
        output_hidden_states = True,
    )

    model_index = re.findall(R"(model\_\d+)_.*", weights_filename)[0]
    filename_prefix = F"{model_index}_hidden_states"

    num_test_set_idx = len(dataset.get_split("X_test"))

    for test_set_idx in range(num_test_set_idx):

        logging.debug(F"Processing test_set_idx={test_set_idx}")

        tokenized_text = dataset.get_tokenized_test_post(test_set_idx)
        print(tokenized_text)

        # generate prediction
        result = model.model_object.predict(tokenized_text)

        # save hidden states
        filepath = os.path.join(
            folderpath, 
            F"{filename_prefix}_test_set_{test_set_idx}_hidden_states.npy"
        )
        with open(filepath, "wb") as fp:
            np.save(fp, np.array(result.hidden_states))


def save_pt_hidden_states(
        model_index: int, 
        pretrained_model_name: str, 
        dataset: Dataset,
        folderpath: str = "../data/hidden_states/"):
    """
    Save hidden states as numpy arrays for models built using pytorch under
    the directory ../data/hidden_states/.

    Args:
        model_index (int): Index of the model
        pretrained_model_name (str): Name of pretrained tokenizer
        dataset (Dataset): A dataset object
        folderpath (str): Folderpath to save the hidden states
    """
    # set up pretrained tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name,
        max_length = 200,
        truncation = True,
        padding = 'max_length'
    )

    # set up pretrained model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        output_hidden_states = True
    )

    filename_prefix = F"model_{model_index}_hidden_states"

    for test_set_idx, text in enumerate(dataset.get_split("X_test")):

        logging.debug(F"Processing test_set_idx={test_set_idx}")

        # tokenizer text
        inputs = tokenizer(text, return_tensors = "pt")

        # generate prediction
        with torch.no_grad():
            result = model(**inputs)

        # save hidden states
        filepath = os.path.join(
            folderpath,
            F"{filename_prefix}_test_set_{test_set_idx}_hidden_states.npy"
        )
        with open(filepath, "wb") as fp:
            np.save(fp, np.array([h.numpy() for h in result.hidden_states]))


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
