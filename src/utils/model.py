import transformers
import numpy as np
import re
import os
import time
import torch
import tensorflow as tf
import keras.backend as K

from sklearn.metrics import classification_report

import logging
logging.basicConfig(level = logging.DEBUG)


class Model():
    """
    Class to build, finetune and evaluate model
    """
    model_object: object
    model_id: str
    initial_epoch: int
    trainable: bool
    optimizer_name: str
    learning_rate: float
    output_hidden_states: bool
    model_checkpoints_folderpath: str
    weights_filename: str


    def __init__(
            self,
            weights_filename: str = None,
            model_index: int = 0,
            trainable: bool = False,
            optimizer_name: str = "adam",
            learning_rate: float = 1e-5,
            output_hidden_states: bool = False,
            model_checkpoints_folderpath: str = "../models/model_checkpoints/"):
        """
        Initialize a model class object

        Args:
            model_index (int): Index of the model which will be used as part of
                model ID
            trainable (bool): Whether to unfreeze BERT hidden layers during
                fine-tuning
            optimizer_name (str): Name of optimizer as per Keras documentations
            learning_rate (float): Learning rate to be used for optimizer
            model_checkpoints_folderpath (str): Full folderpath for model checkpoints
            weights_filename (str): Filename for weights (if available)
            output_hidden_states (bool): Whether to output hidden states
        """
        
        # if weights are provided, get information based on filename
        if weights_filename:
            self.model_id = (
                re.findall(R".*(model_\d+).*", weights_filename)[0] 
                + "_" + re.findall(R".*(\d{10}).*", weights_filename)[0]
            )
            self.initial_epoch = int(re.findall(r".*_epoch-(\d+)_.*", weights_filename)[0])
            self.optimizer_name = re.findall(r".*_opt-(.*)_lr.*", weights_filename)[0]
            self.learning_rate = float(re.findall(r".*_lr-(.*)_epoch.*", weights_filename)[0])
            self.trainable = bool(int(re.findall(r".*_trainable-(.*)_opt.*", weights_filename)[0]))
            print(self.trainable)
        else:
            self.model_id = F"model_{model_index}_{int(time.time())}"
            self.initial_epoch = 0
            self.optimizer_name = optimizer_name
            self.learning_rate = learning_rate
            self.trainable = trainable

        self.weights_filename = weights_filename
        self.output_hidden_states = output_hidden_states
        self.model_checkpoints_folderpath = model_checkpoints_folderpath
        
        self.model_object = None
        self.build()


    def f1(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Compute F1 score

        Args:
            y_true (np.array): An array of true labels
            y_pred (np.array): An array of predicted labels
        """
        actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (actual_positives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

        return f1


    def load_weights(self):
        """
        Load weights if weights filename is provided.
        """
        if not self.weights_filename:
            return 
        
        filepath = F"{self.model_checkpoints_folderpath}{self.weights_filename}"
        self.model_object.load_weights(filepath)
        
        

    def get_optimizer(self):
        """
        Set up the optimizer object based on user selection
        """
        
        optimizer = None
        
        if self.optimizer_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate = self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = self.learning_rate)
        elif self.optimizer_name == "adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate = self.learning_rate)
        elif self.optimizer_name == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate = self.learning_rate)
        elif self.optimizer_name == "adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate = self.learning_rate)
        elif self.optimizer_name == "adafactor":
            optimizer = tf.keras.optimizers.Adafactor(learning_rate = self.learning_rate)
        elif self.optimizer_name == "nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate = self.learning_rate)
        elif self.optimizer_name == "ftrl":
            optimizer = tf.keras.optimizers.Ftrl(learning_rate = self.learning_rate)
        elif self.optimizer_name == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate = self.learning_rate)
        elif self.optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        else:
            raise RuntimeError(F"invalid optimizer_name: {self.optimizer_name}!")

        return optimizer


    def build(self):
        """
        Build a BERT-based model for fine-tuning
        
        Args:
            weights_filename (str): Filename to the model weights (.keras)
                if available
        """

        # load bert base model
        self.model_object = transformers.TFAutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels = 1,
            output_hidden_states = self.output_hidden_states,
        )

        # load weights
        self.load_weights()

        # set bert base model trainability
        self.model_object.bert.trainable = self.trainable

        # set compile arguments
        compile_args = {
            "optimizer": self.get_optimizer(),
            "loss": tf.keras.losses.BinaryCrossentropy(),
            "metrics": [tf.keras.metrics.AUC(name = "auc"), self.f1]
        }

        # compile model
        self.model_object.compile(**compile_args)
    
    
    def finetune(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        patience: int = 5,
        batch_size: int = 32,
        epochs: int = 50):
        """
        Fine-tune model

        Args:
            X_train (np.array): Array of BERT-tokenized post tokens from 
                training set (X_train_tokenized)
            y_train (np.array): Array of labels from training set (y_train)
            X_val (np.array): Array of BERT-tokenized post tokens from 
                training set (X_val_tokenized)
            y_val (np.array): Array of labels from training set (y_val)
            patience (int): Number of iterations to train after the metric
                stops improving
            batch_size (int): Number of samples in each batch
            epochs (int): Number of epochs to train

        Returns:
            model_history: History of training
        """

        # set up model checkpoint callback
        filepath = (
            self.model_checkpoints_folderpath
            + F"{self.model_id}_"
            + F"trainable-{int(self.trainable)}_"
            + F"opt-{self.optimizer_name}_"
            + F"lr-{self.learning_rate}_"
            + "epoch-{epoch:02d}_"
            + "loss-{loss:.4f}_"
            + "val_loss-{val_loss:.4f}_"
            + "auc-{auc:.4f}_"
            + "val_auc-{val_auc:.4f}_"
            + "f1-{f1:.4f}_"
            + "val_f1-{val_f1:.4f}"
            + ".keras"
        )
        callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath = filepath,
            save_weights_only = True,
            save_best_only = True,
        )

        # set up early stopping callback
        callback_early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor = "val_loss",
            mode = "min",
            patience = patience,
        )

        if isinstance(y_train, list):
            y_train = np.array(y_train)
        if isinstance(y_val, list):
            y_val = np.array(y_val)

        # fit model
        model_history = self.model_object.fit(
            x = X_train,
            y = y_train,
            batch_size = batch_size,
            epochs = epochs,
            callbacks = [
                callback_model_checkpoint,
                callback_early_stopping
            ],
            validation_data = (X_val, y_val),
            shuffle = True,
            initial_epoch = self.initial_epoch,
        )

        return model_history


    def evaluate(self, X_test: np.array, y_test: np.array):
        """
        Evaluate model

        Args:
            X_test (np.array): Array of BERT-tokenized post tokens from 
                training set (X_test_tokenized)
            y_text (np.array): Array of labels from training set (y_text)
        """
        # generate predictions on test set
        y_pred = self.model_object.predict(X_test).logits
        y_pred = np.where(y_pred.reshape(-1) > 0.5, 1, 0)

        if isinstance(y_test, list):
            y_test = np.array(y_test)

        # generate classifcation report
        print(
            classification_report(
                y_true = y_test,
                y_pred = y_pred
            )
        )


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
        X_test_tokenized: list[str],
        folderpath: str = "../data/hidden_states/"):
    """
    Save hidden states as numpy arrays for models built using tensorflow under
    the folderpath directory.

    Args:
        model_index (int): Index of the model
        weights_filename (str): Filename of the saved model weights
        X_test_tokenized (list[str]): List of tokenized subword tokens to use
            in extracting the hidden states
    """
    # load models
    model = Model(
        weights_filename = weights_filename,
        output_hidden_states = True,
    )

    model_index = re.findall(R"(model\_\d+)_.*", weights_filename)[0]
    filename_prefix = F"{model_index}_hidden_states"

    for test_set_idx, text in enumerate(X_test_tokenized):

        logging.debug(F"Processing test_set_idx={test_set_idx}")

        # generate prediction
        result = model.model_object.predict(text)

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
        X_test: list[str],
        folderpath: str = "../data/hidden_states/"):
    """
    Save hidden states as numpy arrays for models built using pytorch under
    the directory ../data/hidden_states/.

    Args:
        model_index (int): Index of the model
        pretrained_model_name (str): Name of pretrained tokenizer
        X_test (list[str]): List of post contents to use in extracting the 
            hidden states
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

    for test_set_idx, text in enumerate(X_test):

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