from utils.dataset import Dataset
from utils.token_map import TokenMap
from utils.demo import (
    finetune_model,
    evaluate_tf_model,
    evaluate_pt_model,
    save_tf_hidden_states,
    save_pt_hidden_states,
    run_probes
)


def main():
    """
    This main function serves as an example of how to use this code in a 
    sequence of steps.
    """
    ############################################################################
    # Step 1: Get datasets                                                     #
    ############################################################################

    dataset = Dataset()

    ############################################################################
    # Step 2: Finetune Models 1 and 2                                          #
    ############################################################################

    kwargs = {
        "X_train": dataset.get_split("X_train_tokenized"),
        "X_val": dataset.get_split("X_val_tokenized"),
        "y_train": dataset.get_split("y_train"),
        "y_val": dataset.get_split("y_val"),
    }
    finetune_model(model_index = 1, **kwargs)
    finetune_model(model_index = 2, **kwargs)

    ############################################################################
    # Step 3: Evaluate Models 1, 2 and 3 with test set                         #
    ############################################################################

    model_1_name = "model_1_1698170205_trainable-0_opt-rmsprop_lr-0.001_epoch-18_loss-0.7020_val_loss-0.5995_auc-0.6424_val_auc-0.7326f1-0.6847_val_f1-0.7557.keras"
    evaluate_tf_model(
        filename = model_1_name, 
        X_test = dataset.get_split("X_test_tokenized"),
        y_test = dataset.get_split("y_test"),
    )

    model_2_name = "model_2_1698343700_trainable-1_opt-adamax_lr-0.0001_epoch-06_loss-0.4313_val_loss-0.4682_auc-0.8911_val_auc-0.8619f1-0.8359_val_f1-0.8130.keras"
    evaluate_tf_model(
        filename = model_2_name, 
        X_test = dataset.get_split("X_test_tokenized"),
        y_test = dataset.get_split("y_test"),
    )

    model_3_name = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    evaluate_pt_model(
        filename = model_3_name, 
        X_test = dataset.get_split("X_test"),
        y_test = dataset.get_split("y_test"),
    )

    ############################################################################
    # Step 4: Save hidden states from Models 1, 2 and 3                        #
    ############################################################################

    save_tf_hidden_states(
        model_index = 1,
        weights_filename = model_1_name,
        dataset = dataset,
    )
    save_tf_hidden_states(
        model_index = 2,
        weights_filename = model_2_name,
        dataset = dataset,
    )
    save_pt_hidden_states(
        model_index = 3,
        pretrained_model_name = model_3_name,
        dataset = dataset,
    )

    ############################################################################
    # Step 5: Build token maps for Models 1, 2 and 3                           #
    ############################################################################

    token_maps = {
        1: "bert-base-uncased",
        2: "bert-base-uncased",
        3: "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    }

    kwargs = {
        "X_test_post_tokens": dataset.get_split("X_test_post_tokens"),
        "X_test_rationales": dataset.get_split("X_test_rationales"),
        "X_test_annotators": dataset.get_split("X_test_annotators"),
        "y_test": dataset.get_split("y_test")
    }
    
    for model_index, pretrained_model_name in token_maps.items():
        token_map = TokenMap().build(pretrained_model_name, **kwargs)
        token_map.save(F"model_{model_index}_token_map.csv")

    ############################################################################
    # Step 6: Run probes for Models 1, 2 and 3                                 #
    ############################################################################

    for model_index in range(1, 4):
        run_probes(
            model_index = model_index, 
            token_map_filename = F"model_{model_index}_token_map.csv"
        )

if __name__ == "__main__":
    main()