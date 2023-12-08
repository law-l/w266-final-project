
from utils.dataset import Dataset
from utils.model import evaluate_pt_model

ds = Dataset()


def slice(lst, idx):
    return [
        lst[0][:idx],
        lst[1][:idx],
        lst[2][:idx]
    ]
# filename = "model_1_1701730949_opt-adam_lr-1e-05_epoch-01_loss-9.4902_val_loss-0.0027_auc-0.5714_val_auc-0.0000_f1-0.0000_val_f1-0.0000.keras"
# model = Model(model_index = 1
#             #   weights_filename=filename
#               )
# idx = 10

# model_hist = model.finetune(
#     X_train = slice(ds.get_split("X_train_tokenized"), idx),
#     y_train = ds.get_split("y_train")[:idx],
#     X_val = slice(ds.get_split("X_val_tokenized"),idx),
#     y_val = ds.get_split("y_val")[:idx],
#     epochs = 5,
#     batch_size = 2,
# )
# model.evaluate(
#     X_test = slice(ds.get_split("X_test_tokenized"), idx),
#     y_test = ds.get_split("y_test")[:idx]
# )
# print(model_hist)
# print(model)


# model = Model(  )
# weights_filename="model_1_1701735392_trainable-0_opt-adam_lr-1e-05_epoch-01_loss-8.0195_val_loss-0.0985_auc-0.5238_val_auc-0.0000_f1-0.0000_val_f1-0.0000.keras"

# save_tf_hidden_states(weights_filename, slice(ds.get_split("X_test_tokenized"), 3))

# save_pt_hidden_states(2, "Hate-speech-CNERG/bert-base-uncased-hatexplain", ds.get_split("X_test"))


# token_map = TokenMap()
# token_map.build(
#     pretrained_model_name = "Hate-speech-CNERG/bert-base-uncased-hatexplain",
#     X_test_post_tokens = ds.get_split("X_test_post_tokens"),
#     X_test_rationales = ds.get_split("X_test_rationales"),
#     X_test_annotators = ds.get_split("X_test_annotators"),
#     y_test = ds.get_split("y_test"),
# )


# probe = Probe(2, token_map)

# target = "LGBTQ"
# words = [
# "agender","bulldykes","dyke","dykes","dykey","fag","faggot","faggotry","faggots","faggoty","faggy","fags","fgt","furfaggotry","gay","gays","girlfags","h0m0","homo","homophobe","homophobes","homophobia","homophobic","homosexual","homosexuality","homosexuals","jewfags","lesbian","lesbianism","lesbians","lesbophobia","lgbt","lgbtq","lgbtqwtf","newfag","pansy","queer","queerbaiting","queers","stormfags"
# ]
# probe.run(target, words)


evaluate_pt_model(
    X_test = ds.get_split("X_test")[:10],
    y_test = ds.get_split("y_test")[:10],
)