from utils import *

if __name__ == '__main__':
    # load previous trained model
    saved_record = None
    # with open('saved_models/simple_conv_net_layers_char_level_train_embed.model', 'rb') as f:
    with open('saved_models/wikipedia_conv_deep_word_level', 'rb') as f:
    # with open('saved_models/simple_conv_net_layers_char_level_train_embed.model', 'rb') as f:
        saved_record = pickle.load(f)

    model = saved_record["model"]
    print(model)

    train_loss, val_loss, pack_train_scores, pack_val_scores = saved_record['trends']

    plot_trend(train_loss, val_loss, 'Loss')

    for check_task in ['label1', 'label2', 'label3']:
        train_trend = pack_train_scores['f1_score'][check_task]
        val_trend = pack_val_scores['f1_score'][check_task]
        plot_trend(train_trend, val_trend, "F1 Score {}".format(check_task))

        train_trend = pack_train_scores['auc_score'][check_task]
        val_trend = pack_val_scores['auc_score'][check_task]
        plot_trend(train_trend, val_trend, "AUC score {}".format(check_task))

        train_trend = [1-i for i in pack_train_scores['accuracy'][check_task]]
        val_trend = [1-i for i in pack_val_scores['accuracy'][check_task]]
        plot_trend(train_trend, val_trend, "erro rate {}".format(check_task))
