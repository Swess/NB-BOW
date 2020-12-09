import argparse
import csv
import os

from matplotlib.gridspec import GridSpec
from sklearn.metrics import plot_confusion_matrix

from classifiers import MultinomialNB_BOW
from data_loader import TSVLoader
import matplotlib.pyplot as plt
import numpy as np


def load_tsv(filename, fields=None):
    loader = TSVLoader()
    entries = loader.load(filename, fields)

    return entries


def plot_histogram(classes, data, axis, y_tick_count=25, color=None, label=None, width=0.8):
    if y_tick_count != -1 and y_tick_count < 1:
        raise ValueError("Y ticks count should be greater or equal than 1.")

    # Distribution
    n, bins = np.histogram(data, range(len(classes) + 1))
    axis.bar(classes, n, color=color, label=label, width=width)

    if y_tick_count != -1:
        max_count = np.amax(n)
        axis.set_yticks(np.arange(0, max_count + 1, int(max_count / y_tick_count)))


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_features(text_str: str):
    f = dict()
    words = text_str.split(" ")
    for w in words:
        w = w.lower()
        if w in f:
            f[w] += f[w]
        else:
            f[w] = 1

    return f


def extract_features_of_set(data):
    word_counts = dict()
    for l in data:
        for word in l["text"].split(" "):
            w = word.lower()
            if w in word_counts:
                word_counts[w] += word_counts[w]
            else:
                word_counts[w] = 1
    return word_counts


def train_and_test_model(model_name, vocabulary, training_data, test_data):
    # From current data layout
    train_features = [el["features"] for el in training_data]
    train_class = [el["cat"] for el in training_data]

    features = [el["features"] for el in test_data]
    expected = [el["cat"] for el in test_data]

    print()
    print(f"Processing for model '{model_name}'.")
    model = MultinomialNB_BOW(model_name, vocabulary)

    # Train
    print("Training the model...", end=" ")
    model.fit(train_features, train_class)
    print("Done.")

    # Prediction + Probabilities
    return model.predict(features)

    # Write predictions result to file
    # with open(f"{out_dir}" + model_name + ".txt", 'w', newline='') as resultsFile:
    #     writer = csv.writer(resultsFile, delimiter=',')
    #     res_rows = zip(range(len(expected)), predictions)
    #     writer.writerows(res_rows)

    # accuracy = (predictions == expected).sum() / len(expected)
    #
    # table_cells = []
    # macro_avg_f1 = 0
    # weighted_avg_f1 = 0
    #
    # # Per-class measurements
    # for class_i in range(nb_classes):
    #     # TP (True Positives)
    #     # FN (False Negatives)
    #     # FP (False Positives)
    #     class_expected = (expected == class_i)
    #     TP = np.logical_and((predictions == class_i), class_expected).sum()
    #     FN = np.logical_and((predictions != class_i), class_expected).sum()
    #     FP = np.logical_and((predictions == class_i), np.invert(class_expected)).sum()
    #
    #     precision = ((TP / (TP + FP)) if (TP + FP) > 0 else 1)
    #     recall = ((TP / (TP + FN)) if (TP + FN) > 0 else 1)
    #     f1 = ((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 1
    #
    #     # Fill table row
    #     row = []
    #     row.append(f"{TP}/{(TP + FP)} - {round(precision * 100, 3)} %")
    #     row.append(f"{TP}/{(TP + FN)} - {round(recall * 100, 3)} %")
    #     row.append(f"{round(f1 * 100, 3)} %")
    #     table_cells.append(row)
    #
    #     macro_avg_f1 += f1
    #     weighted_avg_f1 += f1 * class_expected.sum()
    #
    # macro_avg_f1 = macro_avg_f1 / nb_classes
    # weighted_avg_f1 = weighted_avg_f1 / len(expected)
    #
    # # Bar graph + Measurements tables
    # print("Plotting measurements...", end=" ")
    # fig = plt.figure()
    #
    # # Layout
    # gs = GridSpec(2, 4, figure=fig)
    # ax1 = fig.add_subplot(gs[0, :])
    # ax2 = fig.add_subplot(gs[1, :-2])
    #
    # bottom_right_grid = gs[1, -2:].subgridspec(5, 1)
    #
    # ax3 = fig.add_subplot(bottom_right_grid[0, :])
    # ax4 = fig.add_subplot(bottom_right_grid[1:, :])
    #
    # fig.set_figwidth(20)
    # fig.set_figheight(15)
    #
    # plot_histogram(classes, expected, ax1,
    #                color=(74 / 255, 159 / 255, 168 / 255), label='Target count', y_tick_count=-1, width=0.9)
    # plot_histogram(classes, predictions, ax1,
    #                color=(230 / 255, 208 / 255, 46 / 255), label='Predicted', y_tick_count=-1, width=0.4)
    #
    # ax1.grid(axis="y", linestyle='dotted')
    # ax1.legend()
    # ax1.set_title(f'{model_name} - Measurement results')
    # ax1.set_ylabel("Instance count")
    #
    # ax2.set_axis_off()
    # ax3.set_axis_off()
    #
    # ax2.table(cellText=table_cells,
    #           rowLabels=classes,
    #           colLabels=["Precision", "Recall", "f1"], loc='center')
    #
    # ax3.table(cellText=[[
    #     f"{round(accuracy * 100, 3)} %",
    #     f"{round(macro_avg_f1 * 100, 3)} %",
    #     f"{round(weighted_avg_f1 * 100, 3)} %"]],
    #     colLabels=["Accuracy", "Macro-Average f1", "Weighted-Average f1"], loc='center')
    #
    # plot_confusion_matrix(model, features, expected,
    #                       display_labels=classes,
    #                       cmap=plt.cm.Blues, ax=ax4)
    #
    # fig.show()
    # fig.savefig(out_dir + model_name + '-' + dataset_name + '-measurements.png')
    # print("Done.")


def make_trace(data, filename):
    lines = list(
        map(lambda el: f"{el[0]}  {el[1]}  {el[2]}  {el[3]}  {'correct' if el[1] == el[3] else 'wrong'}\n", data))
    with open(filename, 'w') as file:
        file.writelines(lines)


def make_eval(data, filename):
    TP = sum([1 if el[1] == el[3] else 0 for el in data])
    acc = TP / len(data)

    def _calc(data, t_class):
        other_class = "yes" if t_class == "no" else "no"
        FP = sum([1 if el[1] == t_class and el[3] == other_class else 0 for el in data])
        FN = sum([1 if el[1] == other_class and el[3] == t_class else 0 for el in data])

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = (P * R) / (P + R)

        return P, R, F1

    yes_P, yes_R, yes_F1 = _calc(data, "yes")
    no_P, no_R, no_F1 = _calc(data, "no")

    with open(filename, 'w') as file:
        file.write(f"{acc}\n")
        file.write(f"{yes_P}  {no_P}\n")
        file.write(f"{yes_R}  {no_R}\n")
        file.write(f"{yes_F1}  {no_F1}\n")


def main(argv):
    train_in_file, test_in_file, out_dir = args.training, args.test, args.output

    create_dir(out_dir)

    if not os.path.exists(train_in_file):
        print(f"Training set input file does not exist. ({train_in_file})")
        exit(1)
    if not os.path.exists(test_in_file):
        print(f"Test set input file does not exist. ({test_in_file})")
        exit(1)

    orig_train_data = load_tsv(train_in_file)
    orig_test_data = load_tsv(test_in_file, orig_train_data[0].keys())

    # Generate 2 Vocabulary (Used as features)
    voc = extract_features_of_set(orig_train_data)  # All words
    filtered_voc = dict(filter(lambda el: el[1] > 1, voc.items()))  # More than 1 instance

    # print("Plotting instances count...", end=" ")
    # figure, axe = plt.subplots(1)
    #
    # figure.suptitle(f'Instances count - NB_BOW')
    # figure.set_figheight(10)
    # figure.set_figwidth(14)
    #
    # # Display distribution
    # plot_histogram(voc.keys(), get_voc_of_set(train_data), axe)

    # Add titles / labels
    # axe.set_title("NB-BOW")
    # axe.set_ylabel('Count')
    #
    # figure.show()
    # figure.savefig(out_dir + 'instances_count_' + dt_name + '.png')
    # print("Done.")
    # print()

    # Extract features + data formatting
    train_data = [{
        "tweet_id": el["tweet_id"],
        "features": extract_features(el["text"]),
        "cat": 1 if el["q1_label"].lower() == "yes" else 0,
    } for el in orig_train_data]

    test_data = [{
        "tweet_id": el["tweet_id"],
        "features": extract_features(el["text"]),
        "cat": 1 if el["q1_label"].lower() == "yes" else 0,
    } for el in orig_test_data]

    # Process both models
    predictions_ov, probs_ov = train_and_test_model("NB-BOW-OV", list(voc.keys()), train_data, test_data)
    predictions_fv, probs_fv = train_and_test_model("NB-BOW-FV", list(filtered_voc.keys()), train_data, test_data)

    def _pack(preditions, probs):
        return list(zip(
            [el["tweet_id"] for el in orig_test_data],
            ["yes" if el == 1 else "no" for el in preditions],
            probs,
            [el["q1_label"].lower() for el in orig_test_data]
        ))

    data_ov = _pack(predictions_ov, probs_ov)
    data_fv = _pack(predictions_fv, probs_fv)

    make_trace(data_ov, out_dir + "trace_NB-BOW-OV.txt")
    make_trace(data_fv, out_dir + "trace_NB-BOW-FV.txt")

    make_eval(data_ov, out_dir + "eval_NB-BOW-OV.txt")
    make_eval(data_fv, out_dir + "eval_NB-BOW-FV.txt")

    print()
    print("All done!")
    print(f"Results and measurements can be found in ./{out_dir}")


if __name__ == "__main__":
    print("<<<<<<<<<<<<>>>>>>>>>>>>")
    print("COMP 472 - Assignment 3")
    print("Isaac Doré - 40043159")
    print("<<<<<<<<<<<<>>>>>>>>>>>>")
    print()

    arg_parser = argparse.ArgumentParser(
        description='Bag-Of-Word Naive Bayes Classifier. Made for COMP 472 Assignment 3.')

    arg_parser.add_argument("-tr", "--training", metavar="<training_set>", type=str,
                            help="Training dataset input file (Only supports .tsv). Default: _in/covid_training.tsv",
                            default="_in/covid_training.tsv")

    arg_parser.add_argument("-te", "--test", metavar="<test_set>", type=str,
                            help="Test dataset input file (Only supports .tsv). Default: _in/covid_test_public.tsv",
                            default="_in/covid_test_public.tsv")

    arg_parser.add_argument("-o", "--output", metavar="<output>", type=str,
                            help="Output directory relative to current working directory. Default: _out/",
                            default="_out/")

    args = arg_parser.parse_args()

    main(args)
