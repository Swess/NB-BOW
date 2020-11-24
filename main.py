import getopt
import os
import csv
import sys

from matplotlib.gridspec import GridSpec
from sklearn.metrics import plot_confusion_matrix

from configs import *
from data_loader import CSVLoader
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

from tuning import tune_models

out_dir = "_out/"


def load_dataset(filename):
    class_loader = CSVLoader()
    entries = class_loader.load(filename)
    entries.sort(key=lambda x: int(x["index"]))
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


def train_and_train_model(model, dataset_name, classes, training_data, test_data):
    classifiers = CLASSIFIERS_DEFAULT

    if dataset_name.lower() in ("ds1", "dataset1"):
        classifiers = CLASSIFIERS_1
    elif dataset_name.lower() in ("ds2", "dataset2"):
        classifiers = CLASSIFIERS_2

    if model.lower() == "all":
        models = classifiers.keys()
    elif model == "Best-DT":
        models = ["Base-DT", "Best-DT"]
    else:
        models = [model]

    print("====== Classifiers ======")

    nb_classes = len(classes)
    class_names = [x["symbol"] for x in classes]

    # Data layout
    # x : [n_samples, n_features]
    # y : [n_samples]
    train_features = training_data[:, 0:-1]
    train_indexes = training_data[:, -1]

    features = test_data[:, :]
    expected = test_data[:, -1]

    for model_name in models:
        print()
        print(f"Processing for model '{model_name}'.")
        model = classifiers[model_name]

        # Train
        print("Training the model...", end=" ")
        model.fit(train_features, train_indexes)
        print("Done.")

        predictions = model.predict(features)

        # Write predictions result to file
        with open(out_dir + model_name + "-" + dataset_name + ".csv", 'w', newline='') as resultsFile:
            writer = csv.writer(resultsFile, delimiter=',')
            res_rows = zip(range(len(expected)), predictions)
            writer.writerows(res_rows)

        continue

        accuracy = (predictions == expected).sum() / len(expected)

        table_cells = []
        macro_avg_f1 = 0
        weighted_avg_f1 = 0

        # Per-class measurements
        for class_i in range(nb_classes):
            # TP (True Positives)
            # FN (False Negatives)
            # FP (False Positives)
            class_expected = (expected == class_i)
            TP = np.logical_and((predictions == class_i), class_expected).sum()
            FN = np.logical_and((predictions != class_i), class_expected).sum()
            FP = np.logical_and((predictions == class_i), np.invert(class_expected)).sum()

            precision = ((TP / (TP + FP)) if (TP + FP) > 0 else 1)
            recall = ((TP / (TP + FN)) if (TP + FN) > 0 else 1)
            f1 = ((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 1

            # Fill table row
            row = []
            row.append(f"{TP}/{(TP + FP)} - {round(precision * 100, 3)} %")
            row.append(f"{TP}/{(TP + FN)} - {round(recall * 100, 3)} %")
            row.append(f"{round(f1 * 100, 3)} %")
            table_cells.append(row)

            macro_avg_f1 += f1
            weighted_avg_f1 += f1 * class_expected.sum()

        macro_avg_f1 = macro_avg_f1 / nb_classes
        weighted_avg_f1 = weighted_avg_f1 / len(expected)

        # Bar graph + Measurements tables
        print("Plotting measurements...", end=" ")
        fig = plt.figure()

        # Layout
        gs = GridSpec(2, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :-2])

        bottom_right_grid = gs[1, -2:].subgridspec(5, 1)

        ax3 = fig.add_subplot(bottom_right_grid[0, :])
        ax4 = fig.add_subplot(bottom_right_grid[1:, :])

        fig.set_figwidth(20)
        fig.set_figheight(15)

        # Histogram not descriptive enough
        plot_histogram(class_names, expected, ax1,
                       color=(74 / 255, 159 / 255, 168 / 255), label='Target count', y_tick_count=-1, width=0.9)
        plot_histogram(class_names, predictions, ax1,
                       color=(230 / 255, 208 / 255, 46 / 255), label='Predicted', y_tick_count=-1, width=0.4)

        ax1.grid(axis="y", linestyle='dotted')
        ax1.legend()
        ax1.set_title(f'{dataset_name} - {model_name} - Measurement results')
        ax1.set_ylabel("Instance count")

        ax2.set_axis_off()
        ax3.set_axis_off()

        ax2.table(cellText=table_cells,
                  rowLabels=class_names,
                  colLabels=["Precision", "Recall", "f1"], loc='center')

        ax3.table(cellText=[[
            f"{round(accuracy * 100, 3)} %",
            f"{round(macro_avg_f1 * 100, 3)} %",
            f"{round(weighted_avg_f1 * 100, 3)} %"]],
            colLabels=["Accuracy", "Macro-Average f1", "Weighted-Average f1"], loc='center')

        plot_confusion_matrix(model, features, expected,
                              display_labels=class_names,
                              cmap=plt.cm.Blues, ax=ax4)

        fig.show()
        fig.savefig(out_dir + model_name + '-' + dataset_name + '-measurements.png')
        print("Done.")


def parse_args(argv):
    state = {"mode": "normal", "model": "all"}
    try:
        opts, args = getopt.getopt(argv, "hm:", ["model="])
    except getopt.GetoptError:
        print('-m <model_name | default:all>')
        sys.exit(2)

    for arg in args:
        if arg == 'tune':
            state["mode"] = "tune"

    for opt, arg in opts:
        if opt == '-h' or opt == '--help':
            print('-m <model_name | default:all>')
            sys.exit()
        elif opt in ("-m", "--model"):
            state["model"] = arg

    return state


def main(argv):
    state = parse_args(argv)
    create_dir(out_dir)

    print("COMP 472 - Assignment 1")
    print("Isaac Doré - 40043159")
    print("==========")
    print()

    dt_name = input("Please enter the name of your dataset (Ex: DS1 | DS2): ")
    dt_info_filename = input("Please enter the filename of the dataset's information: ")
    dt_train_filename = input("Please enter the filename of the dataset's training data: ")
    dt_test_filename = input("Please enter the filename of the dataset's test data with label: ")
    dt_validation_filename = input("Please enter the filename of the dataset's validation data: ")

    try:
        # Load Dataset1
        print("Loading dataset information...", end=" ")
        classes = load_dataset("datasets/" + dt_info_filename)
        print("Done.")

        print("Loading training data...", end=" ")
        training_data = genfromtxt("datasets/" + dt_train_filename, delimiter=',', dtype=int)
        print("Done.")

        print("Loading test data...", end=" ")
        test_data = genfromtxt("datasets/" + dt_test_filename, delimiter=',', dtype=int)
        print("Done.")

        print("Loading validation data...", end=" ")
        val_data = genfromtxt("datasets/" + dt_validation_filename, delimiter=',', dtype=int)
        print("Done.")
    except Exception as e:
        sys.tracebacklimit = None
        print()
        print(f"ERROR: Failed to load one of the given input file. Please verify and restart.")
        print(e)
        exit(0)

    if state["mode"] == "tune":
        # model = (state["model"] if state["model"] else "DT")
        while True:
            m = input("Choose which model to tune for <DT,MLP>:")
            if m.lower() in ("dt", "mlp"):
                break
            print("Invalid choice...")

        tune_models(training_data, val_data, m)
        return

    print("Plotting instances count...", end=" ")
    figure, axe = plt.subplots(1)

    figure.suptitle(f'Instances count - {dt_name}')
    figure.set_figheight(10)
    figure.set_figwidth(14)

    # Display distribution
    plot_histogram([x["symbol"] for x in classes], training_data[:, -1], axe)

    # Add titles / labels
    axe.set_title(dt_name)
    axe.set_ylabel('Count')

    figure.show()
    figure.savefig(out_dir + 'instances_count_'+dt_name+'.png')
    print("Done.")
    print()

    # Process each datasets
    train_and_train_model(state["model"], dt_name, classes, training_data, test_data)

    print()
    print("All done!")
    print(f"Results and measurements can be found in ./{out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
