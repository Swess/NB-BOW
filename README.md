[Github Repo](https://github.com/Swess/COMP472_A3)

# COMP 472 - Assignment 3
**Author:** Isaac Doré - 40043159

# Running
The program will prompt you for all required filenames containing the datasets and information.
The filename are relative to the `./datasets/` folder.

*Note: This program loads datasets from comma-separated CSV files only.*

The results computed are displayed in generated images for each model, stored in the `_out/` directory.

The program contains already optimized models for the 2 given datasets.
And these models are selected based on the provided dataset name. Namely `DS1` or `DS2`.

# Options
The program also expose some arguments & options if need be.

You can filter which model to run against the dataset with `-m <model_name>`. By default, they are all run.
Available models:
- "GNB"
- "Base-DT"
- "Best-DT"
- "PER"
- "Base-MLP"
- "Best-MLP"

The argument `tune` will run the program in tuning mode for the given dataset and will let you choose which model to optimize between DecisionTree or MultiLayerPerceptron.

*Note: This is VERY time consuming as the program currently runs multiple times for multiple scoring, and are all displayed.*
> ('accuracy', 'precision_macro','precision_weighted', 'recall_macro', 'recall_weighted', 'f1_weighted', 'f1_macro')

