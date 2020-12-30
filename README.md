Implementation of a Bag-Of-Word Naive Bayes Classifier for multidimensional features vector analysis.

# Dependencies
To run the code, the requirements first needs to be installed.
If you are using a Virtual Environment (venv), activate it first.
Regardless if you are using a python Virtual Environment or not, you can install the project dependencies like so:
```
pip install -r requirements.txt
```

# Options
```
usage: main.py [-h] [-tr <training_set>] [-te <test_set>] [-o <output>]

Bag-Of-Word Naive Bayes Classifier.

optional arguments:
  -h, --help            show this help message and exit
  -tr <training_set>, --training <training_set>
                        Training dataset input file (Only supports .tsv).
                        Default: _in/covid_training.tsv
  -te <test_set>, --test <test_set>
                        Test dataset input file (Only supports .tsv). Default:
                        _in/covid_test_public.tsv
  -o <output>, --output <output>
                        Output directory relative to current working
                        directory. Default: _out/
```