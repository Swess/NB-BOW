from abc import abstractmethod
import csv

from sklearn.naive_bayes import MultinomialNB

MultinomialNB

class ILoader:
    @abstractmethod
    def load(self, filename: str):
        pass


class CSVLoader(ILoader):

    def __init__(self) -> None:
        super().__init__()

    def load(self, filename: str):
        res = []
        with open(filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                res.append(row)

        return res


class TSVLoader(ILoader):

    def __init__(self) -> None:
        super().__init__()

    def load(self, filename: str, fieldnames=None):
        res = []
        with open(filename, 'r', encoding="utf8") as tsv_file:
            reader = csv.DictReader(tsv_file, dialect='excel-tab', fieldnames=fieldnames)

            for row in reader:
                res.append(row)

        return res
