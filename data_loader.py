from abc import abstractmethod
import csv


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
