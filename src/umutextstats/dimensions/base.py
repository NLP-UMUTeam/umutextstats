from abc import ABC, abstractmethod


class BaseDimension(ABC):
    def __init__(self, key: str, input_column: str = "text_norm"):
        self.key = key
        self.input_column = input_column

    @abstractmethod
    def compute(self, df):
        pass