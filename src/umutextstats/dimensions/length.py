from umutextstats.dimensions.base import BaseDimension


class LengthDimension(BaseDimension):
    def compute(self, df):
        return df[self.input_column].fillna("").astype(str).str.len()