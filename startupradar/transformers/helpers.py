from sklearn.base import TransformerMixin


class TurnToIter(TransformerMixin):
    """Turn the text column to an iterator for ingestion into TfidfVectorizer"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        list_ = X["text"].values.tolist()
        return list_
