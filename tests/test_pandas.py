import pandas as pd

from startupradar.transformers.pandas import (
    TfidfVectorizerDF,
    OneHotEncoderDF,
    ColumnTransformerDF,
)


def test_tfidf():
    vec = TfidfVectorizerDF()
    texts = ["bla bla bla", "this is text", "this is also text"]
    out = vec.fit_transform(texts)
    for word in " ".join(texts).split():
        word_not_in_cols_msg = f"word not found in columns ({word=}, {out.columns=})"
        assert word in out.columns, word_not_in_cols_msg


def test_one_hot_encoder():
    df = pd.DataFrame([["a"], ["b"], ["c"], ["a"]], columns=["letter"])
    t = OneHotEncoderDF(handle_unknown="ignore")
    result = t.fit_transform(df)
    columns_expected = ["letter_a", "letter_b", "letter_c"]
    assert result.columns.values.tolist() == columns_expected

    df = pd.DataFrame([["a"], ["b"], ["c"], ["d"]], columns=["letter"])
    assert t.transform(df).columns.values.tolist() == columns_expected


def test_column_transformer():
    df = pd.DataFrame({"suffix": ["de", "com", "com", "at"], "length": [1, 2, 3, 4]})
    transformer = ColumnTransformerDF(
        transformers=[
            ("one_hot", OneHotEncoderDF(), ["suffix"]),
            ("pass", "passthrough", ["length"]),
        ]
    )
    df_out = transformer.fit_transform(df)
    assert len(df_out.columns) == 4
