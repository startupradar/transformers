from startupradar.transformers.pandas import TfidfVectorizerDF


def test_tfidf():
    vec = TfidfVectorizerDF()
    texts = ["bla bla bla", "this is text", "this is also text"]
    out = vec.fit_transform(texts)
    for word in " ".join(texts).split():
        word_not_in_cols_msg = f"word not found in columns ({word=}, {out.columns=})"
        assert word in out.columns, word_not_in_cols_msg
