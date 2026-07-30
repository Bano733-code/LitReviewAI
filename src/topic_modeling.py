import pandas as pd

from gensim import corpora, models
from gensim.utils import simple_preprocess

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
def lda_topic_modeling(papers, num_topics=3):
    # Preprocess abstracts with stopword removal
    texts = [
        [word for word in simple_preprocess(p["abstract"]) if word not in stop_words]
        for p in papers if p.get("abstract")
    ]
    # Guard: if no texts or empty docs
    if not texts or all(len(t) == 0 for t in texts):
        return pd.DataFrame([])

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    topics = lda_model.print_topics(num_words=6)
    topic_data = []
    for topic_id, words in topics:
        topic_data.append({"Topic ID": topic_id, "Keywords": words})
    df = pd.DataFrame(topic_data)
    return df
