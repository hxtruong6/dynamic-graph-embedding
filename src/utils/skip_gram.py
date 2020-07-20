from gensim.models import Word2Vec
from joblib import cpu_count


class SkipGram(Word2Vec):
    def __init__(self, sentences, embedding_size, window_size):
        """

        :param sentences:
        :param embedding_size:
        :param window_size:

        Default params:
        LINK: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
        min_count (int, optional) – Ignores all words with total frequency lower than this.
        workers (int, optional) – Use these many worker threads to train the model (=faster training with multicore machines).
        sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
        hs ({0, 1}, optional) – If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
        """
        # TODO: add more parameter here

        super(SkipGram, self).__init__(sentences=sentences, size=embedding_size, window=window_size, sg=1, hs=1,
                                       min_count=0, workers=cpu_count())
