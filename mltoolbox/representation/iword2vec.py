# type: ignore

from gensim.models import Word2Vec
from multiprocessing import cpu_count
import pandas as pd
import numpy as np


class iWord2Vec():
    """An incremental Word2Vec implementation supporting model updates and embedding management.

    This class implements Word2Vec with additional functionality for incremental training,
    embedding management, and model persistence. It uses skip-gram with negative sampling.

    Parameters:
        c (int, optional): Context window size for word sequences. Defaults to 5.
        e (int, optional): Dimensionality of word embeddings. Defaults to 64.
        epochs (int, optional): Number of training epochs. Defaults to 1.
        source (str, optional): Path to load existing model. Defaults to None.
        destination (str, optional): Path to save model. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 15.

    Attributes:
        context_window (int): Size of context window
        embedding_size (int): Dimensionality of embeddings
        epochs (int): Number of training epochs
        seed (int): Random seed
        model (Word2Vec): The underlying gensim Word2Vec model
        source (str): Path for loading model
        destination (str): Path for saving model

    Methods:
        train(corpus, save): Trains Word2Vec model on input corpus
        load_model(): Loads model from source path
        get_embeddings(ips, emb_path): Retrieves word embeddings
        update(corpus, save): Incrementally updates model with new data
        del_embeddings(to_drop, mname): Removes specified embeddings from model

    Example:
        >>> # Define the corpus as a list of list of strings
        >>> corpus = [['Lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur'], 
        ...           ['adipiscing', 'elit', 'sed', 'do'], 
        ...           ['eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore']] 
        >>> from mltoolbox.representation import iWord2Vec
        >>> # Initialize the model
        >>> word2vec = iWord2Vec(c=2, e=10, epochs=1, seed=15)
        >>> # Train the initialized model
        >>> word2vec.train(corpus)
        >>> # Retrieve the embeddings after the first training
        >>> embeddings = word2vec.get_embeddings()
        >>> print(embeddings.shape) # Get the vocabulary size and the embeddings size
        >>> embeddings.head(3)

        >>> (17, 10)
        >>>             0         1         2         3         4         5         6  \
        ... dolore  0.086249  0.038482  0.041049  0.063226 -0.051581 -0.031196 -0.059515   
        ... elit    0.093712 -0.070643  0.096178  0.043789 -0.006850 -0.030944  0.039167   
        ... ipsum  -0.056756  0.056412 -0.080288  0.068822 -0.071940  0.010958  0.004222   

        ...             7         8         9  
        ... dolore -0.091163 -0.011349  0.014431  
        ... elit   -0.008497 -0.046373  0.095279  
        ... ipsum   0.088425  0.077777 -0.096294  

        >>> # Get a new corpus with new words
        >>> corpus = [['magna', 'aliqua', 'Ut', 'enim', 'ad', 'minim', 'veniam', 'quis'],
        ...           ['nostrud', 'exercitation', 'ullamco', 'laboris', 'nisi', 'ut']]
        >>> # Update the existing model with the new corpus
        >>> word2vec.update(corpus)
        >>> # Retrieve the updated embeddings
        >>> new_embeddings = word2vec.get_embeddings()
        >>> print(new_embeddings.shape) # Get the vocabulary and the embeddings size

        >>> (30, 10)

        >>> # Remove the embeddings for a word
        >>> word2vec.del_embeddings(['dolore'])
        >>> # Check the new vocabulary size
        >>> final_embeddings = word2vec.get_embeddings()
        >>> print(final_embeddings.shape)

        >>> (29, 10)
    """

    def __init__(self, c: int = 5, e: int = 64, epochs: int = 1, source: str = None, destination: str = None,
                 seed: int = 15):
        self.context_window = c
        self.embedding_size = e
        self.epochs = epochs
        self.seed = seed

        self.model = None

        self.source = source
        self.destination = destination

        if type(source) != type(None):
            self.load_model()

    def train(self, corpus: list, save: bool = False):
        """Train the Word2Vec model on input corpus.

        Parameters:
            corpus (list): List of tokenized sentences/sequences
            save (bool, optional): Whether to save model to destination path. Defaults to False.

        Notes:
            Uses skip-gram with negative sampling, parallel training on all CPU cores,
            and no minimum word count threshold.
        """
        self.model = Word2Vec(sentences=corpus, vector_size=self.embedding_size,
                              window=self.context_window, epochs=self.epochs,
                              workers=cpu_count(), min_count=0, sg=1,
                              negative=5, sample=0, seed=self.seed)
        if save:
            self.model.save(f'{self.destination}.model')

    def load_model(self):
        """Load a previously saved Word2Vec model from the source path."""
        self.model = Word2Vec.load(f'{self.source}.model')

    def get_embeddings(self, ips: list = None, emb_path: str = None):
        """Retrieve word embeddings as a DataFrame.

        Parameters:
            ips (list, optional): List of words to get embeddings for. If None, gets all. Defaults to None.
            emb_path (str, optional): Path to save embeddings CSV. Defaults to None.

        Returns:
            pandas.DataFrame: Word embeddings with words as index
        """
        if type(ips) == type(None):
            ips = [x for x in self.model.wv.index_to_key]
        embeddings = self.model.wv.vectors
        embeddings = pd.DataFrame(embeddings, index=ips)

        if type(emb_path) != type(None):
            embeddings.to_csv(emb_path)

        return embeddings

    def update(self, corpus: list, save: bool = False):
        """Incrementally update the model with new text data.

        Parameters:
            corpus (list): List of new tokenized sentences/sequences
            save (bool, optional): Whether to save updated model. Defaults to False.
        """
        self.model.build_vocab(corpus, update=True, trim_rule=None)
        self.model.train(corpus, total_examples=self.model.corpus_count,
                         epochs=self.epochs)
        if save:
            self.model.save(f'{self.destination}.model')

    def del_embeddings(self, to_drop: list, mname: str = None):
        """Remove specified word embeddings from the model.

        Parameters:
            to_drop (list): List of words to remove from the model
            mname (str, optional): Path to save updated model. Defaults to None.

        Notes:
            Updates all necessary model components including vocabularies,
            embedding matrices, and negative sampling weights.
        """
        idx = np.isin(self.model.wv.index_to_key, to_drop)
        idx = np.where(idx == True)[0]
        self.model.wv.index_to_key = list(np.delete(self.model.wv.index_to_key,
                                                    idx, axis=0))
        self.model.wv.vectors = np.delete(self.model.wv.vectors, idx, axis=0)
        self.model.syn1neg = np.delete(self.model.syn1neg, idx, axis=0)
        list(map(self.model.wv.key_to_index.__delitem__,
                 filter(self.model.wv.key_to_index.__contains__, to_drop)))

        for i, word in enumerate(self.model.wv.index_to_key):
            self.model.wv.key_to_index[word] = i

        if type(mname) != type(None):
            self.model.save(f'{mname}.model')
