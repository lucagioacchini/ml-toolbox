# type: ignore

from gensim.models import Word2Vec 
from multiprocessing import cpu_count
import pandas as pd
import numpy as np

class iWord2Vec():
    def __init__(self, c=5, e=64, epochs=1, source=None, destination=None, 
                                                                      seed=15):
        self.context_window = c
        self.embedding_size = e
        self.epochs = epochs
        self.seed = seed
        
        self.model = None

        self.source = source
        self.destination = destination

        if type(source) != type(None):
            self.load_model()            
                
    def train(self, corpus, save=False):
        self.model = Word2Vec(sentences=corpus, vector_size=self.embedding_size, 
                              window=self.context_window, epochs=self.epochs, 
                              workers=cpu_count(), min_count=0, sg=1, 
                              negative=5, sample=0, seed=self.seed)
        if save:
            self.model.save(f'{self.destination}.model')

    def load_model(self):
        self.model = Word2Vec.load(f'{self.source}.model')


    def get_embeddings(self, ips=None):
        if type(ips)==type(None):
            ips = [x for x in self.model.wv.index_to_key]
        embeddings = self.model.wv.vectors    
        embeddings = pd.DataFrame(embeddings, index=ips)
                    
        return embeddings
    

    def update(self, corpus, save=False):
        self.model.build_vocab(corpus, update=True, trim_rule=None)
        self.model.train(corpus, total_examples=self.model.corpus_count, 
                         epochs=self.epochs)
        if save:
            self.model.save(f'{self.destination}.model')

    def del_embeddings(self, to_drop, mname=None):
        idx = np.isin(self.model.wv.index2word, to_drop)
        idx = np.where(idx==True)[0]
        self.model.wv.index2word = list(np.delete(self.model.wv.index2word, 
                                                  idx, axis=0))
        self.model.wv.vectors = np.delete(self.model.wv.vectors, idx, axis=0)
        self.model.trainables.syn1neg = np.delete(self.model.trainables.syn1neg, 
                                                  idx, axis=0)
        list(map(self.model.wv.vocab.__delitem__, 
                 filter(self.model.wv.vocab.__contains__,to_drop)))


        for i, word in enumerate(self.model.wv.index2word):
            self.model.wv.vocab[word].index = i

        if type(mname)!=type(None):
            self.model.save(f'{mname}.model')