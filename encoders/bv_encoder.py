from typing import Sequence, AnyStr, List, Union
import numpy as np
import math
import string
import itertools
from collections import defaultdict

class BitVecEncoder():

    def __init__(self,
                 ngram_size, charset = None):
        """
        Constuctor for the BFEncoder class.
        :param ngram_size: Size of the ngrams.
        :param charset: An iterable containing all possible characters in the alphabet.
        """
        self.ngram_size = ngram_size
        if charset != None:
            self.charset = charset
        else:
            self.charset = string.printable + "§öäüßáéàèâêôûïÿëÄÖÜ"
        ngram_universe =[''.join(p) for p in itertools.product(self.charset, repeat=2)]
        self.universe_bigram_count = len(ngram_universe)
        
        
        # Create dictionary for indices
        self.ngram_ind = defaultdict(lambda : self.universe_bigram_count)

        for i, n in enumerate(ngram_universe):
            self.ngram_ind[n] = i

    def _to_onehot(self, data):
        ngram_hotenc = np.zeros((len(data), self.universe_bigram_count+1), dtype=np.uint8)

        # Insert the data into the prepared matrices.
        i = 0
        # Iterate over all records in the data
        for ngr in data:
            # Iterate over the record's n-grams and set the corresponding cell (row defined by record, column defined by index of unique n-gram) to 1.
            for n in ngr:
                ngram_hotenc[i, self.ngram_ind[n]] = 1
            i += 1
        return ngram_hotenc

    def encode(self, data, pass_ngrams=False):

        if not pass_ngrams:            
            data = [''.join(d) for d in data]
            # Split each string in the data into a list of qgrams to process
            data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]

        return self._to_onehot(data)