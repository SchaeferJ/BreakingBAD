# Make sure you have numba installed in your environment:

import numpy as np
import string
import pickle
import os
from hashlib import md5
from collections import defaultdict
from itertools import product
import numba as nb
from scipy.special import binom
from typing import Sequence, AnyStr, List, Tuple, Any, Union


@nb.njit(parallel=True, fastmath=True, cache=True)
def par_encode(data_he, mh_length, tab_hashes):
    enc_data = np.empty((data_he.shape[0], mh_length), dtype=np.bool_)

    for i in nb.prange(data_he.shape[0]):
        record_ngrams_mask = data_he[i]
        present_ngram_indices = np.where(record_ngrams_mask)[0]

        if len(present_ngram_indices) == 0:
            enc_data[i, :] = False
            continue

        selected_hashes = tab_hashes[:, present_ngram_indices, :]
        selected_bits = (selected_hashes % 2).astype(np.uint8)  # Cast to int for sum
        # Use integer arithmetic to avoid floating point issues with fastmath
        sum_of_bits = np.sum(selected_bits, axis=1)
        num_present_ngrams = selected_bits.shape[1]
        intermediate_hashes = (2 * sum_of_bits >= num_present_ngrams)
        enc_data[i, :] = (np.sum(intermediate_hashes, axis=0) % 2).astype(np.bool_)

    return enc_data


class SaulEncoder():

    def __init__(self, secret, ngram_size, mh_length, num_hash_func, workers=-1, charset=string.printable):
        self.secret = secret
        self.ngram_size = ngram_size
        self.mh_length = mh_length
        self.num_hash_func = num_hash_func
        self.workers = workers if workers != -1 else os.cpu_count() - 1

        self.charset = charset
        random_seed = int(md5(self.secret.encode()).hexdigest(), 16) % (2 ** 32 - 1)

        ngram_universe = [''.join(p) for p in product(self.charset, repeat=self.ngram_size)]
        self.universe_ngram_count = len(ngram_universe)

        self.ngram_ind = defaultdict(lambda: self.universe_ngram_count)
        for i, n in enumerate(ngram_universe):
            self.ngram_ind[n] = i

        rng = np.random.default_rng(random_seed)
        self.tab_hashes = rng.integers(0, 2 ** 64 - 1,
                                       (self.num_hash_func, self.universe_ngram_count + 1, self.mh_length),
                                       dtype=np.uint64)

    def encode(self, data, pass_oh=False):
        if not pass_oh:
            # Preprocessing and n-gramming
            # clean_data = ["".join(filter(lambda char: char in self.charset, d)).replace(" ", "").lower() for d in data]
            # ngram_data = [[s[i:i + self.ngram_size] for i in range(len(s) - self.ngram_size + 1)] for s in clean_data]
            clean_data = ["".join(d).replace(" ", "").lower() for d in data]
            ngram_data = [[s[i:i + 2] for i in range(len(s) - 2 + 1)] for s in clean_data]
    
            data_he = np.zeros((len(ngram_data), self.universe_ngram_count + 1), dtype=bool)
            for i, ngr_list in enumerate(ngram_data):
                for ngr in ngr_list:
                    data_he[i, self.ngram_ind[ngr]] = True
    
            original_num_threads = nb.get_num_threads()
            nb.set_num_threads(self.workers)
    
            encoded_data = par_encode(data_he, self.mh_length, self.tab_hashes)
    
            nb.set_num_threads(original_num_threads)
        else:
            original_num_threads = nb.get_num_threads()
            nb.set_num_threads(self.workers)
    
            encoded_data = par_encode(data, self.mh_length, self.tab_hashes)
    
            nb.set_num_threads(original_num_threads)
        return encoded_data

    def to_onehot(self, data):
        # Preprocessing and n-gramming
        # clean_data = ["".join(filter(lambda char: char in self.charset, d)).replace(" ", "").lower() for d in data]
        # ngram_data = [[s[i:i + self.ngram_size] for i in range(len(s) - self.ngram_size + 1)] for s in clean_data]
        clean_data = ["".join(d).replace(" ", "").lower() for d in data]
        ngram_data = [[s[i:i + 2] for i in range(len(s) - 2 + 1)] for s in clean_data]

        data_he = np.zeros((len(ngram_data), self.universe_ngram_count + 1), dtype=bool)
        for i, ngr_list in enumerate(ngram_data):
            for ngr in ngr_list:
                data_he[i, self.ngram_ind[ngr]] = True

        return data_he
