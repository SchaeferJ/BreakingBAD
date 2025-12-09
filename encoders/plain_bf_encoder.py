# Encodes a given using bloom filters for PPRL
import gc
import os
import pickle
from typing import Sequence, AnyStr, List, Union
from .encoder import Encoder
import numpy as np
from hashlib import md5
from joblib import Parallel, delayed


def numpy_pairwise_combinations(x):
    # https://carlostgameiro.medium.com/fast-pairwise-combinations-in-numpy-c29b977c33e2
    tmp = np.triu_indices(len(x), k=1)
    gc.collect()
    idx = np.stack(tmp, axis=-1)
    return x[idx]


def calc_metrics(uids, enc, metric, sim, inds):
    bf_length = enc.shape[1]

    left_matr = enc[inds[:, 0]]
    right_matr = enc[inds[:, 1]]

    del enc
    # Compute number of matching bits and overall bits
    and_sum = np.sum(np.logical_and(left_matr, right_matr), axis=1)
    # Adjust for length of BF
    and_sum = np.log(1 - (and_sum / bf_length))

    if metric == "jaccard":
        or_sum = np.sum(np.logical_or(left_matr, right_matr), axis=1)
        or_sum = np.log(1 - (or_sum / bf_length))
        pw_metrics = and_sum / or_sum
    else:
        hamming_wt_right = np.sum(right_matr, axis=1)
        hamming_wt_left = np.sum(left_matr, axis=1)
        hamming_wt_right = np.log(1 - (hamming_wt_right / bf_length))
        hamming_wt_left = np.log(1 - (hamming_wt_left / bf_length))
        pw_metrics = 2 * and_sum / (hamming_wt_left + hamming_wt_right)

    if not sim:
        pw_metrics = 1 - pw_metrics

    metrics_with_uids = np.zeros((len(inds), 3), dtype=np.float32)
    metrics_with_uids[:, 0] = uids[inds[:, 0]]
    metrics_with_uids[:, 1] = uids[inds[:, 1]]
    metrics_with_uids[:, 2] = pw_metrics
    return metrics_with_uids


class BFEncoder(Encoder):

    def __init__(self, secret: AnyStr, filter_size: int, bits_per_feature: Union[int, Sequence[int]],
                 ngram_size: Union[int, Sequence[int]], diffusion=False, eld_length=None,
                 t=None, xor_reduce=False, workers=-1):
        """
        Constuctor for the BFEncoder class.
        :param secret: Secret to be used in the HMAC
        :param filter_size: Bloom Filter Size
        :param bits_per_feature: Bits to be set per feature (=Number of Hash functions). If an integer is passed, the
        same value is used for all attributes. If a list of integers is passed, one value per attribute must be
        specified.
        :param ngram_size: Size of the ngrams. If an integer is passed, the same value is used for all attributes. If a
        list of integers is passed, one value per attribute must be specified.
        :param diffusion: Specifies whether diffusion should be applied to the Bloom Filter
        See paper of Armknecht, Heng and Schnell for details: https://doi.org/10.56553/POPETS-2023-0054
        :param eld_length: Length of Bloom Filter after diffusion
        :param t: Number of bits to be xor-ed for positions in Bloom Filter.
        :param xor_reduce: If true, XOR is used to combine the individual hash functions, else OR

        """
        self.secret = secret
        self.filter_size = filter_size
        self.bits_per_feature = bits_per_feature
        self.ngram_size = ngram_size
        self.diffusion = diffusion
        self.eld_length = eld_length
        self.t = t
        self.indices = None
        self.xor_reduce = xor_reduce
        self.workers = os.cpu_count() if workers == -1 else workers

        if diffusion:
            assert eld_length is not None, "ELD length must be specified if diffusion is enabled"
            assert t is not None, "Number of XORed bits must be specified if diffusion is enabled"
            assert self.t <= self.filter_size, "Cannot select more bits for XORing than are present in the BF!"

            if type(self.secret) == str:
                random_seed = int(md5(self.secret.encode()).hexdigest(), 16) % (2 ** 32 - 1)
            else:
                random_seed = self.secret
            np.random.seed(random_seed)
            self.indices = []
            available_indices = np.arange(self.filter_size)
            # Generate t random random indices per bit in the diffused BF. Bits at this position of the BF are XORed
            # to set the bit in the diffused BF. Refer to Algorithm 1 in the paper
            for j in range(self.eld_length):
                if available_indices.shape[0] >= self.t:
                    tmp = np.random.choice(available_indices, size=self.t, replace=False)
                    available_indices = np.setdiff1d(available_indices, tmp)
                else:
                    tmp = available_indices
                    available_indices = np.arange(self.filter_size)
                    tt = np.random.choice(np.setdiff1d(available_indices, tmp), size=self.t - tmp.shape[0],
                                          replace=False)
                    available_indices = np.setdiff1d(available_indices, tmp)
                    tmp = np.union1d(tmp, tt)

                self.indices.append(tmp)


    def encode(self, data: Sequence[Sequence[Union[str, int]]], pass_ngrams=False) -> np.ndarray:
        """
        Encodes the given data using bloom filter encoding (CLKHash), returns a MxN array of bits, where M is the number
        of records and N is the size of the bloom filter specified in schema.
        :param data: Data to encode. A list of lists: Inner list represents records with integers or strings as values.
        :return: a MxN array of bits, where M is the number of records (length of data) and N is the size of the bloom
        filter.
        """
        if not type(self.bits_per_feature) == int:
            assert len(self.bits_per_feature) == len(data[0]), "Invalid number (" + str(len(self.ngram_size)) + ") of " \
                                                                                                                "values for bits_per_feature. Must either be one value or one value per attribute (" + str(
                len(data[0])) + ")."

        if not type(self.ngram_size) == int:
            assert len(self.ngram_size) == len(data[0]), "Invalid number (" + str(len(self.ngram_size)) + ") of " \
                                                                                                          "values for ngram_size. Must either be one value or one value per attribute (" + str(
                len(data[0])) + ")."

        if not pass_ngrams:
            data = ["".join(d) for d in data]
            # Split each string in the data into a list of qgrams to process
            data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]

        reduce_func = np.logical_xor if self.xor_reduce else np.logical_or

        enc_data = np.zeros((len(data), self.filter_size), dtype=bool)

        for i, d in enumerate(data):
            tmp_hashes = np.zeros((self.bits_per_feature, self.filter_size), dtype=bool)
            for ngr in d:
                for j in range(self.bits_per_feature):
                    tmp_hashes[j, int(md5((ngr + str(self.secret) +str(j)).encode()).hexdigest(), 16) % (self.filter_size)] = True
            enc_data[i] = reduce_func.reduce(tmp_hashes, axis=0)


        if self.diffusion:
            eld = np.zeros((enc_data.shape[0], self.eld_length), dtype=bool)

            # for i in range(enc_data.shape[0]):
            #    for j in range(self.eld_length):
            #        val = enc_data[i, self.indices[j][0]]
            #        for k in self.indices[j][1:]:
            #            val ^= enc_data[i,k]
            #        eld[i,j] = val
            for i, cur_inds in enumerate(self.indices):
                eld[:, i] = np.logical_xor.reduce(enc_data[:, cur_inds], axis=1)

            #return enc_data, eld
            enc_data = eld

        return enc_data

    def encode_and_compare(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str],
                           metric: str, sim: bool = True, store_encs: bool = False) -> np.ndarray:
        """
        Encodes the given data using bloom filter encoding (CLKHash), then computes and returns the pairwise
        similarities/distances of the bloom filters as a list of tuples.
        :param data: Data to encode. A list of lists: Inner list represents records with integers or strings as values.
        :param uids: The uids of the records in the same order as the records in data
        :param metric: Similarity/Distance metric to compute. Any of the ones supported by scipy's pdist.
        :param sim: Choose if similarities (True) or distances (False) should be returned.
        :return: The similarities/distances as a list of tuples: [(i,j,val),...], where i and j are the indices of
        the records in data and val is the computed similarity/distance.
        """
        available_metrics = ["dice", "jaccard"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        # print("DEB: Encoding")
        if self.diffusion:
            _ , enc = self.encode(data)
        else:
            enc, _ = self.encode(data)

        if store_encs:
            cache = dict(zip(uids, enc))
            with open("./data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

        uids = np.array(uids)
        # Compute all possible unique combinations of indices and split them to as many sub-lists as workers
        ind_combinations = np.array_split(numpy_pairwise_combinations(np.arange(enc.shape[0])), self.workers)

        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(
            delayed(calc_metrics)(uids, enc, metric, sim, inds) for inds in ind_combinations)

        return np.vstack(output_generator)

    def get_encoding_dict(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str]):

        # print("DEB: Encoding")
        data = [["".join(d)] for d in data]
        enc = self.encode(data)

        return dict(zip(uids, enc))