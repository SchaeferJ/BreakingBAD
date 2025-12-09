import numba as nb
import os

@nb.jit((nb.int64[:])(nb.float64[:, :], nb.int64, nb.int64), parallel=True)
def parallel_argmax(A, axis, workers):
    m, n = A.shape    
    if axis == 1:
        # Rowwise
        n_chunks = min(m, workers)
        chunk_size = m // n_chunks
        maxima = A[:,0].copy()
        indices = np.zeros(m, dtype=np.int64)
        for i_chunk in nb.prange(n_chunks):
            chunk_start = i_chunk * chunk_size
            chunk_end = min(chunk_start + chunk_size, m)
            for i in range(chunk_start, chunk_end):
                for j in range(1, n):
                    current_value = A[i, j]
                    if current_value > maxima[i]:
                        indices[i] = j
                        maxima[i] = current_value
    else:
        n_chunks = min(n, workers)
        chunk_size = n // n_chunks
        maxima = A[0, :].copy()
        indices = np.zeros(n, dtype=np.int64)
        for i_chunk in nb.prange(n_chunks):
            chunk_start = i_chunk * chunk_size
            chunk_end = min(chunk_start + chunk_size, n)
            for j in range(1, m):
                for i in range(chunk_start, chunk_end):
                    current_value = A[j, i]
                    if current_value > maxima[i]:
                        indices[i] = j
                        maxima[i] = current_value

    return indices


class SymmetricMatcher():
    def __init__(self, metric: str = "cosine", workers: int = -1):
        available_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                             "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis",
                             "matching", "l1", "l2", "manhattan",
                             "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                             "sqeuclidean", "yule"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        self.metric = metric
        self.workers = os.cpu_count() if workers==-1 else workers

    def match(self, left_data, left_uids, right_data, right_uids):
        dice_sims = 1 - pairwise_distances(left_data, right_data, metric=self.metric, n_jobs=self.workers)
        left_argmax = parallel_argmax(dice_sims, axis=0, workers=self.workers)
        right_argmax = parallel_argmax(dice_sims, axis=1, workers=self.workers)
        matches = np.where(np.arange(left_argmax.shape[0]) == right_argmax[left_argmax])

        matching = {}
        for l, r in zip(left_argmax[matches], right_argmax[left_argmax[matches]]):
            matching["L_"+str(left_uids[l])] = "R_"+str(right_uids[r])
        return matching