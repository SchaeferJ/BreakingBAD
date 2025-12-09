import csv
from gensim.models.callbacks import CallbackAny2Vec
import numba as nb
import numpy as np


def read_tsv(path: str, header: bool = True, as_dict: bool = False, delim: str = "\t"):
    data = {} if as_dict else []
    uid = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        if header:
            next(reader)
        for row in reader:
            if as_dict:
                assert len(row) == 3, "Dict mode only supports rows with two values + uid"
                data[row[0]] = row[1]
            else:
                data.append(row[:-1])
                uid.append(row[-1])
    return data, uid

def save_tsv(data, path: str, delim: str = "\t", mode="w"):
    with open(path, mode, newline="") as f:
        csvwriter = csv.writer(f, delimiter=delim)
        csvwriter.writerows(data)        


def pack_rows(bools):
    # Packs bitvector rows into arrays of unsigned 64-bit integers for
    # use with the SWAR popcount function.
    packed = np.packbits(bools, axis=1, bitorder='little')   # uint8
    return packed.view(np.uint64).copy()                     # uint64


@nb.njit((nb.uint64)(nb.uint64),cache=True)
def popcnt64(x):
    # Efficient Hamming Weight calculation
    # Credit: https://www.playingwithpointers.com/blog/swar.html
    x -= (x >> 1) & 0x5555555555555555
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    return ((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56) & 0xFF

# ==============================================================================
# Find Tuples based on Symdiff thresholds (Plaintext Case)
# ==============================================================================

@nb.njit(parallel=True, fastmath=True, cache=True)
def run_hgma(enc, threshold, n_threads):
    # Inputs packed bitvectors and finds tuples of length 2 to 4, for which the hamming weight of their XOR (symmetric differecne) is below or equal to the threshold.
    # Returns a list of lists (one list per thread) containing the found tuples in the form (row_index_1, row_index_2, row_index_3, row_index_4, hamming_weight).
    # For tuples smaller than 4, unused row indices are set to -1.

    n, nWords = enc.shape
    CHUNK_SIZE = 4096  # Size of temporary buffer per thread. Not sure what a good value is, but 4096 works fine.
    
    # Define the data type for the results (i, j, k, l, weight)
    # We use a 2D array of int16 for the chunks
    result_chunk_type = nb.int16[:, ::1]

    # Set up a tread-safe result store
    thread_storage = [nb.typed.List.empty_list(result_chunk_type) for _ in range(n_threads)]
    
    for thread_id in nb.prange(n_threads):

        # Allocate vectors for intermediate results to avoid redundant computations:
        # For all 2-tuples, store their XOR and search for 3-tuples by XORing the
        # remaining records with the stored result. Analogous logic for 4-tuples.
        xor_sum_j = np.empty(nWords, np.uint64)
        xor_sum_k = np.empty(nWords, np.uint64)
        
        # Allocate a local buffer within the thread to minimize overhead from list operations
        local_buffer = np.empty((CHUNK_SIZE, 5), dtype=np.int16)
        buffer_idx = 0
        
        # Define iterator based on thread id to distribute work over the threads.
        for i in range(thread_id, n - 1, n_threads):
            # Compare to remaining records
            # Starting at i+1 because XOR is symmetric
            for j in range(i + 1, n):
                # Tuples of size 2
                hamm_wt = 0
                for w in range(nWords):
                    val = enc[i, w] ^ enc[j, w]
                    # Store XOR of 2-tuple
                    xor_sum_j[w] = val
                    hamm_wt += popcnt64(val)
                
                # If 2-tuple Hamming Weight is below threshold, add to results
                if hamm_wt <= threshold:
                    local_buffer[buffer_idx, 0] = i
                    local_buffer[buffer_idx, 1] = j
                    local_buffer[buffer_idx, 2] = -1 # Unused, because 2-Tupel
                    local_buffer[buffer_idx, 3] = -1 # Unused, because 2-Tupel
                    local_buffer[buffer_idx, 4] = hamm_wt
                    buffer_idx += 1
                    
                    # Flush buffer if full
                    if buffer_idx == CHUNK_SIZE:
                        thread_storage[thread_id].append(local_buffer.copy())
                        buffer_idx = 0

                # Tuples of size 3
                if j < (n - 1):
                    for k in range(j + 1, n):
                        hamm_wt = 0
                        for w in range(nWords):
                            val = xor_sum_j[w] ^ enc[k, w]
                            # Store XOR of 3-tuple
                            xor_sum_k[w] = val
                            hamm_wt += popcnt64(val)

                        # If 3-tuple Hamming Weight is below threshold, add to results    
                        if hamm_wt <= threshold:
                            local_buffer[buffer_idx, 0] = i
                            local_buffer[buffer_idx, 1] = j
                            local_buffer[buffer_idx, 2] = k
                            local_buffer[buffer_idx, 3] = -1 # Unused, because 3-Tupel
                            local_buffer[buffer_idx, 4] = hamm_wt
                            buffer_idx += 1
                            if buffer_idx == CHUNK_SIZE:
                                thread_storage[thread_id].append(local_buffer.copy())
                                buffer_idx = 0

                        # Tuples of size 4
                        if k < (n - 2):
                            for l in range(k + 1, n):
                                hamm_wt = 0
                                for w in range(nWords):
                                    # No need to store XOR, unless extended to 5-tuples
                                    hamm_wt += popcnt64(xor_sum_k[w] ^ enc[l, w])
                                
                                if hamm_wt <= threshold:
                                    local_buffer[buffer_idx, 0] = i
                                    local_buffer[buffer_idx, 1] = j
                                    local_buffer[buffer_idx, 2] = k
                                    local_buffer[buffer_idx, 3] = l
                                    local_buffer[buffer_idx, 4] = hamm_wt
                                    buffer_idx += 1
                                    if buffer_idx == CHUNK_SIZE:
                                        thread_storage[thread_id].append(local_buffer.copy())
                                        buffer_idx = 0

        # Final flush of the buffers
        if buffer_idx > 0:
            thread_storage[thread_id].append(local_buffer[:buffer_idx].copy())

    return thread_storage

def run_hgma_wrapper(enc, threshold, n_threads):
    # Wrapper that calls the thread-safe Numba function and merges the chunked results 
    # into a single Numpy array.
    
    # Returns a list of lists of arrays
    nested_results = run_hgma(enc, threshold, n_threads)
    
    #Flatten
    total_rows = 0
    all_chunks = []
    
    for thread_list in nested_results:
        for chunk in thread_list:
            total_rows += chunk.shape[0]
            all_chunks.append(chunk)
            
    if total_rows == 0:
        return np.empty((0, 5), dtype=np.int16)
        
    # 3. Concatenate (This is very fast compared to the search)
    final_results = np.vstack(all_chunks)
    
    return final_results


# ==============================================================================
# Find Tuples with lowest symdiff (Encoded Data Case)
# ==============================================================================

# As described in the paper, we want to find encoding-tuples such that their total 
# number does not exceed 110% of the tuples found in the plaintext. Thus, the following
# functions search for the N tuples with the lowest symmetric difference. This avoids the
# problem of specifying a reasonable upper bound for the threshold of what can be considered 
# a possible matching tuple on the encodings. If this threshold would be too low, we would miss
# tuples, if it is too high, we run out of memory and/or the search takes too long.


@nb.njit(cache=True)
def _sift_down(weights, indices, start, end):
    # Standard binary heap sift-down implementation.
    # Maintains a Max-Heap (largest weight at index 0).
    root = start
    while True:
        child = 2 * root + 1
        if child > end:
            break
        
        # Find the larger of the two children
        if child + 1 <= end and weights[child] < weights[child + 1]:
            child += 1
        
        # If the root is smaller than the largest child, swap
        if weights[root] < weights[child]:
            # Swap weights
            w_tmp = weights[root]
            weights[root] = weights[child]
            weights[child] = w_tmp
            
            # Swap indices (manually unrolled for performance)
            i0 = indices[root, 0]; i1 = indices[root, 1]; 
            i2 = indices[root, 2]; i3 = indices[root, 3]
            
            indices[root, 0] = indices[child, 0]
            indices[root, 1] = indices[child, 1]
            indices[root, 2] = indices[child, 2]
            indices[root, 3] = indices[child, 3]
            
            indices[child, 0] = i0; indices[child, 1] = i1; 
            indices[child, 2] = i2; indices[child, 3] = i3
            
            root = child
        else:
            return

@nb.njit(parallel=True, fastmath=True, cache=True)
def run_hgma_top_n(enc, top_n, n_threads):
    # Inputs packed bitvectors and within each thread (!) finds the top_n tuples of length 2 to 4, for which the hamming weight of their XOR (symmetric differecne) is minimal.
    # Returns two list of arrays (one array per thread) containing the Hamming weights of the found tuples and their row indices. For tuples smaller than 4, unused row indices are set to -1.
    # Arrays are sorted by Hamming weight. 

    n, nWords = enc.shape
    
    # Explicit typecasting to avoid Numba errors
    t_dim = int(n_threads)
    k_dim = int(top_n)
    
    # Initialize array with the Hamming Weights of the "best"
    # found tuples. This is initialized with a number that is
    # greater than the maximum possible weight.
    thread_weights = np.empty((t_dim, k_dim), dtype=np.int32)
    thread_weights[:] = 999999  # Fill with "infinity"
    
    # Row indices of the found tuple members, initialize with -1
    thread_indices = np.empty((t_dim, k_dim, 4), dtype=np.int16)
    thread_indices[:] = -1

    for thread_id in nb.prange(t_dim):
        
        # Assign to local variables because I'm lazy and want to type as little as possible
        my_weights = thread_weights[thread_id]
        my_indices = thread_indices[thread_id]
        
        # Current threshold is the WORST (largest) weight in our top-n list.
        current_threshold = my_weights[0] 
        
        # Allocate vectors for storing intermediate XORs
        xor_sum_j = np.empty(nWords, np.uint64)
        xor_sum_k = np.empty(nWords, np.uint64)

        # From here on, logic is equivalent to the search by threshold
        for i in range(thread_id, n - 1, t_dim):
            for j in range(i + 1, n):
                
                # Tuples of size 2
                hamm_wt = 0
                for w in range(nWords):
                    val = enc[i, w] ^ enc[j, w]
                    xor_sum_j[w] = val
                    hamm_wt += popcnt64(val)
                
                # Check against dynamic threshold
                if hamm_wt < current_threshold:
                    my_weights[0] = hamm_wt
                    my_indices[0, 0] = i
                    my_indices[0, 1] = j
                    my_indices[0, 2] = -1
                    my_indices[0, 3] = -1
                    _sift_down(my_weights, my_indices, 0, k_dim - 1)
                    # Update threshold to new worst tuple
                    current_threshold = my_weights[0]

                if j < (n - 1):
                    for k in range(j + 1, n):
                        # Tuples of size 3
                        hamm_wt = 0
                        for w in range(nWords):
                            val = xor_sum_j[w] ^ enc[k, w]
                            xor_sum_k[w] = val
                            hamm_wt += popcnt64(val)
                        
                        if hamm_wt < current_threshold:
                            my_weights[0] = hamm_wt
                            my_indices[0, 0] = i
                            my_indices[0, 1] = j
                            my_indices[0, 2] = k
                            my_indices[0, 3] = -1
                            _sift_down(my_weights, my_indices, 0, k_dim - 1)
                            current_threshold = my_weights[0]

                        if k < (n - 2):
                            for l in range(k + 1, n):
                                # Tuples of size 4
                                hamm_wt = 0
                                for w in range(nWords):
                                    hamm_wt += popcnt64(xor_sum_k[w] ^ enc[l, w])
                                
                                if hamm_wt < current_threshold:
                                    my_weights[0] = hamm_wt
                                    my_indices[0, 0] = i
                                    my_indices[0, 1] = j
                                    my_indices[0, 2] = k
                                    my_indices[0, 3] = l
                                    _sift_down(my_weights, my_indices, 0, k_dim - 1)
                                    current_threshold = my_weights[0]

    return thread_weights, thread_indices

def run_hgma_top_n_wrapper(enc, top_n, n_threads):

    # This returns separated heaps for each thread
    t_weights, t_indices = run_hgma_top_n(enc, top_n, n_threads)
    
    # Flatten the results, as we currently have (n_threads * top_n) candidates.
    flat_weights = t_weights.ravel()
    flat_indices = t_indices.reshape(-1, 4)
    
    # Filter for invalid entries (the initialization 999999s):
    # If the search space was smaller than top_n, we might have leftovers.
    valid_mask = flat_weights < 999999
    flat_weights = flat_weights[valid_mask]
    flat_indices = flat_indices[valid_mask]
    
    # Use argsort to get ascending order and extract tuples with smallest hamming weight.
    sorted_order = np.argsort(flat_weights)
    
    # Take only the top_n from the combined pool
    final_k = min(len(sorted_order), top_n)
    best_indices = sorted_order[:final_k]
    
    # Construct final result (indices + weight column)
    # Result format: (i, j, k, l, weight)
    result_indices = flat_indices[best_indices]
    result_weights = flat_weights[best_indices]
    
    return np.hstack((result_indices, result_weights[:, None]))



class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = [0]

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss-self.losses[self.epoch-1]}')
        self.epoch += 1