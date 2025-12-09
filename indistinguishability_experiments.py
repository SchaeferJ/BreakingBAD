import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import string
import networkx as nx
from gensim.models import Word2Vec
from node2vec import Node2Vec

from encoders.plain_bf_encoder import BFEncoder
from encoders.saul_encoder import SaulEncoder
from encoders.bv_encoder import BitVecEncoder

from bad_utils import *
#from indist_utils import *

from aligners.wasserstein_procrustes import WassersteinAligner
from matchers.bipartite import GaleShapleyMatcher, SymmetricMatcher, MinWeightMatcher
from sklearn.metrics.pairwise import pairwise_distances

from itertools import product

def get_disjoint_bigrams(n):
    """Gets a set of n unique, real character bigrams from the master pool."""
    global bigram_counter
    start, end = bigram_counter, bigram_counter + n
    if end > POOL_SIZE:
        raise ValueError(f"Not enough unique bigrams in the pool.")
    bigram_counter = end
    return set(MASTER_BIGRAM_POOL[start:end])

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 1.0

def symmetric_difference_3(set1, set2, set3):
    return set1.symmetric_difference(set2).symmetric_difference(set3)

def generate_fixed_length_datasets(
    num_entries, k, dissimilarity_threshold=0.5, num_base_entries=5, p_structure=0.5
):
    if num_base_entries < 1: raise ValueError("num_base_entries must be at least 1.")
    
    dataset_A, dataset_B = [], []
    
    # Initialize with a base set
    for _ in range(num_base_entries):
        entry = get_disjoint_bigrams(2 * k)
        dataset_A.append(entry)
        dataset_B.append(entry.copy())

    # Iteratively grow the datasets using a while loop
    while len(dataset_B) < num_entries:
        # Stop if we are about to run out of entries to add
        if len(dataset_B) >= num_entries - 1 and random.random() < p_structure:
            is_structured_step = False # Force random if we can't add two
        else:
            is_structured_step = random.random() < p_structure

        # --- A. Create a structured triplet (adds 2 entries) ---
        if is_structured_step:
            parent_idx = random.randrange(len(dataset_B))
            parent_B = dataset_B[parent_idx]
            parent_A = dataset_A[parent_idx]

            # Deconstruct parent B and create new material
            s1 = set(random.sample(list(parent_B), k))
            s2 = parent_B - s1
            s3 = get_disjoint_bigrams(k)
            
            # Construct two new entries for B
            e_new1 = s2.union(s3)
            e_new2 = s1.union(s3)

            # Create two "mimic" entries for A
            a_new1 = set(random.sample(list(parent_A), k)).union(get_disjoint_bigrams(k))
            a_new2 = set(random.sample(list(parent_A), k)).union(get_disjoint_bigrams(k))
            
            # Check dissimilarity constraints
            # Check new B entries against non-relatives
            valid_b1 = all(jaccard_similarity(e_new1, dataset_B[i]) < dissimilarity_threshold for i in range(len(dataset_B)) if i != parent_idx)
            valid_b2 = all(jaccard_similarity(e_new2, dataset_B[i]) < dissimilarity_threshold for i in range(len(dataset_B)) if i != parent_idx)
            # Check new A entries against all existing
            valid_a1 = all(jaccard_similarity(a_new1, entry) < dissimilarity_threshold for entry in dataset_A)
            valid_a2 = all(jaccard_similarity(a_new2, entry) < dissimilarity_threshold for entry in dataset_A)

            if valid_b1 and valid_b2 and valid_a1 and valid_a2:
                dataset_B.extend([e_new1, e_new2])
                dataset_A.extend([a_new1, a_new2])
                continue # Go to next iteration of while loop

        # create a single random entry if structured step fails or isn't chosen
        dataset_B.append(get_disjoint_bigrams(2 * k))
        dataset_A.append(get_disjoint_bigrams(2 * k))

    return dataset_A[:num_entries], dataset_B[:num_entries]

def find_triplets(dataset):
    # Same as before
    triplets = []
    for i, j, k in itertools.combinations(range(len(dataset)), 3):
        if len(symmetric_difference_3(dataset[i], dataset[j], dataset[k])) == 0:
            triplets.append(tuple(sorted((i, j, k))))
    return list(set(triplets))

for use_saul in [False, True]:

    if not use_saul:
        print("-" * 50)
        print("Experiment for BADs")
        print("-" * 50)
    else:    
        print("-" * 50)
        print("Experiment for SAUL")
        print("-" * 50)


    charset = [chr(i) for i in range(400,580)]
    MASTER_BIGRAM_POOL = [''.join(p) for p in itertools.product(charset, repeat=2)]
    random.shuffle(MASTER_BIGRAM_POOL)
    POOL_SIZE = len(MASTER_BIGRAM_POOL)
    bigram_counter = 0

    # Parameters
    NUM_TOTAL_ENTRIES = 1000
    K_PARAM = 15 # Each entry has 2 * 50 = 100 bigrams
    DISSIMILARITY_THRESHOLD = 0.5
    NUM_BASE_ENTRIES = 1
    P_STRUCTURE = 1 # Higher probability to ensure we see structures

    # Reset counter
    bigram_counter = 0

    print("Generating fixed-length datasets...")
    A, B = generate_fixed_length_datasets(
        num_entries=NUM_TOTAL_ENTRIES,
        k=K_PARAM,
        dissimilarity_threshold=DISSIMILARITY_THRESHOLD,
        num_base_entries=NUM_BASE_ENTRIES,
        p_structure=P_STRUCTURE
    )
    print(f"Successfully generated datasets with {len(A)} entries each.")
    print("-" * 50)

    # --- Requirement: Verify Entry Lengths ---
    print("Verifying entry lengths...")
    lengths_A = [len(entry) for entry in A]
    lengths_B = [len(entry) for entry in B]
    print(f"Lengths in A: Min={min(lengths_A)}, Max={max(lengths_A)}, Mean={np.mean(lengths_A):.2f}")
    print(f"Lengths in B: Min={min(lengths_B)}, Max={max(lengths_B)}, Mean={np.mean(lengths_B):.2f}")
    assert min(lengths_B) == max(lengths_B) == 2 * K_PARAM
    print("Verification PASSED: All entries in B have the same length.")
    print("-" * 50)

    # --- Similarity Distribution Verification ---
    print("Verifying similarity distributions...")
    sims_A = [jaccard_similarity(p[0], p[1]) for p in itertools.combinations(A, 2)]
    sims_B = [jaccard_similarity(p[0], p[1]) for p in itertools.combinations(B, 2)]

    plt.figure(figsize=(12, 6))
    plt.hist(sims_A, bins=40, range=(0, DISSIMILARITY_THRESHOLD + 0.05), alpha=0.7, label='Dataset A', density=True)
    plt.hist(sims_B, bins=40, range=(0, DISSIMILARITY_THRESHOLD + 0.05), alpha=0.7, label='Dataset B', density=True)
    plt.title('Distribution of Pairwise Jaccard Similarities')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.show()

    print("-" * 50)

    rand_data = [list(aa) for aa in A]
    struct_data = [list(bb) for bb in B]

    data = struct_data
    uids = list(range(len(data)))

    # Set up encoders

    oh_encoder = BitVecEncoder(2, charset=charset)

    encoder = BFEncoder(secret = "SomeSuperSecretString1337", filter_size = 1024, bits_per_feature = 10,
                    ngram_size = 2, diffusion=True, eld_length=1024,
                    t=10, xor_reduce=False, workers=-1)

    saul = SaulEncoder("AnotherSaltThatNobodyKnows4242", 2, 1024, 4, workers=-1, charset=charset)

    if use_saul:
        plain_oh = oh_encoder.encode(data, pass_ngrams=True).astype(bool)
        enc = saul.encode(plain_oh, pass_oh=True)
    else:
        enc = encoder.encode(data, pass_ngrams=True)

    rand_oh = oh_encoder.encode(rand_data, pass_ngrams=True).astype(bool)
    struct_oh = oh_encoder.encode(struct_data, pass_ngrams=True).astype(bool)
    pwsim_struct = 1 - pairwise_distances(struct_oh, metric='dice', n_jobs=-1)
    pwsim_rand = 1 - pairwise_distances(rand_oh, metric='dice', n_jobs=-1)

    pwsim_struct = pwsim_struct[np.triu_indices_from(pwsim_struct, k=1)]
    pwsim_rand = pwsim_rand[np.triu_indices_from(pwsim_rand, k=1)]

    print("Pairwise Similarities on Random Data - Min: %0.3f - Max: %0.3f - Median: %0.3f" % (min(pwsim_rand), max(pwsim_rand), np.median(pwsim_rand)))
    print("Pairwise Similarities on Structured Data - Min: %0.3f - Max: %0.3f - Median: %0.3f" % (min(pwsim_struct), max(pwsim_struct), np.median(pwsim_struct)))

    ## Attack
    orig_data = data.copy()
    orig_uids = uids.copy()
    plain_uids = uids.copy()

    data_oh = oh_encoder.encode(data, pass_ngrams=True).astype(bool)
    data_oh = np.pad(data_oh, ((0,0), (0, int(np.ceil(data_oh.shape[1]/64)*64 - data_oh.shape[1]))), 'constant', constant_values=(0))

    plain_combs = run_hgma_wrapper(pack_rows(data_oh), 2, nb.get_num_threads())
    tup_2 = plain_combs[np.logical_and(plain_combs[:,2] < 0, plain_combs[:,3] < 0)]
    tup_3 = plain_combs[np.logical_and(plain_combs[:,2] > 0, plain_combs[:,3] < 0)]
    tup_4 = plain_combs[np.logical_and(plain_combs[:,2] > 0, plain_combs[:,3] > 0)]
    tup_2_el = tup_2[:,[0,1]]
    tup_3_el = np.vstack([tup_3[:,[0,1]], tup_3[:,[0,2]],
                            tup_3[:,[1,2]]]).astype(np.uint64)

    tup_4_el = np.vstack([tup_4[:,[0,1]], tup_4[:,[0,2]], tup_4[:,[0,3]],
                            tup_4[:,[1,2]], tup_4[:,[1,3]], tup_4[:,[2,3]]
                            ]).astype(np.uint64)

    plain_combs = tup_3
    plain_edgelist = tup_3_el
    plain_edgelist = np.unique(plain_edgelist, axis=0)

    plain_comb_set = set()
    for row in plain_combs:
        gt_row = np.array([plain_uids[int(i)] for i in row[:-1] if int(i) != -1])
        ident = "-".join(list(np.sort(gt_row.astype(str))))
        plain_comb_set = plain_comb_set.union(set([ident]))

    uids = orig_uids.copy()
    data = orig_data.copy()
    init_n = len(data)
    data_enc = enc.copy()
    enc_combs = run_hgma_top_n_wrapper(pack_rows(data_enc), round(plain_combs.shape[0]*5), nb.get_num_threads())
    etup_2 = enc_combs[np.logical_and(enc_combs[:,2] < 0, enc_combs[:,3] < 0)]
    etup_3 = enc_combs[np.logical_and(enc_combs[:,2] > 0, enc_combs[:,3] < 0)]
    etup_4 = enc_combs[np.logical_and(enc_combs[:,2] > 0, enc_combs[:,3] > 0)]
    enc_combs = etup_3

    tres_temp = max(enc_combs[:,-1])
    # Reduce the threshold of this run until the number of found encoding pairs is at most 110% of found plaintext pairs.
    while enc_combs.shape[0] > (plain_combs.shape[0]) and tres_temp > 1:
        enc_combs = enc_combs[enc_combs[:,-1] < tres_temp]
        tres_temp -= 1

    etup_2 = enc_combs[np.logical_and(enc_combs[:,2] < 0, enc_combs[:,3] < 0)]
    etup_3 = enc_combs[np.logical_and(enc_combs[:,2] > 0, enc_combs[:,3] < 0)]
    etup_4 = enc_combs[np.logical_and(enc_combs[:,2] > 0, enc_combs[:,3] > 0)]

    etup_2_el = etup_2[:,[0,1]]
    etup_3_el = np.vstack([etup_3[:,[0,1]], etup_3[:,[0,2]],
                            etup_3[:,[1,2]]]).astype(np.uint64)

    etup_4_el = np.vstack([etup_4[:,[0,1]], etup_4[:,[0,2]], etup_4[:,[0,3]],
                            etup_4[:,[1,2]], etup_4[:,[1,3]], etup_4[:,[2,3]]
                            ]).astype(np.uint64)
    enc_edgelist = etup_3_el 

    enc_edgelist = np.unique(enc_edgelist, axis=0)

    G = nx.from_edgelist(plain_edgelist)
    Ge = nx.from_edgelist(enc_edgelist)

    enc_comb_set = set()
    for row in enc_combs:
        gt_row = np.array([uids[int(i)] for i in row[:-1] if int(i) != -1])
        ident = "-".join(list(np.sort(gt_row.astype(str))))
        enc_comb_set = enc_comb_set.union(set([ident]))

    shared = len(enc_comb_set.intersection(plain_comb_set))
    print("--Encoded Data--\nCombs found: %i\nUnique: %i" % (len(enc_combs), len(enc_comb_set)))
    print("--Plain Data--\nCombs found: %i\nUnique: %i\n\n" % (len(plain_combs), len(plain_comb_set)))
    print("Overlap: %i (%0.3f%%)" % (shared, shared/max(len(plain_comb_set), len(enc_comb_set))*100))

    plain_comb_walks = Node2Vec(G, dimensions=128, walk_length=100, num_walks=100, workers=1, seed=1337).walks  # Use temp_folder for big graphs
    enc_comb_walks = Node2Vec(Ge, dimensions=128, walk_length=100, num_walks=100, workers=1, seed=1337).walks  # Use temp_folder for big graphs

    loss_logger = LossLogger()
    cb = [loss_logger]

    plain_model = Word2Vec(
                plain_comb_walks, vector_size=128, window=5, min_count=1, sg=0,
                workers=1, epochs=3, seed=1337,compute_loss=True, callbacks=cb)


    # Ordering refers to row indices in original data
    plain_ordering = [k for k in plain_model.wv.key_to_index]

    plain_embeddings = [plain_model.wv.get_vector(k) for k in plain_ordering]
    plain_embeddings = np.stack(plain_embeddings, axis=0)

    loss_logger = LossLogger()
    cb = [loss_logger]
    enc_model = Word2Vec(
                enc_comb_walks, vector_size=128, window=5, min_count=1, sg=0,
                workers=1, epochs=3, seed=1337,compute_loss=True, callbacks=cb)
    enc_ordering = [k for k in enc_model.wv.key_to_index]
    enc_embeddings = [enc_model.wv.get_vector(k) for k in enc_ordering]
    enc_embeddings = np.stack(enc_embeddings, axis=0)

    enc_embeddings = [enc_model.wv.get_vector(k) for k in enc_ordering]
    enc_embeddings = np.stack(enc_embeddings, axis=0)
    plain_embeddings = [plain_model.wv.get_vector(k) for k in plain_ordering]
    plain_embeddings = np.stack(plain_embeddings, axis=0)

    aligner = WassersteinAligner(reg_init=1, reg_ws=0.1,
                                batchsize=min(plain_embeddings.shape[0], enc_embeddings.shape[0]), lr=200, n_iter_init=5,
                                n_iter_ws=50, n_epoch=500,
                                lr_decay=1, apply_sqrt=True, early_stopping=3,
                                verbose=True)

    transformation_matrix = aligner.align(plain_embeddings, enc_embeddings)
    enc_embeddings = np.dot(enc_embeddings, transformation_matrix.T)
    enc_ord_uids = [uids[int(i)] for i in enc_ordering]
    plain_ord_uids = [plain_uids[int(i)] for i in plain_ordering]

    matcher = MinWeightMatcher("euclidean")
    m = matcher.match(enc_embeddings, enc_ord_uids, plain_embeddings, plain_ord_uids)
    correct = 0
    wrong = 0

    for s, l in m.items():
        s = int(s[2:])
        l = int(l[2:])
        if s != l:
            wrong += 1
        else:
            correct += 1

    print("Done")
    print("Correctly re-identified %i of %i records (%0.3f%%).\nWrong: %i" % (correct, data_enc.shape[0], correct/data_enc.shape[0]*100, wrong))
                