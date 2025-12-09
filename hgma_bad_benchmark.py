# Benchmarks the performance of the H-GMA attack on BAD encodings

import random
import time
import os
import math

import numpy as np
import pecanpy
from tqdm import tqdm
from gensim.models import Word2Vec


from bad_utils import *
from encoders.bf_encoder import BFEncoder
from encoders.bv_encoder import BitVecEncoder
from aligners.wasserstein_procrustes import WassersteinAligner
from matchers.bipartite import MinWeightMatcher


# Specify parameters for which the attack should be run

datasets = ["./data/fakename_eng_intra.tsv"]
overlaps = [o/100 for o in range(10,110,10)]

# BAD Encoding Parameters
# Number of Hash Functions to populate the intermediate bloom filters (k)
bf_bits = list(range(5,46,5))
# Diffusion parameter (t)
ts = list(range(2,11,2))

# Specify fixed parameters
CONFIG_DICT = {
        "Verbose": True, # If true, prints status messages on screen
        "BenchMode": True, # If true, saves results to OutFile
        "OutFile": "./breaking_bad_more_datasets.tsv", # File to store the benchmark results in
        "MaxSize": 1000, # Maximum Dataset size to limit runtime. Data is subsampled if it is too large. Set to math.inf to disable.
        # For BAD encoding
        "Secret": "SuperSecretString1337", # Encoding secret/Salt for hash functions
        "N": 2, # Length of q-grams
        "BFLength": 1024, # Length of intermediate Bloom Filters
        "Diffuse": True,  # If true, enables diffusion in BF Encoding (turns BFs into BADs)
        "BADLength": 1024, # Length of the BAD Encoding. Usually equal to length of intermediate BF
    }


bv_encoder = BitVecEncoder(2)


# Iterate over all Datasets
for cur_dataset in datasets:
    CONFIG_DICT["Dataset"] = cur_dataset

    # Read and normalize (prune whitespace, to lowercase) datasets            
    data, uids = read_tsv(CONFIG_DICT["Dataset"])
    data = [["".join(d).replace(" ", "").lower()] for d in data]

    # If dataset is too large, subsample
    if len(data) > CONFIG_DICT["MaxSize"]:
        if CONFIG_DICT["Verbose"]:
            print("Loaded dataset has %i records, subsampling to %i." % (len(data), CONFIG_DICT["MaxSize"]))
        sel = np.random.choice(np.arange(CONFIG_DICT["MaxSize"]),CONFIG_DICT["MaxSize"], replace=False)
        data = [data[i] for i in sel]
        uids = [uids[i] for i in sel]
    
    # Copy original data so that it isn't affected by the random shuffeling
    orig_data = data.copy()
    orig_uids = uids.copy()
    plain_uids = uids.copy()
    
    
    if CONFIG_DICT["Verbose"]:
        print("Finding combinations on plaintext data")
    
    start_comb_plain = time.time()

    # Represent plaintext data as bitvectors to allow re-using the H-GMA code. Each index represents a possible q-gram and is set to 1 if
    # this q-gram is present in the plaintext
    data_bv = bv_encoder.encode(data).astype(bool)
    data_bv = np.pad(data_bv, ((0,0), (0, int(np.ceil(data_bv.shape[1]/64)*64 - data_bv.shape[1]))), 'constant', constant_values=(0))
    
    # Search for combinations of records s.t. their symmetric difference (XOR of bitvectors) is at most 5.
    # For efficiency, this is only done once per plaintext dataset.
    plain_combs = run_hgma_wrapper(pack_rows(data_bv), 2, nb.get_num_threads())
    # Extract the 2-, 3-, and 4-tuples that fulfill the criteria
    tup_2 = plain_combs[np.logical_and(plain_combs[:,2] < 0, plain_combs[:,3] < 0)]
    tup_3 = plain_combs[np.logical_and(plain_combs[:,2] > 0, plain_combs[:,3] < 0)]
    tup_4 = plain_combs[np.logical_and(plain_combs[:,2] > 0, plain_combs[:,3] > 0)]

    # Construct the edgelists representing the relationship graph
    # For 2-tuples, there is an edge between both elements: (a) - (b)
    tup_2_el = tup_2[:,[0,1]]

    # For 3-tuples, 3 edges are necessary:
    # (a) - (b)
    # (a) - (c)
    # (b) - (c)
    tup_3_el = np.vstack([tup_3[:,[0,1]], tup_3[:,[0,2]],
                            tup_3[:,[1,2]]]).astype(np.uint64)
    
    # For 3-tuples, 6 edges are necessary:
    # (a) - (b)
    # (a) - (c)
    # (a) - (d)
    # (b) - (c)
    # (b) - (d)
    # (c) - (d)
    tup_4_el = np.vstack([tup_4[:,[0,1]], tup_4[:,[0,2]], tup_4[:,[0,3]],
                            tup_4[:,[1,2]], tup_4[:,[1,3]], tup_4[:,[2,3]]
                            ]).astype(np.uint64)
    
    # Stack the edgelists and deduplicate, as a single edge might form part of multiple tuples.
    # Save edgelist as tsv.
    plain_edgelist = np.vstack([tup_2_el, tup_3_el, tup_4_el])
    plain_edgelist = np.unique(plain_edgelist, axis=0)
    np.savetxt("data/edgelists/plain_comb_el.edg", plain_edgelist, delimiter="\t", fmt=["%i", "%i"])
    elapsed_comb_plain = time.time() - start_comb_plain
    
    plain_comb_set = set()
    
    # Calculate Identifiers for the found egdes based on the UIDs of the nodes they connect. Will be later used to
    # evaluate the Attack performance by checking if the same edges have been found on the encoded data.
    for row in plain_combs:
        gt_row = np.array([plain_uids[int(i)] for i in row[:-1] if int(i) != -1])
        ident = "-".join(list(np.sort(gt_row.astype(str))))
        plain_comb_set = plain_comb_set.union(set([ident]))
    
    
    if CONFIG_DICT["Verbose"]:
        print("Done.")
    
    start_emb_plain = time.time()
    
    
    # Embed plaintext relationship graph
    plain_comb_graph = pecanpy.pecanpy.SparseOTF(p=0.05, q=0.95, workers=-1, verbose=CONFIG_DICT["Verbose"], random_state=1337,
                                    extend=True)
    plain_comb_graph.read_edg("data/edgelists/plain_comb_el.edg", weighted=False, directed=False)
    plain_comb_walks = plain_comb_graph.simulate_walks(num_walks=100, walk_length=100)
    
    loss_logger = LossLogger()
    cb = [loss_logger]
    plain_model = Word2Vec(
                plain_comb_walks, vector_size=128, window=5, min_count=1, sg=0,
                workers=20, epochs=20, seed=1337,compute_loss=True, callbacks=cb)
    
    
    # Ordering refers to row indices in original data
    plain_ordering = [k for k in plain_model.wv.key_to_index]
    
    plain_embeddings = [plain_model.wv.get_vector(k) for k in plain_ordering]
    plain_embeddings = np.stack(plain_embeddings, axis=0)
    elapsed_emb_plain = time.time() - start_emb_plain
    
    # Encoding of the data and attack.
    # Has to be run for each combination of parameters and dataset.

    for cur_k in tqdm(bf_bits):
        CONFIG_DICT["Bits"] = cur_k
        for cur_t in ts:
            CONFIG_DICT["T"] = cur_t
            for cur_o in overlaps:
                # Copy the original data that will be encoded and shuffled in the current run.
                uids = orig_uids.copy()
                data = orig_data.copy()
                CONFIG_DICT["Overlap"]= cur_o
                start_total = time.time()
                
                # Encode the data usung BAD Encoder
                encoder = BFEncoder(CONFIG_DICT["Secret"], CONFIG_DICT["BFLength"], CONFIG_DICT["Bits"], CONFIG_DICT["N"], CONFIG_DICT["Diffuse"],
                            CONFIG_DICT["BADLength"], CONFIG_DICT["T"])
                
                init_n = len(data)
                start_encoding = time.time()
                data_enc = encoder.encode(data)
                
                # Randomly selects and shuffles elements from the ecodings according to the specified overlap. If overlap is 1, this
                # is equal to a random shuffle.
                # Shuffling is important to avoid unintentional information leakage to the attacker.
                selected_enc = random.sample(range(data_enc.shape[0]), int(np.round(CONFIG_DICT["Overlap"] * data_enc.shape[0])))
                data_enc = data_enc[selected_enc]
                uids = [uids[i] for i in selected_enc]
                
                start_comb_enc = time.time()
                elapsed_encoding =  start_comb_enc - start_encoding
                
                if CONFIG_DICT["Verbose"]:
                    print("Finding Combinations on encoded data")

                # Find n encoding-tuples with lowest symmetric difference. Set n to 1.3*number of plaintext tuples, to ensure that
                # we find enough, then, iteratively decrease the threshold until we have at most 110% of plaintext tuples.
                enc_combs = run_hgma_top_n_wrapper(pack_rows(data_enc), round(plain_combs.shape[0]*1.3), nb.get_num_threads())
                tres_temp = max(enc_combs[:,-1])
                # Reduce the threshold of this run until the number of found encoding pairs is at most 110% of found plaintext pairs.
                while enc_combs.shape[0] > (plain_combs.shape[0]*1.1) and tres_temp > 1:
                    enc_combs = enc_combs[enc_combs[:,-1] < tres_temp]
                    tres_temp -= 1

                # Build and save edgelsit representing relationship graph on encodings 
                tup_2 = enc_combs[np.logical_and(enc_combs[:,2] < 0, enc_combs[:,3] < 0)]
                tup_3 = enc_combs[np.logical_and(enc_combs[:,2] > 0, enc_combs[:,3] < 0)]
                tup_4 = enc_combs[np.logical_and(enc_combs[:,2] > 0, enc_combs[:,3] > 0)]
                
                tup_2_el = tup_2[:,[0,1]]
                tup_3_el = np.vstack([tup_3[:,[0,1]], tup_3[:,[0,2]],
                                        tup_3[:,[1,2]]]).astype(np.uint64)
                
                tup_4_el = np.vstack([tup_4[:,[0,1]], tup_4[:,[0,2]], tup_4[:,[0,3]],
                                        tup_4[:,[1,2]], tup_4[:,[1,3]], tup_4[:,[2,3]]
                                        ]).astype(np.uint64)
                enc_edgelist = np.vstack([tup_2_el, tup_3_el, tup_4_el])
                
                enc_edgelist = np.unique(enc_edgelist, axis=0)
    
                if enc_edgelist.shape[0] == 0:
                    print("Skipping due to no found combs: %s, %s, %s" % (cur_k, cur_t, oo))
                    continue
                np.savetxt("data/edgelists/enc_comb_el.edg", enc_edgelist, delimiter="\t", fmt=["%i", "%i"])
                
                elapsed_comb_enc = time.time() - start_comb_enc
                
                start_ground_truth = time.time()
                
                # Create identifiers for the found egdes based on the UIDs of the nodes they connect.
                enc_comb_set = set()
                for row in enc_combs:
                    gt_row = np.array([uids[int(i)] for i in row[:-1] if int(i) != -1])
                    ident = "-".join(list(np.sort(gt_row.astype(str))))
                    enc_comb_set = enc_comb_set.union(set([ident]))
                
                # Check how many edges the plaintext- and encoding-graphs have in common (For evaluation only)
                shared = len(enc_comb_set.intersection(plain_comb_set))
                if CONFIG_DICT["Verbose"]:
                    print("--Encoded Data--\nCombs found: %i\nUnique: %i" % (len(enc_combs), len(enc_comb_set)))
                    print("--Plain Data--\nCombs found: %i\nUnique: %i\n\n" % (len(plain_combs), len(plain_comb_set)))
                    print("Overlap: %i (%0.3f%%)" % (shared, shared/max(len(plain_comb_set), len(enc_comb_set))*100))
                
                start_emb_enc = time.time()
                elapsed_ground_truth = start_emb_enc - start_ground_truth
                
                # Calculate Node2Vec-embeddings of the relationship graph of the encoded data
                enc_comb_graph = pecanpy.pecanpy.SparseOTF(p=0.05, q=0.95, workers=-1, verbose=CONFIG_DICT["Verbose"], random_state=1337,
                                                extend=True)
                enc_comb_graph.read_edg("data/edgelists/enc_comb_el.edg", weighted=False, directed=False)
                enc_comb_walks = enc_comb_graph.simulate_walks(num_walks=100, walk_length=100)
                
                loss_logger = LossLogger()
                cb = [loss_logger]
                enc_model = Word2Vec(
                            enc_comb_walks, vector_size=128, window=5, min_count=1, sg=0,
                            workers=20, epochs=20, seed=1337,compute_loss=True, callbacks=cb)
                enc_ordering = [k for k in enc_model.wv.key_to_index]
                enc_embeddings = [enc_model.wv.get_vector(k) for k in enc_ordering]
                enc_embeddings = np.stack(enc_embeddings, axis=0)
                
                start_align = time.time()
                elapsed_emb_enc = start_align - start_emb_enc
                elapsed_emb_total = elapsed_emb_enc + elapsed_emb_plain
    
                # Align the embeddings
                aligner = WassersteinAligner(reg_init=1, reg_ws=0.1,
                                            batchsize=min(plain_embeddings.shape[0], enc_embeddings.shape[0]), lr=200, n_iter_init=5,
                                            n_iter_ws=50, n_epoch=500,
                                            lr_decay=1, apply_sqrt=True, early_stopping=10,
                                            verbose=True)
                
    
                transformation_matrix = aligner.align(plain_embeddings, enc_embeddings)
                enc_embeddings = np.dot(enc_embeddings, transformation_matrix.T)
                
                start_match = time.time()
                elapsed_align = start_match - start_align
                
                enc_ord_uids = [uids[int(i)] for i in enc_ordering]
                plain_ord_uids = [plain_uids[int(i)] for i in plain_ordering]
                
                # Calculate bipartite Minimum Weight Matching
                matcher = MinWeightMatcher("euclidean")
                m = matcher.match(enc_embeddings, enc_ord_uids, plain_embeddings, plain_ord_uids)
                
                elapsed_match = time.time() - start_match
                elapsed_relevant = ((time.time() - start_comb_enc) - elapsed_ground_truth) + elapsed_comb_plain + elapsed_emb_plain
                
                # Count correctly re-identified records.
                correct = 0
                wrong = 0
                
                for s, l in m.items():
                    s = int(s[2:])
                    l = int(l[2:])
                    if s != l:
                        wrong += 1
                    else:
                        correct += 1
                
                if CONFIG_DICT["Verbose"]:
                    print("Done")
                    print("Correctly re-identified %i of %i records (%0.3f%%).\nWrong: %i" % (correct, data_enc.shape[0], correct/data_enc.shape[0]*100, wrong))
                
                elapsed_total = time.time() - start_total
                if CONFIG_DICT["BenchMode"]:
                    keys = ["timestamp"]
                    vals = [time.time()]
                    for key, val in CONFIG_DICT.items():
                        keys.append(key)
                        vals.append(val)
                
                    keys += ["success_rate", "correct", "wrong", "n_encoded","n_plain", "elapsed_encoding", "elapsed_comb_enc", "elapsed_comb_plain", 
                            "elapsed_emb_enc", "elapsed_emb_plain", "elapsed_emb_total", "elapsed_align", "elapsed_match", "elapsed_relevant", "elapsed_ground_truth",
                            "elapsed_total", "combs_plain", "combs_enc", "combs_shared"]
                
                    vals += [correct/len(uids), correct, wrong, data_enc.shape[0], init_n, elapsed_encoding, elapsed_comb_enc, elapsed_comb_plain, 
                            elapsed_emb_enc, elapsed_emb_plain, elapsed_emb_total, elapsed_align, elapsed_match, elapsed_relevant, elapsed_ground_truth,
                            elapsed_total, len(plain_comb_set), len(enc_comb_set), shared]
                    if not os.path.isfile(CONFIG_DICT["OutFile"]):
                            save_tsv([keys], CONFIG_DICT["OutFile"])
                    
                    save_tsv([vals], CONFIG_DICT["OutFile"], mode="a")
