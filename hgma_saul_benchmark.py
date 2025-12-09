# Benchmarks the performance of the H-GMA attack on SAUL encodings
# Refer to attack_benchmark.py for documentation

import random
import time
import numpy as np
import pecanpy
import os
from scipy.special import binom
from collections import Counter, defaultdict
from bad_utils import *
from gensim.models import Word2Vec

from encoders.saul_encoder import SaulEncoder
from encoders.bv_encoder import BitVecEncoder
from aligners.wasserstein_procrustes import WassersteinAligner
from matchers.bipartite import GaleShapleyMatcher, SymmetricMatcher, MinWeightMatcher

from tqdm import tqdm

datasets = ["./data/fakename_eng_intra.tsv"]
overlaps = [1]
num_hashes = list(range(2,11,1))

bv_encoder = BitVecEncoder(2)

CONFIG_DICT = {
        "Dataset": "./data/fakename_eng_intra.tsv",
        "OutFile": "./breaking_saul_benchmark.tsv",
        "Verbose": False,
        "BenchMode": True,
        "Secret": str(random.randint(1,1000000)),
        "N": 2,
        "SAULNumHash": 4,
        # For BF encoding
        "BFLength": 1024,
        "Overlap": 1
    }


            
data, uids = read_tsv(CONFIG_DICT["Dataset"])

orig_data = data.copy()
orig_uids = uids.copy()
plain_uids = uids.copy()



if CONFIG_DICT["Verbose"]:
    print("Finding combinations on plaintext data")

start_comb_plain = time.time()
data_oh = oh_encoder.make_onehot(data).astype(bool)
data_oh = np.pad(data_oh, ((0,0), (0, int(np.ceil(data_oh.shape[1]/64)*64 - data_oh.shape[1]))), 'constant', constant_values=(0))
plain_combs = calc_metrics_new(pack_rows(data_oh), 0.95, 20)

plain_edgelist = np.vstack([plain_combs[:,[0,1]], plain_combs[:,[0,2]], plain_combs[:,[0,3]],
                           plain_combs[:,[1,2]], plain_combs[:,[1,3]], plain_combs[:,[2,3]]
                           ]).astype(np.uint64)
plain_edgelist = np.unique(plain_edgelist, axis=0)
np.savetxt("data/edgelists/plain_comb_el.edg", plain_edgelist, delimiter="\t", fmt=["%i", "%i"])
elapsed_comb_plain = time.time() - start_comb_plain

plain_comb_set = set()
for row in plain_combs:
    gt_row = np.array([plain_uids[int(i)] for i in row[:-1]])
    ident = "-".join(list(np.sort(gt_row.astype(str))))# + "---" + str(gt_row[-1])
    plain_comb_set = plain_comb_set.union(set([ident]))

        
if CONFIG_DICT["Verbose"]:
    print("Done.")

start_emb_plain = time.time()


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

for nn in num_hashes:
    CONFIG_DICT["SAULNumHash"] = nn
    for oo in overlaps:
        uids = orig_uids.copy()
        data = orig_data.copy()
        CONFIG_DICT["Overlap"]= oo
        start_total = time.time()
        
        encoder = SaulEncoder(CONFIG_DICT["Secret"],CONFIG_DICT["N"], CONFIG_DICT["BFLength"],
                        CONFIG_DICT["SAULNumHash"], workers=-1)
        init_n = len(data)
        start_encoding = time.time()
        data_enc = encoder.encode(data)
        
        
        
        selected_enc = random.sample(range(data_enc.shape[0]), int(np.round(CONFIG_DICT["Overlap"] * data_enc.shape[0])))
        data_enc = data_enc[selected_enc]
        uids = [uids[i] for i in selected_enc]
        
        start_comb_enc = time.time()
        elapsed_encoding =  start_comb_enc - start_encoding
        
        if CONFIG_DICT["Verbose"]:
            print("Finding Combinations on encoded data")

        tres = max(0.55, 0.75 - (CONFIG_DICT["SAULNumHash"] * 0.05))
        #tres = 0.55
        enc_combs = calc_metrics_new(pack_rows(data_enc), tres, 20) # 0.65?
        while enc_combs.shape[0] > (plain_combs.shape[0]*1.2) and tres < 1:
            enc_combs = enc_combs[enc_combs[:,-1] > tres]
            tres += 0.001
        
        enc_edgelist = np.vstack([enc_combs[:,[0,1]], enc_combs[:,[0,2]], enc_combs[:,[0,3]],
                         enc_combs[:,[1,2]], enc_combs[:,[1,3]], enc_combs[:,[2,3]]]).astype(np.uint64)
        enc_edgelist = np.unique(enc_edgelist, axis=0)

        if enc_edgelist.shape[0] == 0:
            print("Skipping doe to no found combs: %s, %s, %s" % (bb, tt, oo))
            continue
        np.savetxt("data/edgelists/enc_comb_el.edg", enc_edgelist, delimiter="\t", fmt=["%i", "%i"])
        
        elapsed_comb_enc = time.time() - start_comb_enc
        
        start_ground_truth = time.time()
        
        enc_comb_set = set()
        for row in enc_combs:
            gt_row = np.array([uids[int(i)] for i in row[:-1]])
            ident = "-".join(list(np.sort(gt_row.astype(str))))# + "---" + str(gt_row[-1])
            #ident = "-".join(list(np.sort(gt_row[:-1].astype(str)))) + "---" + str(gt_row[-1])
            #ident = "-".join(list(np.sort(row[:-2].astype(int).astype(str)))) + "---" + str(int(row[-2]))
            enc_comb_set = enc_comb_set.union(set([ident]))
        
        shared = len(enc_comb_set.intersection(plain_comb_set))
        if CONFIG_DICT["Verbose"]:
            print("--Encoded Data--\nCombs found: %i\nUnique: %i" % (len(enc_combs), len(enc_comb_set)))
            print("--Plain Data--\nCombs found: %i\nUnique: %i\n\n" % (len(plain_combs), len(plain_comb_set)))
            print("Overlap: %i (%0.3f%%)" % (shared, shared/max(len(plain_comb_set), len(enc_comb_set))*100))
        
        start_emb_enc = time.time()
        elapsed_ground_truth = start_emb_enc - start_ground_truth
        
        
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
        
        aligner = WassersteinAligner(reg_init=0.01, reg_ws=0.05,
                                     batchsize=min(plain_embeddings.shape[0], enc_embeddings.shape[0]), lr=50, n_iter_init=100,
                                     n_iter_ws=50, n_epoch=100,
                                     lr_decay=0.999, apply_sqrt=True, early_stopping=10,
                                     verbose=CONFIG_DICT["Verbose"])
        transformation_matrix = aligner.align(plain_embeddings, enc_embeddings)
        enc_embeddings = np.dot(enc_embeddings, transformation_matrix.T)
        
        start_match = time.time()
        elapsed_align = start_match - start_align
        
        enc_ord_uids = [uids[int(i)] for i in enc_ordering]
        plain_ord_uids = [plain_uids[int(i)] for i in plain_ordering]
        
        matcher = MinWeightMatcher("euclidean")
        m = matcher.match(enc_embeddings, enc_ord_uids, plain_embeddings, plain_ord_uids)
        
        elapsed_match = time.time() - start_match
        elapsed_relevant = ((time.time() - start_comb_enc) - elapsed_ground_truth) + elapsed_comb_plain + elapsed_emb_plain
        
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
