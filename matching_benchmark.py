import os
import numpy as np
from tqdm import tqdm
import random
import time
import glob

from collections import Counter

from matchers.parallelized import SymmetricMatcher
from bad_utils import read_tsv, save_tsv
from encoders.bf_encoder import BFEncoder
from encoders.saul_encoder import SaulEncoder


# Cofiguration for Encoder
ENC_CONFIG = {
        "AliceSecret": str(random.randint(1,1000000)),
        "Ngram_size": 2,
        # For BAD encoding
        "BF_length": 1024,
        "BF_k": 20,
        "BAD_t": 10,
        "BAD_length": 1024
    }

# Select Encoding Schemes to compare
BF = True # Standard Bloom Filters
BAD = True # Bloom Filters with Diffusion
SAUL = True # SAUL

# Print Status Messages
verbose = True

# Directories containing the datasets:
datadir_1 = "./data/matching_eval"
datadir_2 = "./data/matching_eval/db2"

datasets_1 = glob.glob(glob.escape(datadir_1) + "/*.tsv")
datasets_2 = glob.glob(glob.escape(datadir_2) + "/*.tsv")

# Define which columns to use
colnames_fake = ["Name", "Surname", "Address", "City", "State", "ZIP", "Birthday"] 
colnames_euro = ["Name", "Surname", "Address", "ZIP", "Birthday"]
colnames_ncv = ["Name", "MiddleName", "Surname", "Address", "City", "ZIP"]

# Define 
overlaps = [o/100 for o in range(10,110,10)]
saul_k = list(range(1,11,1))



for data2 in tqdm(datasets_2):
    for oo in tqdm(overlaps, leave=False):
        for cur_k in saul_k:
            ENC_CONFIG["SAULNumHash"] = cur_k
            if "fakename_ger" in data2:
                data1 = datadir_1+'/fakename_ger.tsv'
                colnames = colnames_fake
            elif "fakename_eng" in data2:
                data1 = datadir_1+'/fakename_eng.tsv'
                colnames = colnames_fake
            elif "euro" in data2:
                data1 =  datadir_1+'/euro_ground_truth_prepro_fullatt.tsv'
                colnames = colnames_euro
            elif "ncvoter" in data2:
                data1 = datadir_1+'/ncvoter_a.tsv'
                colnames = colnames_ncv
            elif "fakename_fra" in data2:
                data1 = datadir_1+'/fakename_fra.tsv'
                colnames = colnames_fake
            else:
                raise "Unsupported Dataset"
            
            # Read Alice's data, i.e. the data with mistakes
            alice_data, alice_uids = read_tsv(data1)
            reference_data, reference_uids = read_tsv(data2)
    
            selected_alice = random.sample(range(len(alice_data)), int(np.round(oo * len(alice_data))))
            
            alice_data = [alice_data[p] for p in selected_alice]
            alice_uids = [alice_uids[p] for p in selected_alice]
            
            # Randomly shuffle the data
            perm1 = np.random.permutation(len(alice_data))
            perm2 = np.random.permutation(len(reference_data))
            
            alice_data = [alice_data[p] for p in perm1]
            alice_uids = [alice_uids[p] for p in perm1]
            
            reference_data = [reference_data[p] for p in perm2]
            reference_uids = [reference_uids[p] for p in perm2]
        
            
            keys = ["Dataset_1", "Dataset_2", "True_Matches", "Overlap", "N_D1", "N_D2"]
            vals = [data1, data2, len(set(alice_uids).intersection(set(reference_uids))), oo, len(alice_data), len(reference_data)]
            for k, v in ENC_CONFIG.items():
                keys.append(k)
                vals.append(v)

            # Set up encoders        
            bf_encoder = BFEncoder(ENC_CONFIG["AliceSecret"], ENC_CONFIG["BF_length"],
                            ENC_CONFIG["BF_k"], ENC_CONFIG["Ngram_size"], False,
                            ENC_CONFIG["BAD_length"], ENC_CONFIG["BAD_t"])
         
            bad_encoder = BFEncoder(ENC_CONFIG["AliceSecret"], ENC_CONFIG["BF_length"],
                                ENC_CONFIG["BF_k"], ENC_CONFIG["Ngram_size"], True,
                                ENC_CONFIG["BAD_length"], ENC_CONFIG["BAD_t"])
        
            saul_encoder = SaulEncoder(ENC_CONFIG["AliceSecret"],ENC_CONFIG["Ngram_size"], ENC_CONFIG["BF_length"],
                            ENC_CONFIG["SAULNumHash"], workers=-1)

            # Trigger JIT-Compiler for fair time measurement
            _ = saul_encoder.encode([["ABC"]])
        
            
            if BF and cur_k == saul_k[0]: # Only run once
                if verbose:
                    print("-- BF --")
                    print("Encoding...")
                bf_enc_d1_start = time.time()
                alice_bfs = bf_encoder.encode(alice_data)
                bf_enc_d2_start = time.time()
                reference_bfs = bf_encoder.encode(reference_data)
                bf_enc_end = time.time()

                bf_enc_duration = bf_enc_end - bf_enc_d1_start
                bf_enc_d1_duration = bf_enc_d2_start - bf_enc_d1_start
                bf_enc_d2_duration = bf_enc_end - bf_enc_d2_start
                if verbose:
                    print("Done")
        
                bf_correct = 0
                bf_wrong = 0
                bf_ambiguous = 0
                if verbose:
                    print("Matching...")
                matcher = SymmetricMatcher(metric="dice")
                m = matcher.match(alice_bfs, alice_uids, reference_bfs, reference_uids)
                bf_total_end = time.time()
                if verbose:
                    print("Done.")
                    
                for s, l in m.items():
                    if s[2:] == l[2:]:
                        bf_correct += 1
                    else:
                        bf_wrong += 1
                
                keys += ["BF_Correct", "BF_Wrong",  "BF_Enc_D1_Duration", "BF_Enc_D2_Duration", "BF_Enc_Duration", "BF_Total_Duration"]
                vals += [bf_correct, bf_wrong, bf_enc_d1_duration, bf_enc_d2_duration, bf_enc_duration, bf_total_end - bf_enc_d1_start]
            else:
                keys += ["BF_Correct", "BF_Wrong",  "BF_Enc_D1_Duration", "BF_Enc_D2_Duration", "BF_Enc_Duration", "BF_Total_Duration"]
                vals += [-1, -1, -1, -1, -1, -1]
                
        
            if BAD and cur_k == saul_k[0]:
        
                if verbose:
                    print("-- BAD --")
                    print("Encoding...")
                bad_enc_d1_start = time.time()
                alice_bads = bad_encoder.encode(alice_data)
                bad_enc_d2_start = time.time()
                reference_bads = bad_encoder.encode(reference_data)
                bad_enc_end = time.time()
                
                bad_enc_duration = bad_enc_end - bad_enc_d1_start
                bad_enc_d1_duration = bad_enc_d2_start - bad_enc_d1_start
                bad_enc_d2_duration = bad_enc_end - bad_enc_d2_start
                
                if verbose:
                    print("Done")

        
                
                bad_correct = 0
                bad_wrong = 0
                bad_ambiguous = 0
                if verbose:
                    print("Matching...")
                matcher = SymmetricMatcher(metric="dice")
                m = matcher.match(alice_bads, alice_uids, reference_bads, reference_uids)
                bad_total_end = time.time()
                if verbose:
                    print("Done.")
                    
                for s, l in m.items():
                    if s[2:] == l[2:]:
                        bad_correct += 1
                    else:
                        bad_wrong += 1
        
                keys += ["BAD_Correct", "BAD_Wrong",  "BAD_Enc_D1_Duration", "BAD_Enc_D2_Duration", "BAD_Enc_Duration", "BAD_Total_Duration"]
                vals += [bad_correct, bad_wrong, bad_enc_d1_duration, bad_enc_d2_duration, bad_enc_duration, bad_total_end - bad_enc_d1_start]
            else:
                keys += ["BAD_Correct", "BAD_Wrong",  "BAD_Enc_D1_Duration", "BAD_Enc_D2_Duration", "BAD_Enc_Duration", "BAD_Total_Duration"]
                vals += [-1, -1, -1,-1, -1, -1]
        
            if SAUL:
                if verbose:
                    print("-- SAUL --")
                    print("Encoding...")
        
                saul_enc_d1_start = time.time()
                alice_sauls = saul_encoder.encode(alice_data)
                saul_enc_d2_start = time.time()
                reference_sauls = saul_encoder.encode(reference_data)
                saul_enc_end = time.time()
                
                saul_enc_duration = saul_enc_end - saul_enc_d1_start
                saul_enc_d1_duration = saul_enc_d2_start - saul_enc_d1_start
                saul_enc_d2_duration = saul_enc_end - saul_enc_d2_start
                
                if verbose:
                    print("Done")
                saul_correct = 0
                saul_wrong = 0
                saul_ambiguous = 0
                if verbose:
                    print("Matching...")
                matcher = SymmetricMatcher(metric="dice")
                m = matcher.match(alice_sauls, alice_uids, reference_sauls, reference_uids)
                saul_total_end = time.time()
                if verbose:    
                    print("Done.")
                for s, l in m.items():
                    if s[2:] == l[2:]:
                        saul_correct += 1
                    else:
                        saul_wrong += 1
                keys += ["SAUL_Correct", "SAUL_Wrong",  "SAUL_Enc_D1_Duration", "SAUL_Enc_D2_Duration", "SAUL_Enc_Duration", "SAUL_Total_Duration"]
                vals += [saul_correct, saul_wrong, saul_enc_d1_duration, saul_enc_d2_duration, saul_enc_duration, saul_total_end - saul_enc_d1_start]
        
            else:
                keys += ["SAUL_Correct", "SAUL_Wrong",  "SAUL_Enc_D1_Duration", "SAUL_Enc_D2_Duration", "SAUL_Enc_Duration", "SAUL_Total_Duration"]
                vals += [-1, -1, -1, -1, -1, -1]
            
            if not os.path.isfile("data/matching_eval/benchmark.tsv"):
                save_tsv([keys], "data/matching_eval/benchmark.tsv")
            
            save_tsv([vals], "data/matching_eval/benchmark.tsv", mode="a")
