#
# pre_processing.py - pre processing the data to predict secondary structures
#
# Usage: python3 pre_processing.py
#
# Jeff - Jul, 2022

# Import
import os
import biolib
from Bio import SeqIO

# Pick PU
#os.environ["CUDA_VISIBLE_DEVICE"] = "2"

# Pre_processing
nsp3 = biolib.load('DTU/NetSurfP_3')
input_file = "swiss_prot_bacteria_june_2022_processed.fasta"

data = list(SeqIO.parse(input_file, "fasta"))

count = 0

for seq in data:
    if "U" in seq.seq:
        count = count + 1

print(count)

print(len(data))

