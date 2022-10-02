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
import torch

# Pick PU
os.environ["CUDA_VISIBLE_DEVICE"] = "2"

# Pre_processing
nsp3 = biolib.load('DTU/NetSurfP_3')
input_file = "swiss_prot_bacteria_june_2022_processed.fasta"

data = list(SeqIO.parse(input_file, "fasta"))

os.makedirs("./data/raw")

dn = 25

for i in range(0, int(len(data)/dn)):
    file_des = str("./data/raw/"+str(i)+".fasta")
    with open(file_des, "w") as output_handle:
        if (i+1)*dn < len(data):
            SeqIO.write(data[i*dn : (i+1)*dn], output_handle, "fasta")


#result = nsp3.cli(args='-i ./swiss_prot_bacteria_june_2022.fasta')
#result.save_files("pre_processing_results/")

#result = nsp3.cli(args='-i ./subset1.fasta -o subset1_results -gpu 2')

os.makedirs("./data/pre_processing")

for i in range(0, int(len(data)/dn)):
    torch.cuda.empty_cache()
    input_file = str("./data/raw/"+str(i)+".fasta")
    output_file = str("./data/pre_processing/"+str(i)+".csv")
    result = nsp3.cli(args='-i {} -o {} -gpu = 2'.format(input_file, output_file))
