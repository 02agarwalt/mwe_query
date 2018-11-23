import os
import argparse
from collections import OrderedDict
import io
from collections import defaultdict

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

from logging import getLogger
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch import Tensor as torch_tensor
import torch

from src.evaluation import get_word_translation_accuracy

# main
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
parser.add_argument("--query", type=str, default="")
parser.add_argument("--file_dir", type=str, default="")


# parse parameters
params = parser.parse_args()

# check parameters
assert params.src_lang, "source language undefined"
assert os.path.isfile(params.src_emb)
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

query_words = params.query.split()
file_dir = params.file_dir

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _ = build_model(params, False)
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
evaluator = Evaluator(trainer)

src_emb = trainer.src_emb
tgt_emb = trainer.tgt_emb
src_dico = trainer.src_dico
tgt_dico = trainer.tgt_dico
mapping = trainer.mapping
discriminator = trainer.discriminator
params = trainer.params

src_emb = mapping(src_emb.weight).data
tgt_emb = tgt_emb.weight.data

lang1 = src_dico.lang
word2id1 = src_dico.word2id
emb1 = src_emb
lang2 = tgt_dico.lang
word2id2 = tgt_dico.word2id
emb2 = tgt_emb

emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

# query word
query_list = []
for query_word in query_words:
    if not query_word in word2id2:
        continue
    tens = torch.LongTensor(1, 1)
    tens[0, 0] = word2id2[query_word]
    query = emb2[tens[:, 0]]
    query_list.append(query)

scores = defaultdict(float)
# iterate through all files
for filename in os.listdir(params.file_dir):
    path = os.path.join(params.file_dir, filename)
    file_words = [word for line in open(path, 'r') for word in line.split()]
    
    if query_word in file_words:
        scores[filename] += 0.0 # tune?
        continue
    
    word_list = []
    for word in file_words:
        if word in word2id1:
            word_list.append(word)

    if not word_list:
        continue
    
    tens = torch.LongTensor(len(word_list), 1)
    for i, word in enumerate(word_list):
        tens[i, 0] = word2id1[word]
    query2 = emb1[tens[:, 0]]

    for query in query_list:
        dists = query.mm(query2.transpose(0, 1))
        scores[filename] += torch.min(dists)

#for key, val in scores.items():
#    print(key, val)

counter = 0
k = 10
for key in sorted(scores, key=scores.get, reverse=False):
    if counter < k:
        print(key)
    counter += 1
