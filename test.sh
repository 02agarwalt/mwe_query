#!/bin/bash

source ~/.bashrc
# source activate py36

(source activate py36 && CUDA_VISIBLE_DEVICES=`free-gpu` python query.py --src_lang tl --tgt_lang en --src_emb best_vecs/tl-en/vectors-tl.txt --tgt_emb best_vecs/tl-en/vectors-en.txt --cuda=False --file_dir /export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1B/EVAL1/text/src --query "alcohol change")
