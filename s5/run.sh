#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

stage=-1
aurora4=/mnt/corpus/AURORA4
#we need lm, trans, from WSJ0 CORPUS
wsj0=/mnt/corpus/WSJ0
training=clean
#clean or multi
[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $stage -le -1 ]; then
# data preprocessing
local/aurora4_data_prep.sh $aurora4 $wsj0
local/wsj_prepare_dict.sh || exit 1;
utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;
local/aurora4_format_data.sh || exit 1;
fi

if [ $stage -le 0 ]; then
# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in train_si84_${training} test_eval92 test_0166 dev_0330 dev_1206; do 
 steps/make_mfcc.sh  --nj 8 \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

# make fbank features
#fbankdir=fbank
#mkdir -p data-fbank
#for x in train_si84_${training} dev_0330 dev_1206 test_eval92 test_0166; do
#  cp -r data/$x data-fbank/$x
#  steps/make_fbank.sh --nj 8 \
#    data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
#done
fi
if [ $stage -le 1 ]; then
# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
steps/train_mono.sh --boost-silence 1.25 --nj 8  \
  data/train_si84_${training} data/lang exp/mono0a_${training} || exit 1;

#(
# utils/mkgraph.sh --mono data/lang_test_tgpr exp/mono0a_${training} exp/mono0a_${training}/graph_tgpr && \
# steps/decode.sh --nj 8  \
#   exp/mono0a_${training}/graph_tgpr data/test_eval92 exp/mono0a_${training}/decode_tgpr_eval92 
#) &
fi
if [ $stage -le 2 ]; then
steps/align_si.sh --boost-silence 1.25 --nj 8  \
  data/train_si84_${training} data/lang exp/mono0a_${training} exp/mono0a_${training}_ali || exit 1;

steps/train_deltas.sh --boost-silence 1.25 \
  2000 10000 data/train_si84_${training} data/lang exp/mono0a_${training}_ali exp/tri1_${training} || exit 1;
fi
#while [ ! -f data/lang_test_tgpr/tmp/LG.fst ] || \
#   [ -z data/lang_test_tgpr/tmp/LG.fst ]; do
#  sleep 20;
#done
#sleep 30;
# or the mono mkgraph.sh might be writing 
# data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.
if [ $stage -le 3 ]; then
steps/align_si.sh --nj 8 \
  data/train_si84_${training} data/lang exp/tri1_${training} exp/tri1_${training}_ali_si84 || exit 1;
steps/train_deltas.sh  \
  2500 15000 data/train_si84_${training} data/lang exp/tri1_${training}_ali_si84 exp/tri2a_${training} || exit 1;
fi
if [ $stage -le 4 ]; then
steps/train_lda_mllt.sh \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train_si84_${training} data/lang exp/tri1_${training}_ali_si84 exp/tri2b_${training} || exit 1;
fi
if [ $stage -le 5 ]; then
utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri2b_${training} exp/tri2b_${training}/graph_tgpr_5k || exit 1;
#steps/decode.sh --nj 8 \
#  exp/tri2b_${training}/graph_tgpr_5k data/test_eval92 exp/tri2b_${training}/decode_tgpr_5k_eval92 || exit 1;
fi
if [ $stage -le 6 ]; then
# Align tri2b system with si84 multi-condition data.
steps/align_si.sh  --nj 8 \
  --use-graphs true data/train_si84_${training} data/lang exp/tri2b_${training} exp/tri2b_${training}_ali_si84  || exit 1;

steps/align_si.sh  --nj 8 \
  data/dev_0330 data/lang exp/tri2b_${training} exp/tri2b_${training}_ali_dev_0330 || exit 1;

steps/align_si.sh  --nj 8 \
  data/dev_1206 data/lang exp/tri2b_${training} exp/tri2b_${training}_ali_dev_1206 || exit 1;
fi
echo "Now begin train DNN systems on ${training} data"

if [ $stage -le 7 ]; then
# RBM pretrain
dir=exp/tri3a_${training}_dnn_pretrain
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/nnet/pretrain_dbn.sh --nn-depth 4 --rbm-iter 3 data/train_si84_${training} $dir
fi
dir=exp/tri3a_${training}_dnn
ali=exp/tri2b_${training}_ali_si84
ali_dev=exp/tri2b_${training}_ali_dev_1206
feature_transform=exp/tri3a_${training}_dnn_pretrain/final.feature_transform
dbn=exp/tri3a_${training}_dnn_pretrain/4.dbn
if [ $stage -le 8 ]; then
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data/train_si84_${training} data/dev_1206 data/lang $ali $ali_dev $dir || exit 1;
fi
if [ $stage -le 9 ]; then
if [ ! -f exp/tri3a_${training}_dnn/graph_tgpr_5k/HCLG.fst ]; then
  utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3a_${training}_dnn exp/tri3a_${training}_dnn/graph_tgpr_5k || exit 1;
fi
dir=exp/tri3a_${training}_dnn
steps/nnet/decode.sh --nj 1 --acwt 0.10 --use-gpu yes --config conf/decode_dnn.config \
  exp/tri3a_${training}_dnn/graph_tgpr_5k data/dev_0330 $dir/decode_tgpr_5k_dev0330 || exit 1;
fi
#


if [ $stage -le 100 ]; then
for d in `ls -d exp/*/*decode*`; do
  echo $d;
  cat ${d}/wer_* | utils/best_wer.sh;
done
fi
exit 0;
