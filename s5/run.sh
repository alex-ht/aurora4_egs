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
feat=mfcc
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
 steps/make_mfcc.sh  --nj 16 --cmd "$train_cmd" \
   data/$x exp_$feat/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp_$feat/make_mfcc/$x $mfccdir || exit 1;
done

# make fbank features
#fbankdir=fbank
#mkdir -p data-fbank
#for x in train_si84_${training} dev_0330 dev_1206 test_eval92 test_0166; do
#  cp -r data/$x data-fbank/$x
#  steps/make_fbank.sh --nj 16 \
#    data-fbank/$x exp_$feat/make_fbank/$x $fbankdir || exit 1;
#done
fi
if [ $stage -le 1 ]; then
# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
steps/train_mono.sh --boost-silence 1.25 --nj 16 --cmd "$train_cmd"  \
  ${feat}_data/train_si84_${training} data/lang exp_$feat/mono0a_${training} || exit 1;

#(
# utils/mkgraph.sh --mono data/lang_test_tgpr exp_$feat/mono0a_${training} exp_$feat/mono0a_${training}/graph_tgpr && \
# steps/decode.sh --nj 16  \
#   exp_$feat/mono0a_${training}/graph_tgpr data/test_eval92 exp_$feat/mono0a_${training}/decode_tgpr_eval92 
#) &
fi
if [ $stage -le 2 ]; then
steps/align_si.sh --boost-silence 1.25 --nj 16 --cmd "$train_cmd" \
  ${feat}_data/train_si84_${training} data/lang exp_$feat/mono0a_${training} exp_$feat/mono0a_${training}_ali || exit 1;

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
  2000 10000 ${feat}_data/train_si84_${training} data/lang exp_$feat/mono0a_${training}_ali exp_$feat/tri1_${training} || exit 1;
fi
#while [ ! -f data/lang_test_tgpr/tmp/LG.fst ] || \
#   [ -z data/lang_test_tgpr/tmp/LG.fst ]; do
#  sleep 20;
#done
#sleep 30;
# or the mono mkgraph.sh might be writing 
# data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.
if [ $stage -le 3 ]; then
steps/align_si.sh --nj 16 --cmd "$train_cmd" \
  ${feat}_data/train_si84_${training} data/lang exp_$feat/tri1_${training} exp_$feat/tri1_${training}_ali_si84 || exit 1;
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 15000 ${feat}_data/train_si84_${training} data/lang exp_$feat/tri1_${training}_ali_si84 exp_$feat/tri2a_${training} || exit 1;
fi
if [ $stage -le 4 ]; then
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 ${feat}_data/train_si84_${training} data/lang exp_$feat/tri1_${training}_ali_si84 exp_$feat/tri2b_${training} || exit 1;
fi
if [ $stage -le 5 ]; then
$mkgraph_cmd mkgraph.log \
utils/mkgraph.sh data/lang_test_tgpr_5k exp_$feat/tri2b_${training} exp_$feat/tri2b_${training}/graph_tgpr_5k || exit 1;
#steps/decode.sh --nj 16 \
#  exp_$feat/tri2b_${training}/graph_tgpr_5k data/test_eval92 exp_$feat/tri2b_${training}/decode_tgpr_5k_eval92 || exit 1;
fi
if [ $stage -le 6 ]; then
# Align tri2b system with si84 multi-condition data.
steps/align_si.sh  --nj 16 --cmd "$train_cmd" \
  --use-graphs true ${feat}_data/train_si84_${training} data/lang exp_$feat/tri2b_${training} exp_$feat/tri2b_${training}_ali_si84  || exit 1;
fi
if [ $stage -le 7 ]; then
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  ${feat}_data/dev_0330 data/lang exp_$feat/tri2b_${training} exp_$feat/tri2b_${training}_ali_dev_0330 || exit 1;

#steps/align_si.sh  --nj 16 \
#  ${feat}_data/dev_1206 data/lang exp_$feat/tri2b_${training} exp_$feat/tri2b_${training}_ali_dev_1206 || exit 1;
fi
echo "Now begin train DNN systems on ${training} data"

if [ $stage -le 7 ]; then
# RBM pretrain
dir=exp_$feat/tri3a_${training}_dnn_pretrain
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/nnet/pretrain_dbn.sh --nn-depth 4 --rbm-iter 3 ${feat}_data/train_si84_${training} $dir
fi
dir=exp_$feat/tri3a_${training}_dnn
ali=exp_$feat/tri2b_${training}_ali_si84
ali_dev=exp_$feat/tri2b_${training}_ali_dev_0330
feature_transform=exp_$feat/tri3a_${training}_dnn_pretrain/final.feature_transform
dbn=exp_$feat/tri3a_${training}_dnn_pretrain/4.dbn
if [ $stage -le 8 ]; then
# Fine turning
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    ${feat}_data/train_si84_${training} ${feat}_data/dev_0330 data/lang $ali $ali_dev $dir || exit 1;
fi
if [ $stage -le 9 ]; then
# make HCLG
if [ ! -f exp_$feat/tri3a_${training}_dnn/graph_tgpr_5k/HCLG.fst ]; then
$mkgraph_cmd mkgraph.log \
  utils/mkgraph.sh data/lang_test_tgpr_5k exp_$feat/tri3a_${training}_dnn exp_$feat/tri3a_${training}_dnn/graph_tgpr_5k || exit 1;
fi
# decode
dir=exp_$feat/tri3a_${training}_dnn
steps/nnet/decode.sh --cmd "$cuda_cmd" --nj 1 --acwt 0.10 --use-gpu yes --config conf/decode_dnn.config \
  exp_$feat/tri3a_${training}_dnn/graph_tgpr_5k ${feat}_data/dev_0330 $dir/decode_tgpr_5k_dev0330 || exit 1;
steps/nnet/decode.sh --cmd "$cuda_cmd" --nj 1 --acwt 0.10 --use-gpu yes --config conf/decode_dnn.config \
  exp_$feat/tri3a_${training}_dnn/graph_tgpr_5k ${feat}_data/test_eval92 $dir/decode_tgpr_5k_eval92 || exit 1;
fi
#


if [ $stage -le 100 ]; then
for d in `ls -d exp_$feat/*/*decode*`; do
  echo $d;
  cat ${d}/wer_* | utils/best_wer.sh;
done
fi
exit 0;
