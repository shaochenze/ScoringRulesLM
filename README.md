# Language Generation with Strictly Proper Scoring Rules

The official repository for the ICML 2024 paper "**[``Language Generation with Strictly Proper Scoring Rules``](https://arxiv.org/pdf/2405.18906)**". Based on the open-source toolkit [fairseq-0.12.2](https://github.com/facebookresearch/fairseq/tree/v0.12.2), we implemented scoring rules based losses in [scoring_rule_loss.py](https://github.com/shaochenze/ScoringRulesLM/blob/main/fairseq/criterions/scoring_rule_loss.py). These losses do not perform as well as the cross-entropy loss when training models from scratch, but they show substantial improvements when fine-tuning models pre-trained with cross-entropy.

# Requirements

+ Python version >= 3.8
+ Pytorch version >= 1.10.0
+ Build fairseq with `python setup.py build_ext --inplace`

# Replicate the TED results

Follow these instructions to replicate results on the TED dataset. For other datasets, adjust the hyper-parameters as per the guidelines in the paper.

## Pre-processing

We use the tokenized TED dataset released by [VOLT](https://github.com/Jingjing-NLP/VOLT), which can be downloaded [here](https://drive.google.com/drive/folders/1FNH7cXFYWWnUdH2LyUFFRYmaWYJJveKy) and pre-processed into subword units by [prepare-ted-bilingual.sh](https://github.com/Jingjing-NLP/VOLT/blob/master/examples/prepare-ted-bilingual.sh).

For convenience, we include the pre-processed TED Fr-En dataset in this repository. Convert it into the fairseq format by running:

```
TEXT=./data
python preprocess.py --source-lang fr --target-lang en \
        --trainpref $TEXT/fr-en.train \
        --validpref $TEXT/fr-en.valid \
        --testpref $TEXT/fr-en.test \
        --destdir data-bin/ted_fren \
        --joined-dictionary  --workers 16
```

## Training

Pre-train the Transformer model for 13k steps using the cross-entropy loss, and then fine-tune with the Brier score for an additional 5k steps.  To fine-tune with the Spherical score instead, set `--scoring-rule` to `spherical`.

```
data_dir=data-bin/ted_fren
save_dir=output/fren_brier

# Pre-train with the logarithmic score for 13k steps
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $data_dir \
    --scoring-rule logarithmic --dropout 0.3 --fp16  --save-dir $save_dir \
    --arch transformer_wmt_en_de  --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 \
    --weight-decay 0.0 --criterion scoring_rule_loss --score-smoothing 0.1 --max-tokens 4096 --update-freq 1\
    --no-progress-bar --log-format json --log-interval 100 --save-interval-updates 500 \
    --max-update 13000 --keep-interval-updates 5 --no-epoch-checkpoints

# Fine-tune with the brier score for 5k steps
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $data_dir \
    --scoring-rule brier --dropout 0.3 --fp16  --save-dir $save_dir \
    --arch transformer_wmt_en_de  --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 \
    --weight-decay 0.0 --criterion scoring_rule_loss --score-smoothing 0.1 --max-tokens 4096 --update-freq 1\
    --no-progress-bar --log-format json --log-interval 100 --save-interval-updates 500 \
    --max-update 18000 --keep-interval-updates 5 --no-epoch-checkpoints

python average_checkpoints.py --inputs $save_dir \
 --num-update-checkpoints 5  --output $save_dir/average-model.pt
```

For comparison, you may also train a baseline Transformer model for 18k steps using the same procedure.

```
data_dir=data-bin/ted_fren
save_dir=output/fren_base

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $data_dir \
    --scoring-rule logarithmic --dropout 0.3 --fp16  --save-dir $save_dir \
    --arch transformer_wmt_en_de  --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 \
    --weight-decay 0.0 --criterion scoring_rule_loss --score-smoothing 0.1 --max-tokens 4096 --update-freq 1\
    --no-progress-bar --log-format json --log-interval 100 --save-interval-updates 500 \
    --max-update 18000 --keep-interval-updates 5 --no-epoch-checkpoints

python average_checkpoints.py --inputs $save_dir \
 --num-update-checkpoints 5  --output $save_dir/average-model.pt
```

The above commands assume 8 GPUs on the machine. When the number of GPUs is different, adapt `--update-freq` to make sure that the batch size is 32k.

## Inference

Run the following command for inference.

```
python generate.py data-bin/ted_fren  --path output/fren_brier/average-model.pt --gen-subset test --beam 5 --batch-size 100 --remove-bpe --lenpen 1 > out
# because fairseq's output is unordered, we need to recover its order
grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.en
sed -r 's/(@@ )|(@@ ?$)//g' data/fr-en.test.en > ref.en
perl multi-bleu.perl ref.en < pred.en
```

Expected BLEU scores are ~40.8 for the Transformer baseline and ~41.4 for models fine-tuned with Brier/Spherical scores.

# Citation

If you find the resources in this repository useful, please cite as:

```bibtex
@inproceedings{scoringrule,
  title = {Language Generation with Strictly Proper Scoring Rules},
  author= {Chenze Shao and Fandong Meng and Yijin Liu and Jie Zhou},
  booktitle = {Proceedings of ICML 2024},
  year = {2024},
}
```
