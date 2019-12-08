On Building
、

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.5

# Machine Translation
* Training Command
```
python3 train.py data-bin/iwslt14.tokenized.de-en.share \
    -a transformer_iwslt_de_en --share-all-embeddings  --encoder-normalize-before  --decoder-normalize-before --attention-dropout 0.1 --activation-dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --dropout 0.3  --lnv 'adanorm' --adanorm_scale 1 --init_method 'km'\
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
    --lr 0.0015 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
    --max-tokens 4096 --save-dir checkpoints/temp --max-update 90000 \
    --update-freq 2 --no-progress-bar --log-interval 1000 \
    --no-epoch-checkpoints \
    --ddp-backend no_c10d \
    --save-interval-updates 10000 --keep-interval-updates 5 \
    | tee -a checkpoints/save.txt
```
* Testing Command
```
python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/temp/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/checkpoint_best.txt
```


# License
MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

```bibtex
@inproceedings{adanorm,
  title = {Understanding and Improving Layer Normalization},
  author = {Xu, Jingjing and Sun, Xu and Zhang, Zhiyuan and Zhao, Guangxiang and Lin, Junyang},
  booktitle = {Proceedings of NeurIPS 2019},
  year = {2019},
}
```
