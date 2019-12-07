save=deen_transformer_prenorm_km_init_for_figure
init_method='km'
export CUDA_VISIBLE_DEVICES=1
for seed in 1
do
    cur_save=${save}
    python3 train.py data-bin/iwslt14.tokenized.de-en.share \
    -a transformer_iwslt_de_en --share-all-embeddings  --encoder-normalize-before  --decoder-normalize-before --attention-dropout 0.1 --activation-dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --dropout 0.3 --init_method ${init_method} \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
    --lr 0.0015 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
    --max-tokens 4096 --save-dir checkpoints/$cur_save --max-update 90000 \
    --update-freq 2 --no-progress-bar --log-interval 1000 \
    --no-epoch-checkpoints \
    --ddp-backend no_c10d \
    --save-interval-updates 10000 --keep-interval-updates 5 \
    | tee -a checkpoints/${cur_save}.txt
    #python3 average_checkpoints.py --inputs checkpoints/${cur_save}  --num-epoch-checkpoints 10 --output checkpoints/${cur_save}/avg_final.pt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_avg_final.txt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
done

#save=deen_transformer_prenorm_km_init_wb_for_figure
#lnv='nowb'
#for seed in 1
#do
#    cur_save=${save}
#    python3 train.py data-bin/iwslt14.tokenized.de-en.share \
#    -a transformer_iwslt_de_en --share-all-embeddings  --encoder-normalize-before  --decoder-normalize-before --attention-dropout 0.1 --activation-dropout 0.1 \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --dropout 0.3 --lnv ${lnv} --init_method ${init_method} \
#    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
#    --lr 0.0015 --min-lr 1e-09 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
#    --max-tokens 4096 --save-dir checkpoints/$cur_save --max-update 90000 \
#    --update-freq 2 --no-progress-bar --log-interval 1000 \
#    --no-epoch-checkpoints \
#    --ddp-backend no_c10d \
#    --save-interval-updates 10000 --keep-interval-updates 5 \
#    | tee -a checkpoints/${cur_save}.txt
    #python3 average_checkpoints.py --inputs checkpoints/${cur_save}  --num-epoch-checkpoints 10 --output checkpoints/${cur_save}/avg_final.pt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_avg_final.txt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
#done


#save=deen_transformer_prenorm_km_init_detachmean
#lnv='nowb'
#for seed in 1
#do
#    cur_save=${save}
    #python3 train.py data-bin/iwslt14.tokenized.de-en.share \
    #-a transformer_iwslt_de_en --share-all-embeddings  --encoder-normalize-before  --decoder-normalize-before --attention-dropout 0.1 --activation-dropout 0.1 \
    #--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    #--dropout 0.3 --lnv ${lnv} --init_method ${init_method} --mean_detach 1 \
    #--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
    #--lr 0.0015 --min-lr 1e-09 \
    #--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
    #--max-tokens 4096 --save-dir checkpoints/$cur_save --max-update 30000 \
    #--update-freq 2 --no-progress-bar --log-interval 1000 \
    #--no-epoch-checkpoints \
    #--ddp-backend no_c10d \
    #--save-interval-updates 10000 --keep-interval-updates 5 \
    #| tee -a checkpoints/${cur_save}.txt
    #python3 average_checkpoints.py --inputs checkpoints/${cur_save}  --num-epoch-checkpoints 10 --output checkpoints/${cur_save}/avg_final.pt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_avg_final.txt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
#done

#save=deen_transformer_prenorm_km_init_detachvariance
#lnv='nowb'
#for seed in 1
#do
#    cur_save=${save}
    #python3 train.py data-bin/iwslt14.tokenized.de-en.share \
    #-a transformer_iwslt_de_en --share-all-embeddings  --encoder-normalize-before  --decoder-normalize-before --attention-dropout 0.1 --activation-dropout 0.1 \
    #--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    #--dropout 0.3 --lnv ${lnv} --init_method ${init_method} --std_detach 1 \
    #--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
    #--lr 0.0015 --min-lr 1e-09 \
    #--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
    #--max-tokens 4096 --save-dir checkpoints/$cur_save --max-update 30000 \
    #--update-freq 2 --no-progress-bar --log-interval 1000 \
    #--no-epoch-checkpoints \
    #--ddp-backend no_c10d \
    #--save-interval-updates 10000 --keep-interval-updates 5 \
    #| tee -a checkpoints/${cur_save}.txt
    #python3 average_checkpoints.py --inputs checkpoints/${cur_save}  --num-epoch-checkpoints 10 --output checkpoints/${cur_save}/avg_final.pt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_avg_final.txt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
#done

#save=deen_transformer_prenorm_km_init_no_norm
#lnv='no_norm'
#for seed in 1
#do
#    cur_save=${save}
    #python3 train.py data-bin/iwslt14.tokenized.de-en.share \
    #-a transformer_iwslt_de_en --share-all-embeddings  --encoder-normalize-before  --decoder-normalize-before --attention-dropout 0.1 --activation-dropout 0.1 \
    #--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    #--dropout 0.3 --lnv ${lnv} --init_method ${init_method}  \
    #--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
    #--lr 0.0015 --min-lr 1e-09 \
    #--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
    #--max-tokens 4096 --save-dir checkpoints/$cur_save --max-update 30000 \
    #--update-freq 2 --no-progress-bar --log-interval 1000 \
    #--no-epoch-checkpoints \
    #--ddp-backend no_c10d \
    #--save-interval-updates 10000 --keep-interval-updates 5 \
    #| tee -a checkpoints/${cur_save}.txt
    #python3 average_checkpoints.py --inputs checkpoints/${cur_save}  --num-epoch-checkpoints 10 --output checkpoints/${cur_save}/avg_final.pt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_avg_final.txt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
#done
#save=deen_transformer_prenorm_km_init_detachboth_for_figure
#lnv='nowb'
#for seed in 1
#do
#    cur_save=${save}
#    python3 train.py data-bin/iwslt14.tokenized.de-en.share \
#    -a transformer_iwslt_de_en --share-all-embeddings  --encoder-normalize-before  --decoder-normalize-before --attention-dropout 0.1 --activation#-dropout 0.1 \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --dropout 0.3 --lnv ${lnv} --init_method ${init_method} --std_detach 1 --mean_detach 1 \
#    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
#    --lr 0.0015 --min-lr 1e-09 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
#    --max-tokens 4096 --save-dir checkpoints/$cur_save --max-update 90000 \
#    --update-freq 2 --no-progress-bar --log-interval 1000 \
#    --no-epoch-checkpoints \
#    --ddp-backend no_c10d \
#   --save-interval-updates 10000 --keep-interval-updates 5 \
#    | tee -a checkpoints/${cur_save}.txt
    #python3 average_checkpoints.py --inputs checkpoints/${cur_save}  --num-epoch-checkpoints 10 --output checkpoints/${cur_save}/avg_final.pt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_avg_final.txt
    #python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
#done

#save=deen_transformer_prenorm_km_init_detachmean
#lnv='nowb'
#for seed in 1
#do
#    cur_save=${save}
#    python3 train.py data-bin/iwslt14.tokenized.de-en.share \
#    -a transformer_iwslt_de_en --share-all-embeddings  --encoder-normalize-before  --decoder-normalize-before --attention-dropout 0.1 --activation#-dropout 0.1 \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --dropout 0.3 --lnv ${lnv} --init_method ${init_method} --mean_detach 1 \
#    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
#    --lr 0.0015 --min-lr 1e-09 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
#    --max-tokens 4096 --save-dir checkpoints/$cur_save --max-update 30000 \
#    --update-freq 2 --no-progress-bar --log-interval 1000 \
#    --no-epoch-checkpoints \
#    --ddp-backend no_c10d \
#    --save-interval-updates 10000 --keep-interval-updates 5 \
#    | tee -a checkpoints/${cur_save}.txt
#    python3 average_checkpoints.py --inputs checkpoints/${cur_save}  --num-epoch-checkpoints 10 --output checkpoints/${cur_save}/avg_final.pt
#    python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_avg_final.txt
#    python3 generate.py data-bin/iwslt14.tokenized.de-en.share --path checkpoints/${cur_save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${cur_save}_checkpoint_best.txt
#done

