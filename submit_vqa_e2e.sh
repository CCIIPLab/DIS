python run_e2e.py --seed 8888 --lr_decrease_start 10 \
    --num_epochs 18 --lr_default 0.0001 --lr_decay_rate 0.5 \
    --output exp --id baseline_e2e \
    --meta "meta_info" --max_layer 5 \
    --model ProgramTransformer --stacking 2 --visual_dim 2048 \
    --coordinate_dim 6 --hidden_dim 512 --n_head 8 --dropout 0.1 \
    --num_tokens 32 --pre_layers 3 --feat_size 10 --do_submission \
    --length 9 --weight 0.5 \
    --batch_size 64 --num_workers 10 --intermediate_num 4 --object_grids 256 \
    --accumulate 2 \
    --word_glove "meta_info/en_emb.npy" \
    --data_dir "data/gqa_program_t5" \
    --image_folder "data/gqa_grid_152_10_10" \
    --load_from "exp/baseline_e2e/model_ep15"
