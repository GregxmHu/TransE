cd src
python train.py\
 --train_data_path ../train_id_triplets.tsv\
 --test_data_path ../test_id_triplets.tsv\
 --checkpoint_save_path /data1/private/huxiaomeng/kg/fp/checkpoints/best.bin\
 --batch_size 128\
 --log_dir /data1/private/huxiaomeng/kg/fp/logs/\
 --epochs 20\
 --warmup_steps 0\
 --lr 0.5\
 --seed 13\
 --norm_number 2\

 cd ..