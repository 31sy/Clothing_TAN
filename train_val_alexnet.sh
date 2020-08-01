CUDA_VISIBLE_DEVICES=1  python -u main.py --gpu 0 --save_path ./snapshot_alexnet_TAN/ \
		--model alexnet 2>&1 | tee ./train_TAN.log

