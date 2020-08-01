CUDA_VISIBLE_DEVICES=1  python -u main_vgg16.py --gpu 0 --save_path ./snapshot_vgg16_TAN/ \
		--model vgg16 2>&1 | tee ./train_TAN_vgg16.log