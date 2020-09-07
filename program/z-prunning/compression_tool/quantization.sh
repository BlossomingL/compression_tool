# 量化运行脚本
python main.py --mode quantization \
               --model resnet100 \
               --best_model_path work_space/model_train_best/2019-09-29-11-37_SVGArcFace-O1-b0.4s40t1.1_fc_0.4_112x112_2019-09-27-Adult-padSY-Bus_fResNet100v3cv-d512_model_iter-340000.pth \
               --from_data_parallel \
               --data_source company \
               --test_root_path data/test_data/fc_0.4_112x112 \
               --img_list_label_path data/test_data/fc_0.4_112x112/pair_list/id_life_image_list_bmppair.txt \
               --test_batch_size 256 \
               --fp16

