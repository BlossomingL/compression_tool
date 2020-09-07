#!/bin/sh

# ==============================剪枝脚本===============================================

# mobilefacenet_y2_zkx_0.7889
#python main.py    --mode prune \
#                  --model mobilefacenet_y2 \
#                  --best_model_path /media/minivision/C4BC2F49BC2F34F8/1_Work/project/Face/z-prunning/compression_tool/work_space/model_train_best/2020-03-13-18-16_CombineMargin-#zk-O1D1L-m0.9m0.4m0.15s64_le_re_0.4_112x112_2020-03-10-PadMaskGlassClean-Rest-NeighborGlass+TiktokGE5_MobileFaceNety2-d512_model_iter-146680_XCHoldClean-0.7226_PadMaskYTBYGlass-0.8612.pth  \
#                  --from_data_parallel \
#                  --save_model_pt \
#                  --data_source company \
#                  --hrank \
#                  --rank_path ./work_space/rank_conv/mobilefacenety2_limit10/ \
#                  --test_root_path /media/minivision/C4BC2F49BC2F34F8/ssd_data/Test/O2N/Pad_MaskBYYT_glass_2020-03-16/id_life_ssd12_patches/le_re_0.4_112x112 \
#                  --img_list_label_path /media/minivision/C4BC2F49BC2F34F8/ssd_data/Test/O2N/Pad_MaskBYYT_glass_2020-03-16/id_life_ssd12_result/id_life_image_list_bmppair.txt \

# resnet34_lzc
#python main.py    --mode prune \
#                  --model resnet34_lzc \
#                  --best_model_path /home/user1/linx/LiveBody/model/2019-10-06-07-23_LiveBody_re_rm_n_0.15_80x80_fake-20190924-train-data_live-0926_ResNet34v3-d128-c4_pytorch_iter_42000/2019-10-06-07-23_LiveBody_re_rm_n_0.15_80x80_fake-20190924-train-data_live-0926_ResNet34v3-d128-c4_pytorch_iter_42000.pth \
#                  --from_data_parallel \
#                  --fpgm \
#                  --save_model_pt \
#                  --test_root_path /home/user1/linx/LiveBody/testset/re_rm_n_0.15_80x80 \
#                  --img_list_label_path /home/user1/linx/LiveBody/testset/0926_Test_list_label.txt

# resnet50
python main.py    --mode prune \
                  --model resnet34 \
                  --best_model_path work_space/model_train_best/2020-07-09-18-04_CombineMargin-ljt-m0.9m0.4m0.15s64_le_re_0.4_112x112_2020-05-26-PNTMS-CLEAN-MIDDLE-70_fResNet34v3cv-d512_model_iter-96628_Idoa-0.7541_IdoaMask-0.8671_TYLG-0.7711.pth \
                  --from_data_parallel \
                  --fpgm \
                  --save_model_pt \
                  --test_root_path /media/minivision/C4BC2F49BC2F34F8/ssd_data/FaceRecognition/Test/O2N/Pad_TYLG_Foreigner_2020-06-05/id_life_ssd13_patches/le_re_0.4_112x112 \
                  --img_list_label_path /media/minivision/C4BC2F49BC2F34F8/ssd_data/FaceRecognition/Test/O2N/Pad_TYLG_Foreigner_2020-06-05/id_life_ssd13_result/id_life_image_list_bmppair.txt \
                  --data_source company

# resnet100
#python main.py    --mode prune \
#                  --model resnet100 \
#                  --best_model_path work_space/model_train_best/2019-09-29-11-37_SVGArcFace-O1-b0.4s40t1.1_fc_0.4_112x112_2019-09-27-Adult-padSY-Bus_fResNet100v3cv-d512_model_iter-340000.pth \
#                  --from_data_parallel \
#                  --fpgm \
#                  --save_model_pt \
#                  --test_root_path data/test_data/fc_0.4_112x112 \
#                  --img_list_label_path data/test_data/fc_0.4_112x112/pair_list/id_life_image_list_bmppair.txt \
#                  --data_source company


# ==============================敏感度分析脚本===========================================

#mobilefacenet_y2
#python main.py    --mode sa \
#                  --model mobilefacenet_y2 \
#                  --best_model_path /media/minivision/C4BC2F49BC2F34F8/1_Work/project/Face/z-prunning/compression_tool/work_space/model_train_best/2020-03-13-18-16_CombineMargin-zk-O1D1L-m0.9m0.4m0.15s64_le_re_0.4_112x112_2020-03-10-PadMaskGlassClean-Rest-NeighborGlass+TiktokGE5_MobileFaceNety2-d512_model_iter-146680_XCHoldClean-0.7226_PadMaskYTBYGlass-0.8612.pth \
#                  --from_data_parallel \
#                  --test_root_path /media/minivision/C4BC2F49BC2F34F8/ssd_data/Test/O2N/Pad_MaskBYYT_glass_2020-03-16/id_life_ssd12_patches/le_re_0.4_112x112 \
#                  --img_list_label_path /media/minivision/C4BC2F49BC2F34F8/ssd_data/Test/O2N/Pad_MaskBYYT_glass_2020-03-16/id_life_ssd12_result/id_life_image_list_bmppair.txt \
#	               --data_source company \
#                  --hrank \
#	               --rank_path ./work_space/rank_conv/mobilefacenety2_limit10/

#python main.py    --mode sa \
#                  --model mobilefacenet_y2 \
#                  --best_model_path /media/minivision/C4BC2F49BC2F34F8/1_Work/project/Face/z-prunning/compression_tool/work_space/model_train_best/2020-03-13-18-16_CombineMargin-zk-O1D1L-m0.9m0.4m0.15s64_le_re_0.4_112x112_2020-03-10-PadMaskGlassClean-Rest-NeighborGlass+TiktokGE5_MobileFaceNety2-d512_model_iter-146680_XCHoldClean-0.7226_PadMaskYTBYGlass-0.8612.pth \
#                  --from_data_parallel \
#                  --test_root_path /home/user1/linx/program/LightFaceNet/data/val_data/fc_0.4_112x112_zkx \
#                  --img_list_label_path /home/user1/linx/program/LightFaceNet/data/val_data/fc_0.4_112x112_zkx/pair_list/id_life_image_listpairjpg.txt

# resnet50
#python main.py    --mode sa \
#                  --model resnet50 \
#                  --best_model_path work_space/model_train_best/2019-09-29-05-31_SVGArcFace-O1-b0.4s40t1.1_fc_0.4_112x112_2019-09-27-Adult-padSY-Bus_fResNet50v3cv-d512_pytorch_iter_360000.pth \
#                  --from_data_parallel \
#                  --test_root_path data/test_data/fc_0.4_112x112 \
#                  --img_list_label_path data/test_data/fc_0.4_112x112/pair_list/id_life_image_list_bmppair.txt \
#                  --fpgm \
#                  --data_source company

# resnet100
#python main.py    --mode sa \
#                  --model resnet100 \
#                  --best_model_path work_space/model_train_best/2019-09-29-11-37_SVGArcFace-O1-b0.4s40t1.1_fc_0.4_112x112_2019-09-27-Adult-padSY-Bus_fResNet100v3cv-d512_model_iter-340000.pth \
#                  --from_data_parallel \
#                  --test_root_path data/test_data/fc_0.4_112x112 \
#                  --img_list_label_path data/test_data/fc_0.4_112x112/pair_list/id_life_image_list_bmppair.txt \
#                  --fpgm \
#                  --data_source company

# 对resnet-50 imagenet 稀疏度分析(fpgm)
#--mode
#sa
#--model
#resnet50_imagenet
#--best_model_path
#/home/linx/program/InsightFace_Pytorch/work_space/2020-08-20-10-32/models/model_accuracy:0.9708333333333334_step:163760_best_acc_lfw.pth
#--test_root_path
#/home/linx/dataset/face_recognition/ms1m_10k
#--fpgm
#--data_source
#lfw
#--test_batch_size
#64

# resnet-50 imagenet face recognition finetune(L1Rank)
python main.py  --mode finetune \
                --lr 0.01 \
                --epoch 20 \
                --train_data_path /home/linx/dataset/face_recognition/ms1m_10k/ \
                --pruned_checkpoint work_space/pruned_model/model_resnet50_imagenet.pt \
                --head_path /home/linx/program/z-prunning/compression_tool/work_space/head/head_2020-08-21-06-14_accuracy:0.9708333333333334_step:163760_best_acc_lfw.pth \
                --test_root_path /home/linx/dataset/face_recognition/ms1m_10k

python main.py    --mode prune \
                  --model resnet50_imagenet \
                  --best_model_path /home/linx/program/InsightFace_Pytorch/work_space/2020-08-20-10-32/models/model_accuracy:0.9708333333333334_step:163760_best_acc_lfw.pth \
                  --save_model_pt \
                  --test_root_path /home/linx/dataset/face_recognition/ms1m_10k \
                  --data_source lfw

# 重新训练一个mobilefacenet_y2模型
python main.py  --mode finetune \
                --lr 0.1 \
                --epoch 20 \
                --train_data_path /home/linx/dataset/face_recognition/ms1m_10k/ \
                --test_root_path /home/linx/dataset/face_recognition/ms1m_10k \
                --model mobilefacenet_y2

# 对mobilefacenet_y2稀疏度分析
python main.py    --mode sa \
                  --model mobilefacenet_y2 \
                  --best_model_path work_space/finetune/2020-09-03-10-06_mobilefacenet_y2_baseline/model/model_accuracy:0.9866666666666667_step:182160_best_acc_lfw.pth \
                  --test_root_path /home/linx/dataset/face_recognition/ms1m_10k \
                  --fpgm \
                  --data_source lfw

# 重新训练一个resnet_50(公司)模型
python main.py  --mode finetune \
                --lr 0.1 \
                --epoch 20 \
                --train_data_path /home/linx/dataset/face_recognition/ms1m_10k/ \
                --test_root_path /home/linx/dataset/face_recognition/ms1m_10k \
                --model resnet50

