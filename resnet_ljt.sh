# ================================ResNet-50==============================================================================

# TYLG
python main.py    --mode sa \
                  --model resnet_50_ljt \
                  --best_model_path /home/linx/model/ljt/2020-08-11-22-35_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-MIDDLE-30_fResNet50v3cv-d512_model_iter-76608_TYLG-0.8070_PadMaskYTBYGlassM280-0.9305_BusIDPhoto-0.6541.pth \
                  --test_root_path /home/linx/dataset/company_test_data/TYLG/le_re_0.4_144x122 \
                  --img_list_label_path /home/linx/dataset/company_test_data/TYLG/id_life_image_list_bmppair.txt \
                  --data_source company \
                  --fpgm \
                  --from_data_parallel





# ================================ResNet-100==============================================================================
python main.py    --mode sa \
                  --model resnet_100_ljt \
                  --best_model_path /home/linx/model/ljt/2020-06-27-12-59_CombineMargin-zk-O1D1Ls-m0.9m0.4m0.15s64_fc_0.4_144x122_2020-05-26-PNTMS-CLEAN-MIDDLE-70_fResNet100v3cv-d512_model_iter-96628_Idoa-0.8996_IdoaMask-0.9127_TYLG-0.9388.pth \
                  --test_root_path /media/linx/B0C6A127C6A0EF32/200914_data_model_ljt/TYLG/fc_0.4_144x122 \
                  --img_list_label_path /home/linx/dataset/company_test_data/TYLG/id_life_image_list_bmppair.txt \
                  --data_source company \
                  --fpgm \
                  --from_data_parallel

