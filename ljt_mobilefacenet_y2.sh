#
python main.py    --mode sa \
                  --model mobilefacenet_y2_ljt \
                  --best_model_path /home/linx/model/ljt/2020-08-23-08-09_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-ID-INTRA-MIDDLE-30-INTER-90-HARD_MobileFaceNety2-d512-k-9-8_model_iter-125993_TYLG-0.7520_PadMaskYTBYGlassM280-0.9104_BusIDPhoto-0.7489-noamp.pth \
                  --test_root_path /home/linx/dataset/company_test_data/BusID/le_re_0.4_144x122 \
                  --img_list_label_path /home/linx/dataset/company_test_data/BusID/id_life_image_list_bmppair.txt \
                  --test_batch_size 1