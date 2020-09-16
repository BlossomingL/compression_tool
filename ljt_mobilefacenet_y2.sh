# BUSID
python main.py    --mode sa \
                  --model mobilefacenet_y2_ljt \
                  --best_model_path /home/yeluyue/lz/model/2020-08-23-08-09_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-ID-INTRA-MIDDLE-30-INTER-90-HARD_MobileFaceNety2-d512-k-9-8_model_iter-125993_TYLG-0.7520_PadMaskYTBYGlassM280-0.9104_BusIDPhoto-0.7489-noamp.pth \
                  --test_root_path /home/yeluyue/lz/dataset/200914_data_model_ljt/BusID/le_re_0.4_144x122 \
                  --img_list_label_path /home/yeluyue/lz/dataset/200914_data_model_ljt/BusID/id_life_image_list_bmppair.txt \
                  --data_source company \
                  --fpgm

python main.py    --mode prune \
                  --model mobilefacenet_y2_ljt \
                  --save_model_pt \
                  --data_source company \
                  --fpgm \
                  --best_model_path /home/yeluyue/lz/model/2020-08-23-08-09_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-ID-INTRA-MIDDLE-30-INTER-90-HARD_MobileFaceNety2-d512-k-9-8_model_iter-125993_TYLG-0.7520_PadMaskYTBYGlassM280-0.9104_BusIDPhoto-0.7489-noamp.pth \
                  --test_root_path /home/yeluyue/lz/dataset/200914_data_model_ljt/BusID/le_re_0.4_144x122 \
                  --img_list_label_path /home/yeluyue/lz/dataset/200914_data_model_ljt/BusID/id_life_image_list_bmppair.txt \


# TYLG
python main.py    --mode sa \
                  --model mobilefacenet_y2_ljt \
                  --best_model_path /home/yeluyue/lz/model/2020-08-23-08-09_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-ID-INTRA-MIDDLE-30-INTER-90-HARD_MobileFaceNety2-d512-k-9-8_model_iter-125993_TYLG-0.7520_PadMaskYTBYGlassM280-0.9104_BusIDPhoto-0.7489-noamp.pth \
                  --test_root_path /home/yeluyue/lz/dataset/200914_data_model_ljt/TYLG/le_re_0.4_144x122 \
                  --img_list_label_path /home/yeluyue/lz/dataset/200914_data_model_ljt/TYLG/id_life_image_list_bmppair.txt \
                  --data_source company \
                  --fpgm

python main.py    --mode prune \
                  --model mobilefacenet_y2_ljt \
                  --save_model_pt \
                  --data_source company \
                  --fpgm \
                  --best_model_path /home/yeluyue/lz/model/2020-08-23-08-09_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-ID-INTRA-MIDDLE-30-INTER-90-HARD_MobileFaceNety2-d512-k-9-8_model_iter-125993_TYLG-0.7520_PadMaskYTBYGlassM280-0.9104_BusIDPhoto-0.7489-noamp.pth \
                  --test_root_path /home/yeluyue/lz/dataset/200914_data_model_ljt/TYLG/le_re_0.4_144x122 \
                  --img_list_label_path /home/yeluyue/lz/dataset/200914_data_model_ljt/TYLG/id_life_image_list_bmppair.txt \

# XCH
python main.py    --mode sa \
                  --model mobilefacenet_y2_ljt \
                  --best_model_path /home/yeluyue/lz/model/2020-08-23-08-09_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-ID-INTRA-MIDDLE-30-INTER-90-HARD_MobileFaceNety2-d512-k-9-8_model_iter-125993_TYLG-0.7520_PadMaskYTBYGlassM280-0.9104_BusIDPhoto-0.7489-noamp.pth \
                  --test_root_path /home/yeluyue/lz/dataset/200914_data_model_ljt/XCH/le_re_0.4_144x122 \
                  --img_list_label_path /home/yeluyue/lz/dataset/200914_data_model_ljt/XCH/id_life_image_list_bmppair.txt \
                  --data_source company \
                  --fpgm