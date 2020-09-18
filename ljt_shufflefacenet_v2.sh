# BUSID
python main.py    --mode sa \
                  --model shufflefacenet_v2_ljt \
                  --best_model_path /home/yeluyue/lz/model/2020-09-15-10-53_CombineMargin-ljt914-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-MIDDLE-30_ShuffleFaceNetA-2.0-d512_model_iter-76608_TYLG-0.7319_XCHoldClean-0.8198_BusIDPhoto-0.7310-noamp.pth \
                  --test_root_path /home/yeluyue/lz/dataset/200914_data_model_ljt/BusID/le_re_0.4_144x122 \
                  --img_list_label_path /home/yeluyue/lz/dataset/200914_data_model_ljt/BusID/id_life_image_list_bmppair.txt \
                  --data_source company \
                  --fpgm

# TYLG
python main.py    --mode sa \
                  --model shufflefacenet_v2_ljt \
                  --best_model_path /home/yeluyue/lz/model/2020-09-15-10-53_CombineMargin-ljt914-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-MIDDLE-30_ShuffleFaceNetA-2.0-d512_model_iter-76608_TYLG-0.7319_XCHoldClean-0.8198_BusIDPhoto-0.7310-noamp.pth \
                  --test_root_path /home/yeluyue/lz/dataset/200914_data_model_ljt/TYLG/le_re_0.4_144x122 \
                  --img_list_label_path /home/yeluyue/lz/dataset/200914_data_model_ljt/TYLG/id_life_image_list_bmppair.txt \
                  --data_source company \
                  --fpgm

# XCH
python main.py    --mode sa \
                  --model shufflefacenet_v2_ljt \
                  --best_model_path /home/yeluyue/lz/model/2020-09-15-10-53_CombineMargin-ljt914-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-MIDDLE-30_ShuffleFaceNetA-2.0-d512_model_iter-76608_TYLG-0.7319_XCHoldClean-0.8198_BusIDPhoto-0.7310-noamp.pth \
                  --test_root_path /home/yeluyue/lz/dataset/200914_data_model_ljt/XCH/le_re_0.4_144x122 \
                  --img_list_label_path /home/yeluyue/lz/dataset/200914_data_model_ljt/XCH/id_life_image_list_bmppair.txt \
                  --data_source company \
                  --fpgm