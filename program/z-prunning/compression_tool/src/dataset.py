'''
save Training dataset and TestSet
'''
#from config_path import ConfigDataPath

# test_root_path = ConfigDataPath.test_root_path
# list_root_path = ConfigDataPath.list_root_path
# image_root_path = ConfigDataPath.image_root_path


# --------------------Training  Config ---------------------------
# list_root_path = "/gpfs01/data/recognize_data/FaceRecognition/Train/O2N_Patches/Patches_mtcnn_95to5_list"
# image_root_path = "/gpfs01/data/recognize_data/FaceRecognition/Train/O2N_Patches/Patches_mtcnn_95to5"

list_root_path = "/ssd/data/recognize_data/FaceRecognition/Train/O2N_Patches/Patches_mtcnn_95to5_list"
image_root_path = "/ssd/data/recognize_data/FaceRecognition/Train/O2N_Patches/Patches_mtcnn_95to5"
test_root_path = "/gpfs10/data/recognize_data/FaceRecognition/Test/O2N"

TrainSet = {

    # New SSD Patch
    "2020-02-14-Pad-Rest": {"root_path": image_root_path,
                            "label_list": "{}/combine_folder_list/2020-02-14_PadAll+Rest_list_label.txt".format(
                                list_root_path),
                            },


    "2020-02-19-PadMask-Rest": {"root_path": image_root_path,
                                "label_list": "{}/combine_folder_list/2020-02-19_PadAll-Mask+Rest_list_label.txt".format(
                                    list_root_path),
                                },
    "2020-02-19-PadMask": {"root_path": image_root_path,
                           "label_list": "{}/combine_folder_list/2020-02-19_PadAll-Mask_list_label.txt".format(
                               list_root_path),
                           },
    "2020-02-19-Pad": {"root_path": image_root_path,
                       "label_list": "{}/combine_folder_list/2020-02-19_PadAll_list_label.txt".format(list_root_path),
                       },

    "2020-02-20-PadMask-Rest-VideoCap": {"root_path": image_root_path,
                                         "label_list": "{}/combine_folder_list/2020-02-20_PadAll-Mask+Rest+VideoCapb1Clean_list_label.txt".format(list_root_path),
                                         },

    "2020-02-24-PadMask-Rest-VideoCapMask": {"root_path": image_root_path,
                                             "label_list": "{}/combine_folder_list/2020-02-24_PadAll-Mask+Rest+VideoCapb1CleanMask_list_label.txt".format(list_root_path),
                                             },

    "2020-02-27-PadMask-Rest-Neighbor+TiktokGE5": {"root_path": image_root_path,
                                                   "label_list": "{}/combine_folder_list/2020-02-27_PadAll-Mask+Rest+PadNeighborMask20200225-GE5+Tiktok-GE5_list_label.txt".format(list_root_path),
                                                   },

    "2020-03-02-PadMask-Rest-Neighbor+TiktokGE5": {"root_path": image_root_path,
                                                   "label_list": "{}/combine_folder_list/2020-03-02_PadAll-Mask+Rest+PadNeighborMask20200320-GE5+Tiktok-GE5_list_label.txt".format(list_root_path),
                                                   },
    "2020-03-04-PadMaskGlassClean-Rest-Neighbor+TiktokGE5": {"root_path": image_root_path,
                                                             "label_list": "{}/combine_folder_list/2020-03-03_PadAll-Mask-glassClean+Rest+PadNeighborMask20200320-GE5+Tiktok-GE5_list_label.txt".format(list_root_path),
                                                             },

    "2020-03-06-PadMaskGlassClean-Rest-NeighborGlass+TiktokGE5": {"root_path": image_root_path,
                                                                  "label_list": "{}/combine_folder_list/2020-03-06_PadAll-Mask-glassClean+Rest+PadNeighborMask20200306_glass-GE5+Tiktok-GE5_list_label.txt".format(list_root_path),
                                                                  },

    "2020-03-10-PadMaskGlassClean-Rest-NeighborGlass+TiktokGE5": {"root_path": image_root_path,
                                                                  "label_list": "{}/combine_folder_list/2020-03-10_PadAll-Mask-glassClean+Rest+PadNeighborMask20200310_glass-GE5+Tiktok-GE5_list_label.txt".format(
                                                                      list_root_path),
                                                                  },
    # Triplet

    "2020-02-17-PadNeighborMask": {"root_path": image_root_path,
                                   "label_list": "{}/single_folder_list/GE_1/PadNeighborMask_GE_1_list_label.txt".format(list_root_path)},

    "2020-02-24-PadNeighborMask": {"root_path": image_root_path,
                                   "label_list": "{}/combine_folder_list/2020-02-24_PadNeighborMask_GE_1_list_label.txt".format(list_root_path),
                                   },

    "2020-02-25-PadMask-Neighbor": {"root_path": image_root_path,
                                    "label_list": "{}/combine_folder_list/2020-02-25_PadAll-Mask+PadNeighborMask0225_list_label.txt".format(list_root_path),
                                    },

    "2020-02-25-PadMask-Rest-Neighbor-Tiktok": {"root_path": image_root_path,
                                                "label_list": "{}/combine_folder_list/2020-02-25_PadAll-Mask+Rest+PadNeighborMask20200225+Tiktok_list_label.txt".format(list_root_path),
                                                },

    "2020-03-06-PadMaskGlass-NeighborGlass-Tiktok": {"root_path": image_root_path,
                                                     "label_list": "{}/combine_folder_list/2020-03-06_PadAll-Mask-glass+PadNeighborMask-glass0306GE1+TiktokGE1_list_label.txt".format(list_root_path),
                                                     },

    "2020-03-09-PadMaskGlass-NeighborGlass-Tiktok": {"root_path": image_root_path,
                                                     "label_list": "{}/combine_folder_list/2020-03-09_PadAll-Mask-glass+PadNeighborMask-glass0306GE1+TiktokGE1_list_label.txt".format(list_root_path),
                                                     },

    "2020-03-09-PadNeighborGlass-Tiktok": {"root_path": image_root_path,
                                           "label_list": "{}/combine_folder_list/2020-03-09_PadNeighborMask-glass0306GE1+TiktokGE1_list_label.txt".format(list_root_path),
                                           },

    "2020-03-10-PadMaskGlass-NeighborGlass-Tiktok": {"root_path": image_root_path,
                                                     "label_list": "{}/combine_folder_list/2020-03-10_PadAll-Mask-glass+PadNeighborMask-glass0310GE1+TiktokGE1_list_label.txt".format(
                                                         list_root_path),
                                                     },

    "2020-03-10-PadNeighborGlass-Tiktok": {"root_path": image_root_path,
                                           "label_list": "{}/combine_folder_list/2020-03-10_PadNeighborMask-glass0310GE1+TiktokGE1_list_label.txt".format(
                                               list_root_path),
                                           },
}

# --------------------Training Test Config ------------------------
TestSet = {

    # ==================================更新清理之后的测试集===========================================================================
    # Pad 成人+老年人场景
    "XCHoldClean-ssd-19-04-20": {
        "root_path": "{}/XCH_Pad_2019-04-20_mergeold_ssd_clean/id_life_ssd3_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2019-04-20_mergeold_ssd_clean/id_life_ssd3_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/XCH_Pad_2019-04-20_mergeold_ssd_clean/id_life_ssd3_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },

    "XCHoldClean-ssd-19-04-20-12": {
        "root_path": "{}/XCH_Pad_2019-04-20_mergeold_ssd_clean/id_life_ssd12_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2019-04-20_mergeold_ssd_clean/id_life_ssd12_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/XCH_Pad_2019-04-20_mergeold_ssd_clean/id_life_ssd12_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    # Pad 成人+老年人场景 + 小学生场景
    "XCHoldstuClean-ssd": {
        "root_path": "{}/XCH_Pad_2019-04-20_mergeold-studnet_clean/id_life_ssd3_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2019-04-20_mergeold-studnet_clean/id_life_ssd3_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/XCH_Pad_2019-04-20_mergeold-studnet_clean/id_life_ssd3_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },

    # Pad 成人+老年人场景 + 小学生场景 + 幼儿场景
    "Full-ssd-20-01-02": {
        "root_path": "{}/XCH_Pad_2020-01-02_stu_child_clean/id_life_ssd3_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2020-01-02_stu_child_clean/id_life_ssd3_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/XCH_Pad_2020-01-02_stu_child_clean/id_life_ssd3_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },

    # 减少Pad 人数 一半
    "Full-balance-a-ssd-20-01-02": {
        "root_path": "{}/XCH-half_Pad_2020-01-02_stu_child_clean/id_life_ssd3_patches".format(test_root_path),
        "image_list": "{}/XCH-half_Pad_2020-01-02_stu_child_clean/id_life_ssd3_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/XCH-half_Pad_2020-01-02_stu_child_clean/id_life_ssd3_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    # Pad 幼儿场景
    "ChildClean-ssd": {
        "root_path": "{}/Kingdergarten_Child-2_clean/id_life_ssd3_patches".format(test_root_path),
        "image_list": "{}/Kingdergarten_Child-2_clean/id_life_ssd3_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/Kingdergarten_Child-2_clean/id_life_ssd3_result/id_life_image_list_bmppair.txt".format(test_root_path)
    },

    # Pad 幼儿场景 + 小学生场景
    "ChildStuClean-ssd": {
        "root_path": "{}/XCH_Pad_2020-01-06_Kids_stu_child/id_life_ssd3_patches".format(test_root_path),
        "image_list": "{}XCH_Pad_2020-01-06_Kids_stu_child/id_life_ssd3_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/XCH_Pad_2020-01-06_Kids_stu_child/id_life_ssd3_result/id_life_image_list_bmppair.txt".format(
            test_root_path)
    },

    # Pad 小学生
    "StuClean-ssd": {
        "root_path": "{}/Pad_2019-10-12_student_testset_clean/id_life_ssd3_patches".format(test_root_path),
        "image_list": "{}/Pad_2019-10-12_student_testset_clean/id_life_ssd3_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/Pad_2019-10-12_student_testset_clean/id_life_ssd3_result/id_life_image_list_bmppair.txt".format(
            test_root_path)
    },

    # 商圈 底库抓拍照场景
    "BushardcapClean-ssd-95to5": {
        "root_path": "{}/Business_07-22_mtcnn_capture_ssd_clean/id_life_ssd3_patches".format(test_root_path),
        "image_list": "{}/Business_07-22_mtcnn_capture_ssd_clean/id_life_ssd3_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/Business_07-22_mtcnn_capture_ssd_clean/id_life_ssd3_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },
    # Pad 成人+老年人场景:id-351 life-4930
    "XCHP": {
        "root_path": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2019-04-20_mergeold_ssd_clean_part/id_life_ssd4_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2019-04-20_mergeold_ssd_clean_part/id_life_ssd4_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2019-04-20_mergeold_ssd_clean_part/id_life_ssd4_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },
    # 太原理工外国人
    "TYLG": {
        "root_path": "{}/XCH_Pad_2020-01-21_oversea/TYLG_Foreigner/id_life_ssd4_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2020-01-21_oversea/TYLG_Foreigner/id_life_ssd4_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/XCH_Pad_2020-01-21_oversea/TYLG_Foreigner/id_life_ssd4_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },
    # LFW:id-444,life-3842(名字去掉下划线)
    "LfwP": {
        "root_path": "{}/XCH_Pad_2020-01-21_oversea/LFW_merge/id_life_ssd4_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2020-01-21_oversea/LFW_merge/id_life_ssd4_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/XCH_Pad_2020-01-21_oversea/LFW_merge/id_life_ssd4_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },
    # 印度尼西亚
    "Idoa": {
        "root_path": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2020-01-21_Indonesia/id_life_ssd4_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2020-01-21_Indonesia/id_life_ssd4_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2020-01-21_Indonesia/id_life_ssd4_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },
    # 亚裔+印尼裔+外国人  XCHP TYLG LfwP Idoa
    "Oversea": {
        "root_path": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2020-01-21_Full/id_life_ssd4_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2020-01-21_Full/id_life_ssd4_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/XCH_Pad_2020-01-21_oversea/XCH_Pad_2020-01-21_Full/id_life_ssd4_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },
    # 口罩测试集
    "PadMask": {
        "root_path": "{}/Pad_Mask_2020-02-07/id_life_ssd7_patches".format(test_root_path),
        "image_list": "{}/Pad_Mask_2020-02-07/id_life_ssd7_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/Pad_Mask_2020-02-07/id_life_ssd7_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },

    # N95口罩测试集
    "PadMaskN95": {
        "root_path": "{}/Pad_Maskn95_2020-02-09/id_life_ssd7_patches".format(test_root_path),
        "image_list": "{}/Pad_Maskn95_2020-02-09/id_life_ssd7_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/Pad_Maskn95_2020-02-09/id_life_ssd7_result/id_life_image_list_bmppair.txt".format(test_root_path),
    },

    "PadMaskN95T2": {
        "root_path": "{}/Pad_Maskn95_2020-02-09/id_life_ssd9_patches".format(test_root_path),
        "image_list": "{}/Pad_Maskn95_2020-02-09/id_life_ssd9_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/Pad_Maskn95_2020-02-09/id_life_ssd9_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },
    "PadMaskMix": {
        "root_path": "{}/Pad_Maskn95+Collect_2020-02-11/id_life_ssd9_patches".format(test_root_path),
        "image_list": "{}/Pad_Maskn95+Collect_2020-02-11/id_life_ssd9_result/id_life_image_list_bmp.txt".format(test_root_path),
        "image_pairs": "{}/Pad_Maskn95+Collect_2020-02-11/id_life_ssd9_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    "PadMaskMix10": {
        "root_path": "{}/Pad_Maskn95+Collect_2020-02-11/id_life_ssd10_patches".format(test_root_path),
        "image_list": "{}/Pad_Maskn95+Collect_2020-02-11/id_life_ssd10_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/Pad_Maskn95+Collect_2020-02-11/id_life_ssd10_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    "PadMaskYTHY": {
        "root_path": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd10_patches".format(test_root_path),
        "image_list": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd10_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd10_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    "PadMaskYTHY-11": {
        "root_path": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd11_patches".format(test_root_path),
        "image_list": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd11_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd11_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },
    "PadMaskYTHY-12": {
        "root_path": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd12_patches".format(test_root_path),
        "image_list": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd12_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/Pad_MaskCollectYTHY_2020-02-16/id_life_ssd12_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    "PadMaskYTBY-clean": {
        "root_path": "{}/Pad_MaskBYQHYTHY_Clean_2020-02-28/id_life_ssd12_patches".format(test_root_path),
        "image_list": "{}/Pad_MaskBYQHYTHY_Clean_2020-02-28/id_life_ssd12_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/Pad_MaskBYQHYTHY_Clean_2020-02-28/id_life_ssd12_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    "PadMaskYTBY-glass": {
        "root_path": "{}/Pad_MaskBYYT_glass_2020-03-05/id_life_ssd12_patches".format(test_root_path),
        "image_list": "{}/Pad_MaskBYYT_glass_2020-03-05/id_life_ssd12_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/Pad_MaskBYYT_glass_2020-03-05/id_life_ssd12_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    "PadMaskYTBYGlass": {
        "root_path": "{}/Pad_MaskBYYT_glass_2020-03-16/id_life_ssd12_patches".format(test_root_path),
        "image_list": "{}/Pad_MaskBYYT_glass_2020-03-16/id_life_ssd12_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/Pad_MaskBYYT_glass_2020-03-16/id_life_ssd12_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    "XCHGlassAll": {
        "root_path": "{}/XCH_Pad-2020-03-04_glassall/id_life_ssd12_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad-2020-03-04_glassall/id_life_ssd12_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/XCH_Pad-2020-03-04_glassall/id_life_ssd12_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },
    "XCHGlassPair": {
        "root_path": "{}/XCH_Pad-2020-03-05_glasspair/id_life_ssd12_patches".format(test_root_path),
        "image_list": "{}/XCH_Pad-2020-03-05_glasspair/id_life_ssd12_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/XCH_Pad-2020-03-05_glasspair/id_life_ssd12_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },

    "ChildMask":{
        "root_path": "{}/ChildMask_2020-03-10/id_life_ssd12_patches".format(test_root_path),
        "image_list": "{}/ChildMask_2020-03-10/id_life_ssd12_result/id_life_image_list_bmp.txt".format(
            test_root_path),
        "image_pairs": "{}/ChildMask_2020-03-10/id_life_ssd12_result/id_life_image_list_bmppair.txt".format(
            test_root_path),
    },
#tmp=========================================================================================


}
