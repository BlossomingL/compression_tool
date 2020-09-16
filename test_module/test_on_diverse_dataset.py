# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/12 下午2:02
from test_module.test_on_face_recognition import TestOnFaceRecognition
from test_module.test_on_face_classification import TestOnFaceClassification
from test_module.test_with_insight_face import TestWithInsightFace


def test(args, model):
    if args.model == 'mobilenetv3' or args.model == 'mobilefacenet_lzc' or args.model == 'resnet34_lzc' and args.data_source != 'lfw':
        test = TestOnFaceClassification(model, args.test_root_path, args.img_list_label_path)
        acc = test.test(args.test_batch_size)
        return acc
    elif args.model == 'resnet50_imagenet' or args.data_source == 'lfw':
        test = TestWithInsightFace(model)
        agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = test.get_val_data(args.test_root_path)
        acc_lfw = test.test(lfw, lfw_issame, args.test_batch_size, args.device)
        # acc_cfp = test.test(cfp_fp, cfp_fp_issame, args.test_batch_size, args.device)
        # acc_agedb = test.test(agedb_30, agedb_30_issame, args.test_batch_size, args.device)
        return acc_lfw
    else:
        test = TestOnFaceRecognition(model, args.test_root_path, args.img_list_label_path, args.data_source)
        accuracy = test.test(args.test_batch_size)
        return accuracy