import numpy as np


def get_score_for_one_patient(labels, predicts, threshold=0.5):
    '''
    计算一个病人的dice、IOU分数
    :param truths: numpy.array, [189, 224, 176, 1]
    :param predicts: numpy.array, [189, 224, 176, 1]
    :param threshold: threshold for computing dice score
    :return: dice score of this patient
    '''
    if labels.shape[0] != 189 or predicts.shape[0] != 189:
        print('ERROR: 每个病人的切片数量应当是189！')
        return 0
    label_positive = labels > threshold
    lp_count = np.count_nonzero(label_positive)
    predict_positive = predicts > threshold
    pp_count = np.count_nonzero(predict_positive)

    TP_count = np.count_nonzero(np.logical_and(label_positive, predict_positive))
    FN_count = lp_count - TP_count
    FP_count = pp_count - TP_count

    dice_score = 2 * TP_count / (lp_count + pp_count) if lp_count + pp_count != 0 else 0
    iou_score = TP_count / (lp_count + pp_count - TP_count) if lp_count + pp_count - TP_count != 0 else 0
    precision = TP_count / (TP_count + FP_count) if FP_count + TP_count != 0 else 0
    recall = TP_count / (TP_count + FN_count) if TP_count + FN_count != 0 else 0
    f1_score = 2 * TP_count / (2 * TP_count + FN_count + FP_count) if 2 * TP_count + FN_count + FP_count != 0 else 0
    voe = 2 * (pp_count - lp_count) / (pp_count + lp_count) if pp_count + lp_count != 0 else 0
    rvd = pp_count / lp_count - 1 if lp_count != 0 else -1

    print('Label positive:', lp_count,
          '\t Predict positive:', pp_count,
          '\t TP:', TP_count,
          '\t FN:', FN_count,
          '\t FP:', FP_count,
          '\t Dice score:', dice_score,
          '\t IOU score:', iou_score,
          '\t Precision:', precision,
          '\t Recall:', recall,
          '\t F1 score:', f1_score,
          '\t VOE score:', voe,
          '\t RVD score:', rvd)
    return dice_score, iou_score, precision, recall, f1_score, voe, rvd


def get_score_from_all_slices(labels, predicts, threshold=0.5):
    '''
    输入2维切片，计算每一个病人的3维的分数，返回按照病人计算的平均评价指标。n为切片数量，且须有n%189==0
    :param truths: np.array, [n, 224, 176, 1]
    :param predicts: np.array, [n, 224, 176, 1]
    :param threshold: threshold for computing dice
    :return: a dice scores
    '''
    if labels.shape[0] % 189 != 0 or predicts.shape[0] % 189 != 0:
        print('ERROR: 输入切片数量应当是189的整数倍！')
        return np.array([])
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    voe_scores = []
    rvd_scores = []
    for i in range(labels.shape[0] // 189):
        tmp_labels = labels[i*189 : (i+1)*189]
        tmp_pred = predicts[i*189 : (i+1)*189]
        tmp_dice, tmp_iou, tmp_precision,\
            tmp_recall, tmp_f1, tmp_voe,\
            tmp_rvd = get_score_for_one_patient(labels=tmp_labels, predicts=tmp_pred, threshold=threshold)
        dice_scores.append(tmp_dice)
        iou_scores.append(tmp_iou)
        precision_scores.append(tmp_precision)
        recall_scores.append(tmp_recall)
        f1_scores.append(tmp_f1)
        voe_scores.append(tmp_voe)
        rvd_scores.append(tmp_rvd)

    scores = {}
    scores['dice'] = dice_scores
    scores['iou'] = iou_scores
    scores['precision'] = precision_scores
    scores['recall'] = recall_scores

    return scores


if __name__ == '__main__':
    import pandas as pd
    num_fold = 4
    for fold in range(num_fold):
        csv_path = 'fold_' + str(fold) + '/score_record.csv'
        df = pd.read_csv(csv_path)
        print('In fold', fold)
        for key in df.keys():
            print(key, ' = ', np.mean(df[key]))
