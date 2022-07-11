import numpy as np

def topKaccuracy(y_out, y, k):
    '''
        The following three codes are designed for TopL accuracy, a widely used metirc in contact map
        y_out: predict matrix, L*L: ndarray
        y: label matirx, L*L: ndarray
    '''
    L = y.shape[0]

    m = np.ones_like(y, dtype=np.int8)
    lm = np.triu(m, 24)
    mm = np.triu(m, 12)
    sm = np.triu(m, 6)

    sm = sm - mm
    mm = mm - lm

    avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((avg_pred[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
    for x in [lm, mm, sm]:
        selected_truth = truth[x.nonzero()]
        selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort()[::-1]]
        tops_num = min(selected_truth_sorted.shape[0], L / k)
        truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
        corrects_num = np.bincount(truth_in_pred[0: int(tops_num)], minlength=2)
        acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
        accs.append(acc)

    return accs


def evaluate(predict_matrix, contact_matrix):
    '''
        calculation for TopL, TopL/2, TopL/5, TopL/10 accuracy
    '''
    acc_k_1 = topKaccuracy(predict_matrix, contact_matrix, 1)
    acc_k_2 = topKaccuracy(predict_matrix, contact_matrix, 2)
    acc_k_5 = topKaccuracy(predict_matrix, contact_matrix, 5)
    acc_k_10 = topKaccuracy(predict_matrix, contact_matrix, 10)
    tmp = []
    tmp.append(acc_k_1)
    tmp.append(acc_k_2)
    tmp.append(acc_k_5)
    tmp.append(acc_k_10)
    return tmp

def output_result(avg_acc):
    '''
        just print out the TopL accuracy results
    '''
    print ("Long Range(> 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][0], avg_acc[2][0], avg_acc[1][0], avg_acc[0][0]))
    print ("Medium Range(12 - 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][1], avg_acc[2][1], avg_acc[1][1], avg_acc[0][1]))
    print ("Short Range(6 - 12):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][2], avg_acc[2][2], avg_acc[1][2], avg_acc[0][2]))