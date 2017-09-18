#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os

default_num_threshs = 2000
default_threshs = np.linspace(0, 1, default_num_threshs, endpoint=False)

fname_pr = "pr_curve_%d_threshs.txt" % default_num_threshs
fname_roc = "roc_curve_%d_threshs.txt" % default_num_threshs
fname_balanced_pr = "balanced_pr_curve_%d_threshs.txt" % default_num_threshs

fname_pr_img = 'pr_curve_%d_threshs.png' % default_num_threshs
fname_roc_img = "roc_curve_%d_threshs.png" % default_num_threshs
fname_balanced_pr_img = 'balanced_pr_curve_%d_threshs.png' % default_num_threshs


def load_roc_txt(fname_roc):
    if not osp.exists(fname_roc):
        return None

    print "Load ROC from file " + fname_roc
    fp_roc = open(fname_roc, 'r')
    line = fp_roc.readline()  # skip the 1st line

    tp = []
    fn = []
    tn = []
    fp = []

    for line in fp_roc:
        spl = line.split()
        tp.append(float(spl[1]))
        fn.append(float(spl[2]))
        tn.append(float(spl[3]))
        fp.append(float(spl[4]))

    tp = np.array(tp)
    fn = np.array(fn)
    tn = np.array(tn)
    fp = np.array(fp)

    fp_roc.close()

    return tp, fn, tn, fp


def load_similarity_file(sim_file):
    sim_list = []
    with open(sim_file, 'r') as fp_sim:
        for line in fp_sim:
            img1_fn, img2_fn, score = line.split('\t')
            sim_list.append(float(score))
    fp_sim.close()

    return sim_list


def calc_roc(same_pairs_sim_list, diff_pairs_sim_list, threshs=None, save_dir='./'):
    if threshs is None:
        threshs = default_threshs

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    num_threshs = len(threshs)

    fn = np.zeros(num_threshs)
    tp = np.zeros(num_threshs)
    tn = np.zeros(num_threshs)
    fp = np.zeros(num_threshs)

    print "Processing same pairs"
    for score in same_pairs_sim_list:
        for i in range(num_threshs):
            if score < threshs[i]:
                fn[i] += 1
            else:
                tp[i] += 1

    print "Finished Processing same pairs"

    print "Processing diff paris"
    for score in diff_pairs_sim_list:
        for i in range(num_threshs):
            if score < threshs[i]:
                tn[i] += 1
            else:
                fp[i] += 1
    print "Finished Processing diff paris"

    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    fp_roc = open(osp.join(save_dir, fname_roc), "w")

    fp_roc.write("thresh\t tp\t fn\t tn\t fp\t tpr\t \tfpr\n")
    for i in range(num_threshs):
        fp_roc.write("%f\t %d\t %d\t %d\t %d\t %f\t %f\n" %
                     (threshs[i], tp[i], fn[i], tn[i], fp[i], recall[i], fpr[i]))

    fp_roc.close()

    return tp, fn, tn, fp


def load_file_and_calc_roc(same_pairs_result_file, diff_pairs_result_file,
                           threshs=None, save_dir='./'):
    if threshs is None:
        threshs = default_threshs

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    same_pairs_sim_list = load_similarity_file(same_pairs_result_file)
    diff_pairs_sim_list = load_similarity_file(diff_pairs_result_file)

    tp, fn, tn, fp = calc_roc(
        same_pairs_sim_list, diff_pairs_sim_list, threshs, save_dir)

    return tp, fn, tn, fp


def calc_presicion_recall(tp, fn, tn, fp, threshs=None, save_dir='./'):
    if threshs is None:
        threshs = default_threshs

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    num_threshs = len(threshs)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    fp_pr = open(osp.join(save_dir, fname_pr), "w")

    fp_pr.write("thresh\t presicion\t recall\n")
    for i in range(num_threshs):
        fp_pr.write("%f\t %f\t %f\n" %
                    (threshs[i], precision[i], recall[i]))

    fp_pr.close()

    balanced_precision = tp / (tp + fp / 3.1)

    fp_pr_bl = open(osp.join(save_dir, fname_balanced_pr), "w")

    fp_pr_bl.write("thresh\t presicion\t recall\n")
    for i in range(num_threshs):
        fp_pr_bl.write("%f\t %f\t %f\n" %
                       (threshs[i], balanced_precision[i], recall[i]))

    fp_pr_bl.close()


# def calculate_pr(same_pairs_result_file, diff_pairs_result_file):
#     print "Calc PR with %d threshs" % num_threshs

#     fn = np.zeros(num_threshs)
#     tp = np.zeros(num_threshs)
#     tn = np.zeros(num_threshs)
#     fp = np.zeros(num_threshs)

#     print "Processing same pairs"
#     with open(same_pairs_result_file, 'r') as fp_sim:
#         for line in fp_sim:
#             img1_fn, img2_fn, score = line.split('\t')
#             score = float(score)

#             for i in range(num_threshs):
#                 if score < threshs[i]:
#                     fn[i] += 1
#                 else:
#                     tp[i] += 1

#     print "Finished Processing same pairs"

#     print "Processing diff paris"
#     with open(diff_pairs_result_file, 'r') as fp_sim:
#         for line in fp_sim:
#             img1_fn, img2_fn, score = line.split('\t')
#             score = float(score)

#             for i in range(num_threshs):
#                 if score < threshs[i]:
#                     tn[i] += 1
#                 else:
#                     fp[i] += 1
#     print "Finished Processing diff paris"

#     recall = tp / (tp + fn)
#     precision = tp / (tp + fp)
#     fpr = fp / (tn + fp)

#     fp_pr = open(fname_pr, "w")
#     fp_roc = open(fname_roc, "w")

#     fp_pr.write("thresh\t presicion\t recall\n")
#     fp_roc.write("thresh\t tp\t fn\t tn\t fp\t tpr\t \tfpr\n")
#     for i in range(num_threshs):
#         fp_pr.write("%f\t %f\t %f\n" %
#                     (threshs[i], precision[i], recall[i]))
#         fp_roc.write("%f\t %d\t %d\t %d\t %d\t %f\t %f\n" %
#                      (threshs[i], tp[i], fn[i], tn[i], fp[i], recall[i], fpr[i]))

#     fp_pr.close()
#     fp_roc.close()

#     balanced_precision = tp / (tp + fp / 3.1)

#     fp_pr_bl = open(fname_balanced_pr, "w")

#     fp_pr_bl.write("thresh\t presicion\t recall\n")
#     for i in range(len(threshs=None)):
#         fp_pr_bl.write("%f\t %f\t %f\n" %
#                        (threshs[i], balanced_precision[i], recall[i]))

#     fp_pr_bl.close()

#     return tp, fn, tn, fp


def calc_roc_and_pr(same_pairs_result_file, diff_pairs_result_file, threshs=None, save_dir='./'):
    if threshs is None:
        threshs = default_threshs

    tp, fn, tn, fp = load_file_and_calc_roc(
        same_pairs_result_file, diff_pairs_result_file, threshs, save_dir)

    calc_presicion_recall(tp, fn, tn, fp, threshs, save_dir)

    return tp, fn, tn, fp


def draw_analysis_figure(tp, fn, tn, fp, save_dir='./', draw_balanced_pr=False):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    print "Draw PR curve"
    plt.figure(1)
    plt.plot(recall, precision)

    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('Precision vs. Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(color='r', linestyle='--', linewidth=1)
    plt.show()
    plt.savefig(osp.join(save_dir, fname_pr_img))

    fpr = fp / (fp + tn)

    print "Draw ROC curve"
    plt.figure(2)
    plt.plot(fpr, recall)

    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('TPR(recall) vs. FPR Curve')
    plt.xlabel('FPR')
    plt.ylabel('Recall')
    plt.grid(which='both', color='r', linestyle='--', linewidth=1)
    plt.show()
    plt.savefig(osp.join(save_dir, fname_roc_img))

	if draw_balanced_pr:
		balanced_precision = tp / (tp + fp / 3.1)

		#    fp_pr = open(fname_balanced_pr, "w")
		#
		#    fp_pr.write("thresh\t presicion\t recall\n")
		#    for i in range(num_threshs):
		#        fp_pr.write("%f\t %f\t %f\n" % (threshs[i], balanced_precision[i], recall[i]))
		#
		#    fp_pr.close()

		#    fp_roc = open(fname_roc, "w")
		#
		#    fp_roc.write("thresh\t tp\t fn\t tn\t fp\t tpr\t \tfpr\n")
		#    for i in range(num_threshs):
		#        fp_roc.write("%f\t %d\t %d\t %d\t %d\t %f\t %f\n" % (threshs[i], tp[i], fn[i], tn[i], fp[i], recall[i], fpr[i]))
		#
		#    fp_roc.close()

		print "Draw balanced PR curve"
		plt.figure(3)
		plt.plot(recall, balanced_precision)

		plt.xticks(np.arange(0, 1.1, 0.1))
		plt.yticks(np.arange(0, 1.1, 0.1))
		plt.title('Balanced-Precision vs. Recall Curve')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.grid(which='both', color='r', linestyle='--', linewidth=1)
		plt.show()

		plt.savefig(osp.join(save_dir, fname_balanced_pr_img))


if __name__ == "__main__":

    same_pairs_result_file = "samefacesim.txt"
    diff_pairs_result_file = "diffacesim.txt"
    save_dir = './'

    roc_file = osp.join(save_dir, fname_roc)
    if osp.exists(roc_file):
        tp, fn, tn, fp = load_roc_txt(roc_file)
    else:
        tp, fn, tn, fp = calc_roc_and_pr(
            same_pairs_result_file, diff_pairs_result_file, save_dir=save_dir)

    draw_analysis_figure(tp, fn, tn, fp, save_dir=save_dir)
