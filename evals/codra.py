"""Use the same evaluation procedure Evaluate the output of CODRA

"""

from __future__ import absolute_import, print_function

from collections import defaultdict
import itertools

import numpy as np

from educe.rst_dt.codra import load_codra_output_files
from educe.rst_dt.dep2con import deptree_to_rst_tree
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.document_plus import align_edus_with_paragraphs
#
from attelo.io import load_edus
from attelo.metrics.constituency import (parseval_detailed_report,
                                         parseval_report)
from attelo.metrics.deptree import compute_uas_las


def eval_codra_output(ctree_true, dtree_true,
                      codra_out_dir, edus_file,
                      rel_conv=None,
                      nary_enc='chain',
                      nuc_clf=None, rnk_clf=None,
                      detailed=False):
    """Load and evaluate the .dis files output by CODRA.

    This currently runs on the document-level files (.doc_dis).
    """
    # WIP 2016-06-29 sent_idx
    att_edus = load_edus(edus_file)
    edu2sent_idx = defaultdict(dict)
    for att_edu in att_edus:
        doc_name = att_edu.grouping
        edu_num = int(att_edu.id.rsplit('_', 1)[1])
        sent_idx = int(att_edu.subgrouping.split('_sent')[1])
        edu2sent_idx[doc_name][edu_num] = sent_idx
    # sort EDUs by num
    # rebuild educe-style edu2sent ; prepend 0 for the fake root
    doc_name2edu2sent = {doc_name: ([0]
                                    + [s_idx for e_num, s_idx
                                       in sorted(edu2sent.items())])
                         for doc_name, edu2sent in edu2sent_idx.items()}
    doc_name2edu2para = dict()

    for doc_name, rtree_true in sorted(ctree_true.items()):
        # WIP 2016-06-29 para_idx
        doc_edus = rtree_true.leaves()
        doc_txt = doc_edus[0].context._text
        # retrieve paragraph idx
        doc_paras = doc_edus[0].context.paragraphs
        if doc_paras is not None:
            edu2para = align_edus_with_paragraphs(
                doc_edus, doc_paras, doc_txt)
            # yerk: interpolate values in edu2para where missing
            edu2para_fix = []
            for edu_idx in edu2para:
                if edu_idx is not None:
                    edu2para_fix.append(edu_idx)
                else:
                    # interpolation strategy: copy the last regular value
                    # that has been seen
                    edu2para_fix.append(edu2para_fix[-1])
            edu2para = edu2para_fix
            # end yerk: interpolate
            edu2para = [0] + list(np.array(edu2para) + 1)
            doc_name2edu2para[doc_name] = edu2para
        else:
            doc_name2edu2para[doc_name] = None
        # end retrieve paragraph idx

    # load predicted trees
    data_pred = load_codra_output_files(codra_out_dir)
    # filenames = data_pred['filenames']
    doc_names_pred = data_pred['doc_names']
    rst_ctrees_pred = data_pred['rst_ctrees']

    # gather predictions
    dtree_pred = dict()  # dependency trees
    ctree_pred = dict()  # constituency trees

    for doc_name, ct_pred in itertools.izip(doc_names_pred, rst_ctrees_pred):
        # constituency tree
        # replace fine-grained labels with coarse-grained labels
        # no need to replace labels: the files we have already contain
        # the coarse labels
        if rel_conv is not None:
            ct_pred = rel_conv(ct_pred)
        ctree_pred[doc_name] = ct_pred
        # convert to weakly-ordered dependency tree
        dt_pred = RstDepTree.from_rst_tree(ct_pred, nary_enc='chain')
        dtree_pred[doc_name] = dt_pred

    # compare pred and true
    common_doc_names = set(dtree_true.keys()) & set(dtree_pred.keys())

    # dep scores
    dtree_true_list = [dt for doc_name, dt in sorted(dtree_true.items())
                       if doc_name in common_doc_names]
    dtree_pred_list = [dt for doc_name, dt in sorted(dtree_pred.items())
                       if doc_name in common_doc_names]

    score_uas, score_las, score_ls = compute_uas_las(dtree_true_list,
                                                     dtree_pred_list)
    print('UAS / LAS / LS : {:.4f} / {:.4f} / {:.4f}'.format(
        score_uas, score_las, score_ls))

    skipped_docs = set()
    # convert dicts to aligned lists of SimpleRSTTrees, skipping docs where
    # needed
    ctree_true = [ct for doc_name, ct in sorted(ctree_true.items())
                  if doc_name not in skipped_docs]
    ctree_pred = [ct for doc_name, ct in sorted(ctree_pred.items())
                  if doc_name not in skipped_docs]
    # compute and print PARSEVAL scores
    print(parseval_report(ctree_true, ctree_pred, digits=4))
    # detailed report on S+N+R
    if detailed:
        print(parseval_detailed_report(ctree_true, ctree_pred,
                                       metric_type='S+R'))

    if nuc_clf is not None and rnk_clf is not None:
        # WIP 2016-06-29 use our deterministic classifiers for nuc and rank
        # => estimate degradation on Joty's output => hint at ours
        # nuclearity
        # rebuild ctrees
        ctree_pred2 = dict()
        for doc_name, dt_pred in sorted(dtree_pred.items()):
            # set nuclearity
            dt_pred.nucs = nuc_clf.predict([dt_pred])[0]
            # set ranking, needs sent_idx (WIP on para_idx)
            edu2sent = doc_name2edu2sent[doc_name]
            dt_pred.sent_idx = edu2sent
            # 2016-06-28 same for edu2para
            edu2para = doc_name2edu2para[doc_name]
            dt_pred.para_idx = edu2para
            dt_pred.ranks = rnk_clf.predict([dt_pred])[0]
            # end NEW
            rtree_pred = deptree_to_rst_tree(dt_pred)
            ctree_pred2[doc_name] = rtree_pred
        #
        skipped_docs = set()
        ctree_pred2 = [ct for doc_name, ct in sorted(ctree_pred2.items())
                       if doc_name not in skipped_docs]
        print(parseval_report(ctree_true, ctree_pred2, digits=4))
        if detailed:
            print(parseval_detailed_report(ctree_true, ctree_pred2,
                                           metric_type='S+R'))
