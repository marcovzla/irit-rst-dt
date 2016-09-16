"""This module evaluates the output of discourse parsers.

Included are dependency and constituency tree metrics.
"""

from __future__ import print_function

import argparse
import os

from educe.rst_dt.annotation import _binarize
from educe.rst_dt.corpus import (RstRelationConverter,
                                 Reader as RstReader)
from educe.rst_dt.dep2con import (DummyNuclearityClassifier,
                                  InsideOutAttachmentRanker)
from educe.rst_dt.deptree import RstDepTree
#
from attelo.metrics.constituency import (parseval_detailed_report,
                                         parseval_report)
from attelo.metrics.deptree import compute_uas_las

# local to this package
from evals.codra import load_codra_ctrees, load_codra_dtrees
from evals.ours import (load_deptrees_from_attelo_output,
                        load_attelo_ctrees,
                        load_attelo_dtrees)


# RST corpus
CORPUS_DIR = os.path.join('corpus', 'RSTtrees-WSJ-main-1.0/')
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
CD_TEST = os.path.join(CORPUS_DIR, 'TEST')
# relation converter (fine- to coarse-grained labels)
RELMAP_FILE = os.path.join('/home/mmorey/melodi/educe',
                           'educe', 'rst_dt',
                           'rst_112to18.txt')
REL_CONV = RstRelationConverter(RELMAP_FILE).convert_tree


#
# EVALUATIONS
#

# * syntax: pred vs gold
EDUS_FILE = os.path.join('/home/mmorey/melodi',
                         'irit-rst-dt/TMP/syn_gold_coarse',
                         'TEST.relations.sparse.edu_input')
# outputs of parsers
EISNER_OUT_SYN_PRED = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/syn_pred_coarse',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

# 2016-09-14 "tree" transform, predicted syntax
EISNER_OUT_TREE_SYN_PRED_SU = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/latest',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt_su-eisner')
# end 2016-09-14


EISNER_OUT_SYN_PRED_SU = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/latest',  # lbl
    'scratch-current/combined',
    'output.maxent-AD.L-jnt_su-eisner')

EISNER_OUT_SYN_GOLD = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/syn_gold_coarse',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

CODRA_OUT_DIR = '/home/mmorey/melodi/rst/joty/Doc-level'

# level of detail for parseval
DETAILED = False
# hyperparams
NUC_STRATEGY = 'unamb_else_most_frequent'
RNK_STRATEGY = 'sdist-edist-rl'
RNK_PRIORITY_SU = True


def setup_dtree_postprocessor(nary_enc):
    """Setup the nuclearity and rank classifiers to flesh out dtrees."""
    # tie the order with the encoding for n-ary nodes
    order = 'weak' if nary_enc == 'tree' else 'strict'
    # load train section of the RST corpus, fit (currently dummy) classifiers
    # for nuclearity and rank
    reader_train = RstReader(CD_TRAIN)
    corpus_train = reader_train.slurp()
    # gold RST trees
    ctree_true = dict()  # ctrees
    dtree_true = dict()  # dtrees from the original ctrees ('tree' transform)

    for doc_id, ct_true in sorted(corpus_train.items()):
        doc_name = doc_id.doc
        # flavours of ctree
        ct_true = REL_CONV(ct_true)  # map fine to coarse relations
        ctree_true[doc_name] = ct_true
        # flavours of dtree
        dt_true = RstDepTree.from_rst_tree(ct_true, nary_enc=nary_enc)
        dtree_true[doc_name] = dt_true
    # fit classifiers for nuclearity and rank (DIRTY)
    # NB: both are (dummily) fit on weakly ordered dtrees
    X_train = []
    y_nuc_train = []
    y_rnk_train = []
    for doc_name, dt in sorted(dtree_true.items()):
        X_train.append(dt)
        y_nuc_train.append(dt.nucs)
        y_rnk_train.append(dt.ranks)
    # nuclearity clf
    nuc_clf = DummyNuclearityClassifier(strategy=NUC_STRATEGY)
    nuc_clf.fit(X_train, y_nuc_train)
    # rank clf
    rnk_clf = InsideOutAttachmentRanker(strategy=RNK_STRATEGY,
                                        prioritize_same_unit=RNK_PRIORITY_SU,
                                        order=order)
    rnk_clf.fit(X_train, y_rnk_train)
    return nuc_clf, rnk_clf


# FIXME:
# * [ ] create summary table with one system per row, one metric per column,
#   keep only the f-score (because for binary trees with manual segmentation
#   precision = recall = f-score).
def main():
    """Run the eval"""
    parser = argparse.ArgumentParser(
        description="Evaluate parsers' output against a given reference")
    # predictions
    parser.add_argument('authors_pred', nargs='+',
                        choices=['gold', 'silver',
                                 'joty', 'feng', 'ji',
                                 'ours_chain', 'ours_tree'],
                        help="Author(s) of the predictions")
    parser.add_argument('--nary_enc_pred', default='tree',
                        choices=['tree', 'chain'],
                        help="Encoding of n-ary nodes for the predictions")
    # reference
    parser.add_argument('--author_true', default='gold',
                        choices=['gold', 'silver',
                                 'joty', 'feng', 'ji',
                                 'ours_chain', 'ours_tree'],
                        help="Author of the reference")
    # * dtree eval
    parser.add_argument('--nary_enc_true', default='tree',
                        choices=['tree', 'chain'],
                        help="Encoding of n-ary nodes for the reference")
    # * ctree eval
    parser.add_argument('--binarize_true', action='store_true',
                        help="Binarize the reference ctree for the eval")

    #
    args = parser.parse_args()
    author_true = args.author_true
    nary_enc_true = args.nary_enc_true
    authors_pred = args.authors_pred
    nary_enc_pred = args.nary_enc_pred
    binarize_true = args.binarize_true

    # 0. setup the postprocessors to flesh out unordered dtrees into ordered
    # ones with nuclearity
    nuc_clf, rnk_clf = setup_dtree_postprocessor(nary_enc_pred)

    # the eval compares parses for the test section of the RST corpus
    reader_test = RstReader(CD_TEST)
    corpus_test = reader_test.slurp()

    # reference
    # current assumption: author_true is 'gold'
    if author_true != 'gold':
        raise NotImplementedError('Not yet')

    ctree_true = dict()  # ctrees
    dtree_true = dict()  # dtrees from the original ctrees ('tree' transform)
    for doc_id, ct_true in sorted(corpus_test.items()):
        doc_name = doc_id.doc
        # original reference ctree, with coarse labels
        ct_true = REL_CONV(ct_true)  # map fine to coarse relations
        # corresponding dtree
        dt_true = RstDepTree.from_rst_tree(ct_true, nary_enc=nary_enc_true)
        dtree_true[doc_name] = dt_true
        # binarize ctree if necessary
        if binarize_true:
            ct_true = _binarize(ct_true)
        ctree_true[doc_name] = ct_true

    
    c_preds = []  # predictions: [(parser_name, dict(doc_name, ct_pred))]
    d_preds = []  # predictions: [(parser_name, dict(doc_name, dt_pred))]
    if 'joty' in authors_pred:
        # CODRA outputs RST ctrees ; eval_codra_output maps them to RST dtrees
        c_preds.append(
            ('joty', load_codra_ctrees(CODRA_OUT_DIR, REL_CONV))
        )
        d_preds.append(
            ('joty', load_codra_dtrees(CODRA_OUT_DIR, REL_CONV,
                                       nary_enc='chain'))
        )
        # joty-{chain,tree} would be the same except nary_enc='tree' ;
        # the nary_enc does not matter because codra outputs binary ctrees,
        # hence both encodings result in (the same) strictly ordered dtrees

    if 'ours_chain' in authors_pred:
        # Eisner, predicted syntax, chain
        c_preds.append(
            ('ours-chain', load_attelo_ctrees(EISNER_OUT_SYN_PRED, EDUS_FILE,
                                              nuc_clf, rnk_clf))
        )
        d_preds.append(
            ('ours-chain', load_attelo_dtrees(EISNER_OUT_SYN_PRED, EDUS_FILE,
                                              nuc_clf, rnk_clf))
        )

    if 'ours_tree' in authors_pred:
        # Eisner, predicted syntax, tree + same-unit
        c_preds.append(
            ('ours-tree', load_attelo_ctrees(EISNER_OUT_TREE_SYN_PRED_SU,
                                             EDUS_FILE,
                                             nuc_clf, rnk_clf))
        )
        d_preds.append(
            ('ours-tree', load_attelo_dtrees(EISNER_OUT_TREE_SYN_PRED_SU,
                                             EDUS_FILE,
                                             nuc_clf, rnk_clf))
        )

    if False:  # FIXME repair (or forget) these
        print('Eisner, predicted syntax + same-unit')
        load_deptrees_from_attelo_output(ctree_true, dtree_true,
                                         EISNER_OUT_SYN_PRED_SU, EDUS_FILE,
                                         nuc_clf, rnk_clf,
                                         detailed=False)
        print('======================')

        print('Eisner, gold syntax')
        load_deptrees_from_attelo_output(ctree_true, dtree_true,
                                         EISNER_OUT_SYN_GOLD, EDUS_FILE,
                                         nuc_clf, rnk_clf,
                                         detailed=False)
        print('======================')

    # dependency eval

    # report
    # * table format
    digits = 4
    width = max(len(parser_name) for parser_name, _ in d_preds)

    headers = ["UAS", "LAS", "LS"]
    fmt = '%% %ds' % width  # first col: parser name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'
    # end table format and header line

    # * table content
    for parser_name, dtree_pred in d_preds:
        doc_names = sorted(dtree_true.keys())
        dtree_true_list = [dtree_true[doc_name] for doc_name in doc_names]
        dtree_pred_list = [dtree_pred[doc_name] for doc_name in doc_names]
        score_uas, score_las, score_ls = compute_uas_las(dtree_true_list,
                                                         dtree_pred_list)
        # append to report
        values = ['{pname: <{fill}}'.format(pname=parser_name, fill=width)]
        for v in (score_uas, score_las, score_ls):
            values += ["{0:0.{1}f}".format(v, digits)]
        report += fmt % tuple(values)
    # end table content
    print(report)
    # end report

    # constituency eval
    for parser_name, ctree_pred in c_preds:
        doc_names = sorted(ctree_true.keys())
        ctree_true_list = [ctree_true[doc_name] for doc_name in doc_names]
        ctree_pred_list = [ctree_pred[doc_name] for doc_name in doc_names]
        # FIXME
        # compute and print PARSEVAL scores
        print(parser_name)
        print(parseval_report(ctree_true_list, ctree_pred_list, digits=4))
        # detailed report on S+N+R
        if DETAILED:
            print(parseval_detailed_report(ctree_true_list, ctree_pred_list,
                                           metric_type='S+R'))
        # end FIXME


if __name__ == '__main__':
    main()
