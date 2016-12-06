"""This module evaluates the output of discourse parsers.

Included are dependency and constituency tree metrics.
"""

from __future__ import print_function

import argparse
import codecs
import os

from educe.rst_dt.annotation import _binarize, SimpleRSTTree
from educe.rst_dt.corpus import (RstRelationConverter,
                                 Reader as RstReader)
from educe.rst_dt.dep2con import (DummyNuclearityClassifier,
                                  InsideOutAttachmentRanker)
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.metrics.rst_parseval import (rst_parseval_detailed_report,
                                               rst_parseval_report)
#
from attelo.metrics.deptree import (compute_uas_las,
                                    compute_uas_las_undirected)

# local to this package
from evals.codra import load_codra_ctrees, load_codra_dtrees
from evals.feng import load_feng_ctrees, load_feng_dtrees
from evals.gcrf_tree_format import load_gcrf_ctrees, load_gcrf_dtrees
from evals.hayashi_cons import (load_hayashi_hilda_ctrees,
                                load_hayashi_hilda_dtrees)
from evals.hayashi_deps import load_hayashi_dtrees
from evals.ji import load_ji_ctrees, load_ji_dtrees
from evals.li_qi import load_li_qi_ctrees, load_li_qi_dtrees
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
EISNER_OUT_TREE_SYN_PRED = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/2016-09-12T0825',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

EISNER_OUT_TREE_SYN_PRED_SU = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/2016-09-12T0825',  # lbl
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

# output of Joty's parser CODRA
CODRA_OUT_DIR = '/home/mmorey/melodi/rst/joty/Doc-level'
# output of Ji's parser DPLP
JI_OUT_DIR = os.path.join('/home/mmorey/melodi/rst/ji_eisenstein/DPLP/data/docs/test/')
# Feng's parsers
FENG_DIR = '/home/mmorey/melodi/rst/feng_hirst/'
FENG1_OUT_DIR = os.path.join(FENG_DIR, 'phil', 'tmp')
FENG2_OUT_DIR = os.path.join(FENG_DIR, 'gCRF_dist/texts/results/test_batch_gold_seg')
# Li Qi's parser
LI_QI_OUT_DIR = '/home/mmorey/melodi/rst/li_qi/result'
# Hayashi's HILDA
HAYASHI_OUT_DIR = '/home/mmorey/melodi/rst/hayashi/SIGDIAL'
HAYASHI_HILDA_OUT_DIR = os.path.join(HAYASHI_OUT_DIR, 'auto_parse/cons/HILDA')

# level of detail for parseval
DETAILED = False
EVAL_LI_DEP = True
STRINGENT = False
# additional dependency metrics
INCLUDE_LS = False
UNDIRECTED_DEPS = False
EVAL_NUC_RANK = True
# hyperparams
NUC_STRATEGY = 'unamb_else_most_frequent'
NUC_CONSTANT = None  # only useful for NUC_STRATEGY='constant'
RNK_STRATEGY = 'sdist-edist-rl'
RNK_PRIORITY_SU = True


def setup_dtree_postprocessor(nary_enc='chain', order='strict',
                              nuc_strategy=NUC_STRATEGY,
                              nuc_constant=NUC_CONSTANT,
                              rnk_strategy=RNK_STRATEGY,
                              rnk_prioritize_same_unit=RNK_PRIORITY_SU):
    """Setup the nuclearity and rank classifiers to flesh out dtrees."""
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
    nuc_clf = DummyNuclearityClassifier(strategy=nuc_strategy,
                                        constant=nuc_constant)
    nuc_clf.fit(X_train, y_nuc_train)
    # rank clf
    rnk_clf = InsideOutAttachmentRanker(
        strategy=rnk_strategy, prioritize_same_unit=rnk_prioritize_same_unit,
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
                                 'joty', 'feng', 'feng2', 'ji',
                                 'li_qi', 'hayashi_hilda',
                                 'ours_chain', 'ours_tree', 'ours_tree_su'],
                        help="Author(s) of the predictions")
    parser.add_argument('--nary_enc_pred', default='tree',
                        choices=['tree', 'chain'],
                        help="Encoding of n-ary nodes for the predictions")
    # reference
    parser.add_argument('--author_true', default='gold',
                        choices=['gold', 'silver',
                                 'joty', 'feng', 'feng2', 'ji',
                                 'li_qi', 'hayashi_hilda',
                                 'ours_chain', 'ours_tree'],
                        help="Author of the reference")
    # * dtree eval
    parser.add_argument('--nary_enc_true', default='tree',
                        choices=['tree', 'chain'],
                        help="Encoding of n-ary nodes for the reference")
    # * ctree eval
    parser.add_argument('--binarize_true', action='store_true',
                        help="Binarize the reference ctree for the eval")
    parser.add_argument('--simple_rsttree', action='store_true',
                        help="Binarize ctree and move relations up")
    parser.add_argument('--per_doc', action='store_true',
                        help="Doc-averaged scores (cf. Ji's eval)")
    #
    args = parser.parse_args()
    author_true = args.author_true
    nary_enc_true = args.nary_enc_true
    authors_pred = args.authors_pred
    nary_enc_pred = args.nary_enc_pred
    binarize_true = args.binarize_true
    simple_rsttree = args.simple_rsttree

    # "per_doc = True" computes p, r, f as in DPLP: compute scores per doc
    # then average over docs
    # it should be False, except for comparison with the DPLP paper
    per_doc = args.per_doc
    #
    if binarize_true and nary_enc_true != 'chain':
        raise ValueError("--binarize_true is compatible with "
                         "--nary_enc_true chain only")

    # 0. setup the postprocessors to flesh out unordered dtrees into ordered
    # ones with nuclearity
    # * tie the order with the encoding for n-ary nodes
    order = 'weak' if nary_enc_pred == 'tree' else 'strict'
    nuc_clf, rnk_clf = setup_dtree_postprocessor(nary_enc=nary_enc_pred,
                                                 order=order)

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
        if binarize_true:
            # binarize ctree if required
            ct_true = _binarize(ct_true)
        ctree_true[doc_name] = ct_true
        # corresponding dtree
        dt_true = RstDepTree.from_rst_tree(ct_true, nary_enc=nary_enc_true)
        dtree_true[doc_name] = dt_true

    
    c_preds = []  # predictions: [(parser_name, dict(doc_name, ct_pred))]
    d_preds = []  # predictions: [(parser_name, dict(doc_name, dt_pred))]

    for author_pred in authors_pred:
        if author_pred == 'hayashi_hilda':
            c_preds.append(
                ('hayashi_hilda', load_hayashi_hilda_ctrees(
                    HAYASHI_HILDA_OUT_DIR, REL_CONV))
            )
            d_preds.append(
                ('hayashi_hilda', load_hayashi_hilda_dtrees(
                    HAYASHI_HILDA_OUT_DIR, REL_CONV, nary_enc='chain'))
            )

        if author_pred == 'li_qi':
            c_preds.append(
                ('li_qi', load_li_qi_ctrees(LI_QI_OUT_DIR, REL_CONV))
            )
            d_preds.append(
                ('li_qi', load_li_qi_dtrees(LI_QI_OUT_DIR, REL_CONV,
                                            nary_enc='chain'))
            )

        if author_pred == 'feng':
            c_preds.append(
                ('gSVM', load_feng_ctrees(FENG1_OUT_DIR, REL_CONV))
            )
            d_preds.append(
                ('gSVM', load_feng_dtrees(FENG1_OUT_DIR, REL_CONV,
                                          nary_enc='chain'))
            )

        if author_pred == 'feng2':
            c_preds.append(
                ('gCRF', load_gcrf_ctrees(FENG2_OUT_DIR, REL_CONV))
            )
            d_preds.append(
                ('gCRF', load_gcrf_dtrees(FENG2_OUT_DIR, REL_CONV,
                                          nary_enc='chain'))
            )

        if author_pred == 'joty':
            # CODRA outputs RST ctrees ; eval_codra_output maps them to RST dtrees
            c_preds.append(
                ('TSP 1-1', load_codra_ctrees(CODRA_OUT_DIR, REL_CONV))
            )
            d_preds.append(
                ('TSP 1-1', load_codra_dtrees(CODRA_OUT_DIR, REL_CONV,
                                              nary_enc='chain'))
            )
            # joty-{chain,tree} would be the same except nary_enc='tree' ;
            # the nary_enc does not matter because codra outputs binary ctrees,
            # hence both encodings result in (the same) strictly ordered dtrees

        if author_pred == 'ji':
            # DPLP outputs RST ctrees in the form of lists of spans;
            # load_ji_dtrees maps them to RST dtrees
            c_preds.append(
                ('DPLP', load_ji_ctrees(JI_OUT_DIR, REL_CONV))
            )
            d_preds.append(
                ('DPLP', load_ji_dtrees(JI_OUT_DIR, REL_CONV,
                                        nary_enc='chain'))
            )
            # ji-{chain,tree} would be the same except nary_enc='tree' ;
            # the nary_enc does not matter because codra outputs binary ctrees,
            # hence both encodings result in (the same) strictly ordered dtrees

        if author_pred == 'ours_chain':
            # Eisner, predicted syntax, chain
            c_preds.append(
                ('ours-chain', load_attelo_ctrees(EISNER_OUT_SYN_PRED, EDUS_FILE,
                                                  nuc_clf, rnk_clf))
            )
            d_preds.append(
                ('ours-chain', load_attelo_dtrees(EISNER_OUT_SYN_PRED, EDUS_FILE,
                                                  nuc_clf, rnk_clf))
            )

        if author_pred == 'ours_tree':
            # Eisner, predicted syntax, tree + same-unit
            c_preds.append(
                ('ours-tree', load_attelo_ctrees(EISNER_OUT_TREE_SYN_PRED,
                                                 EDUS_FILE,
                                                 nuc_clf, rnk_clf))
            )
            d_preds.append(
                ('ours-tree', load_attelo_dtrees(EISNER_OUT_TREE_SYN_PRED,
                                                 EDUS_FILE,
                                                 nuc_clf, rnk_clf))
            )
        if author_pred == 'ours_tree_su':
            # Eisner, predicted syntax, tree + same-unit
            c_preds.append(
                ('ours-tree-su', load_attelo_ctrees(EISNER_OUT_TREE_SYN_PRED_SU,
                                                    EDUS_FILE,
                                                    nuc_clf, rnk_clf))
            )
            d_preds.append(
                ('ours-tree-su', load_attelo_dtrees(EISNER_OUT_TREE_SYN_PRED_SU,
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

    headers = ["UAS", "LAS"]
    if INCLUDE_LS:
        headers += ["LS"]
    if EVAL_NUC_RANK:
        headers += ["LAS+N", "LAS+O", "LAS+N+O"]
    if UNDIRECTED_DEPS:
        headers += ["UUAS", "ULAS"]
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
        # WIP print per doc eval
        if not os.path.exists(parser_name):
            os.makedirs(parser_name)
        for doc_name, dt_true, dt_pred in zip(
                doc_names, dtree_true_list, dtree_pred_list):
            with codecs.open(parser_name + '/' + doc_name + '.d_eval',
                             mode='w', encoding='utf-8') as f:
                print(', '.join('{:.4f}'.format(x)
                                for x in compute_uas_las(
                                        [dt_true], [dt_pred],
                                        include_ls=INCLUDE_LS,
                                        include_las_n_o_no=EVAL_NUC_RANK)),
                      file=f)
                if UNDIRECTED_DEPS:
                    # scores for undirected edges
                    print(', '.join('{:.4f}'.format(x)
                                    for x in compute_uas_las_undirected(
                                            [dt_true], [dt_pred])),
                          file=f)
        # end WIP print

        all_scores = []
        all_scores += list(compute_uas_las(
            dtree_true_list, dtree_pred_list, include_ls=INCLUDE_LS,
            include_las_n_o_no=EVAL_NUC_RANK))
        if UNDIRECTED_DEPS:
            score_uuas, score_ulas = compute_uas_las_undirected(
                dtree_true_list, dtree_pred_list)
            all_scores += [score_uuas, score_ulas]
        # append to report
        values = ['{pname: <{fill}}'.format(pname=parser_name, fill=width)]
        for v in all_scores:
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
        if simple_rsttree:
            ctree_true_list = [SimpleRSTTree.from_rst_tree(x)
                               for x in ctree_true_list]
            ctree_pred_list = [SimpleRSTTree.from_rst_tree(x)
                               for x in ctree_pred_list]
            ctree_type = 'SimpleRST'
        else:
            ctree_type = 'RST'

        # WIP print SimpleRSTTrees
        if not os.path.exists('gold'):
            os.makedirs('gold')
        for doc_name, ct in zip(doc_names, ctree_true_list):
            with codecs.open('gold/' + ct.origin.doc, mode='w',
                             encoding='utf-8') as f:
                print(ct, file=f)
        if not os.path.exists(parser_name):
            os.makedirs(parser_name)
        for doc_name, ct in zip(doc_names, ctree_pred_list):
            with codecs.open(parser_name + '/' + doc_name, mode='w',
                             encoding='utf-8') as f:
                print(ct, file=f)
        # WIP eval each tree in turn
        for doc_name, ct_true, ct_pred in zip(
                doc_names, ctree_true_list, ctree_pred_list):
            with codecs.open(parser_name + '/' + doc_name + '.c_eval',
                             mode='w', encoding='utf-8') as f:
                print(rst_parseval_report([ct_true], [ct_pred],
                                          ctree_type=ctree_type,
                                          digits=4,
                                          per_doc=per_doc,
                                          add_trivial_spans=EVAL_LI_DEP,
                                          stringent=STRINGENT),
                      file=f)
        # end WIP
        # FIXME
        # compute and print PARSEVAL scores
        print(parser_name)
        print(rst_parseval_report(ctree_true_list, ctree_pred_list,
                                  ctree_type=ctree_type, digits=4,
                                  per_doc=per_doc,
                                  add_trivial_spans=EVAL_LI_DEP,
                                  stringent=STRINGENT))
        # detailed report on S+N+R
        if DETAILED:
            print(rst_parseval_detailed_report(
                ctree_true_list, ctree_pred_list, ctree_type=ctree_type,
                metric_type='S+R'))
        # end FIXME


if __name__ == '__main__':
    main()
