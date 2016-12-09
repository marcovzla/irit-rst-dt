"""TODO

"""

from __future__ import absolute_import, print_function
from collections import Counter
from glob import glob
import os

# educe
from educe.learning.edu_input_format import load_edu_input_file
from educe.rst_dt.corpus import (RstRelationConverter,
                                 Reader as RstReader)
from educe.rst_dt.dep2con import deptree_to_rst_tree
from educe.rst_dt.deptree import NUC_S, RstDepTree, RstDtException
from educe.rst_dt.metrics.rst_parseval import rst_parseval_report
# attelo
from attelo.metrics.deptree import compute_uas_las as att_compute_uas_las
# local imports
from evals.showdown import EDUS_FILE_PAT, setup_dtree_postprocessor


# RST corpus
CORPUS_DIR = os.path.join('corpus', 'RSTtrees-WSJ-main-1.0/')
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
CD_TEST = os.path.join(CORPUS_DIR, 'TEST')
# relation converter (fine- to coarse-grained labels)
RELMAP_FILE = os.path.join('/home/mmorey/melodi/educe',
                           'educe', 'rst_dt',
                           'rst_112to18.txt')
REL_CONV = RstRelationConverter(RELMAP_FILE).convert_tree


# output of Li et al.'s parser
SAVE_DIR = "/home/mmorey/melodi/rst/li_sujian/TextLevelDiscourseParser/mybackup/mstparser-code-116-trunk/mstparser/save"
COARSE_FILES = [
    "136.0detailedOutVersion2.txt",
    "151.0detailedOut.txt",
    "164.0detailedOut.txt",
    "177.0detailedOut.txt",
    "335.0detailedOut.txt",
    "37.0detailedOut.txt",
    "424.0detailedOut.txt",
    "448.0detailedOut.txt",
    "455.0detailedOutVersion2.txt",
    "513.0detailedOutVersion2.txt",
    "529.0detailedOut.txt",
    "615.0detailedOutVersion2.txt",
    "712.0detailedOut.txt",
    "917.0detailedOut.txt",
]
FINE_FILES = [
    "190.0detailedOut.txt",
    "473.0detailedOutVersion2.txt",
    "561.0detailedOut.txt",
    "723.0detailedOut.txt",
    "747.0detailedOutVersion2.txt",
    "825.0detailedOut.txt",
    "947.0detailedOut.txt",
    "965.0detailedOutVersion2.txt",
]
# different format for predicted labels and description of EDU
COARSE_FEAT_FILES = [
    "441.0detailedOut.txt",
]

# default file(s) to include ; I picked a coarse-grained one with good scores
DEFAULT_FILES = ["712.0detailedOut.txt"]


def load_output_file(out_file):
    """Load an output file from Li et al.'s dep parser.
    """
    doc_names = []
    heads_true = []
    labels_true = []
    heads_pred = []
    labels_pred = []
    with open(out_file) as f:
        for line in f:
            if line.startswith(".\\testdata"):
                # file
                doc_name = line.strip().split("\\")[2][:12]  # drop .edus or else
                # print(doc_name)
                doc_names.append(doc_name)
                heads_true.append([-1])  # initial pad for fake root
                labels_true.append([''])
                heads_pred.append([-1])
                labels_pred.append([''])
            else:
                edu_idx, hd_true, hd_pred, lbl_true, lbl_pred, edu_str = line.strip().split(' ', 5)
                if lbl_pred == '<no-type>':
                    # not sure whether this should be enabled
                    lbl_pred = 'Elaboration'
                heads_true[-1].append(int(hd_true))
                labels_true[-1].append(lbl_true)
                heads_pred[-1].append(int(hd_pred))
                labels_pred[-1].append(lbl_pred)
    res = {
        'doc_names': doc_names,
        'heads_true': heads_true,
        'labels_true': labels_true,
        'heads_pred': heads_pred,
        'labels_pred': labels_pred,
    }
    return res


if __name__ == "__main__":
    # load dep trees from corpus
    reader_test = RstReader(CD_TEST)
    corpus_test = reader_test.slurp()

    # choice of predictions: granularity of relations
    RST_RELS = 'coarse'
    if RST_RELS == 'coarse':
        PRED_FILES = DEFAULT_FILES  # COARSE_FILES
    else:
        PRED_FILES = FINE_FILES
    # eval procedure: the one in the parser of Li et al. vs standard one
    EVAL_LI = False

    # setup conversion from c- to d-tree and back, and eval type
    nary_enc = 'chain'

    if EVAL_LI:
        # reconstruction of the c-tree
        order = 'strict'
        nuc_strategy = 'constant'
        nuc_constant = NUC_S
        rnk_strategy = 'lllrrr'
        rnk_prioritize_same_unit = False
        # eval
        TWIST_GOLD = True
        ADD_TRIVIAL_SPANS = True
    else:  # comparable setup to what we use for our own parsers
        order = 'weak'
        nuc_strategy = "unamb_else_most_frequent"
        nuc_constant = None
        rnk_strategy = "sdist-edist-rl"
        rnk_prioritize_same_unit = True
        TWIST_GOLD = False
        ADD_TRIVIAL_SPANS = False

    nuc_clf, rnk_clf = setup_dtree_postprocessor(
        nary_enc=nary_enc, order=order, nuc_strategy=nuc_strategy,
        nuc_constant=nuc_constant, rnk_strategy=rnk_strategy,
        rnk_prioritize_same_unit=rnk_prioritize_same_unit)

    ctree_true = dict()
    dtree_true = dict()
    labelset_true = Counter()
    for doc_id, ct_true in sorted(corpus_test.items()):
        doc_name = doc_id.doc
        if RST_RELS == 'coarse':
            # map fine to coarse rels
            ct_true = REL_CONV(ct_true)
        ctree_true[doc_name] = ct_true
        dt_true = RstDepTree.from_rst_tree(ct_true, nary_enc=nary_enc)
        # dirty hack: lowercase ROOT
        dt_true.labels = [x.lower() if x == 'ROOT' else x
                          for x in dt_true.labels]

        dtree_true[doc_name] = dt_true
        labelset_true.update(dt_true.labels[1:])

    # load parser output
    for fname in PRED_FILES:
        dtree_pred = dict()
        labelset_pred = Counter()
        #
        f_cur = os.path.join(SAVE_DIR, fname)
        dep_bunch = load_output_file(f_cur)
        doc_names = dep_bunch['doc_names']
        # load and process _pred
        for doc_name, heads_pred, labels_pred in zip(
                dep_bunch['doc_names'], dep_bunch['heads_pred'],
                dep_bunch['labels_pred']):
            # create dtree _pred
            edus_data = load_edu_input_file(EDUS_FILE_PAT.format(doc_name),
                                            edu_type='rst-dt')
            edus = edus_data['edus']
            edu2sent = edus_data['edu2sent']
            dt_pred = RstDepTree(edus)
            # add predicted edges
            for dep_idx, (gov_idx, lbl) in enumerate(zip(
                    heads_pred[1:], labels_pred[1:]), start=1):
                if lbl == '<no-type>':
                    lbl = 'Elaboration'
                # print(lbl)
                lbl = lbl.lower()
                labelset_pred[lbl] += 1
                dt_pred.add_dependency(gov_idx, dep_idx, lbl)
            dt_pred.sent_idx = [0] + edu2sent  # 0 for fake root + dirty
            dtree_pred[doc_name] = dt_pred
        # end WIP

        if RST_RELS == 'coarse':
            expected_labelset = ['attribution', 'background', 'cause', 'comparison', 'condition', 'contrast', 'elaboration', 'enablement', 'evaluation', 'explanation', 'joint', 'manner-means', 'root', 'same-unit', 'summary', 'temporal', 'textual', 'topic-change', 'topic-comment']
            assert sorted(labelset_pred.keys()) == expected_labelset
            # wsj_1189 has a weird "span" label in a multinuclear rel at [7--9]
            # see footnote in Hayashi et al's SIGDIAL 2016 paper
            assert sorted(labelset_true.keys()) == sorted(
                expected_labelset + ['span'])

        # build predicted c-trees using our heuristics for nuc and rank
        ctree_pred = dict()
        for doc_name, dt_pred in dtree_pred.items():
            # 1. enrich d-tree with nuc and order
            # a. order: the procedure that generates spans produces a
            # left-heavy branching: ((A B) C), which should be our
            # "lllrrr" heuristic
            dt_pred.ranks = rnk_clf.predict([dt_pred])[0]
            # b. nuclearity: heuristic baseline
            dt_pred.nucs = nuc_clf.predict([dt_pred])[0]
            # 2. build _pred c-tree
            try:
                ct_pred = deptree_to_rst_tree(dt_pred)
                ctree_pred[doc_name] = ct_pred
            except RstDtException as rst_e:
                print(rst_e)
                raise
            # 3. predict nuc and order in _true d-tree, replace the _true
            # c-tree with a twisted one, like in their eval
            if TWIST_GOLD:
                dt_true = dtree_true[doc_name]
                dt_true.sent_idx = [0] + edu2sent
                dt_true.ranks = rnk_clf.predict([dt_true])[0]
                dt_true.nucs = nuc_clf.predict([dt_true])[0]
                ct_true = ctree_true[doc_name]
                try:
                    ct_true = deptree_to_rst_tree(dt_true)
                except RstDtException as rst_e:
                    print(rst_e)
                    raise
                ctree_true[doc_name] = ct_true

        # compute UAS and LAS on the _true values from the corpus and
        # _pred Educe RstDepTrees re-built from their output files
        dtree_true_list = [dtree_true[doc_name] for doc_name in doc_names]
        dtree_pred_list = [dtree_pred[doc_name] for doc_name in doc_names]
        sc_uas, sc_las, sc_las_n, sc_las_o, sc_las_no = att_compute_uas_las(
            dtree_true_list, dtree_pred_list, include_ls=False,
            include_las_n_o_no=True)
        print(("{}\tUAS={:.4f}\tLAS={:.4f}\tLAS+N={:.4f}\tLAS+O={:.4f}\t"
               "LAS+N+O={:.4f}").format(
                   fname, sc_uas, sc_las, sc_las_n, sc_las_o, sc_las_no))
            
        # compute RST-Parseval of these c-trees
        ctree_true_list = [ctree_true[doc_name] for doc_name in doc_names]
        ctree_pred_list = [ctree_pred[doc_name] for doc_name in doc_names]
        print(rst_parseval_report(ctree_true_list, ctree_pred_list,
                                  ctree_type='RST', digits=4,
                                  per_doc=False,
                                  add_trivial_spans=ADD_TRIVIAL_SPANS,
                                  stringent=False))
