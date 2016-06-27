"""Evaluate our parsers.

"""

from __future__ import print_function

from collections import defaultdict
import os

from educe.annotation import Span as EduceSpan
from educe.rst_dt.annotation import (EDU as EduceEDU,
                                     SimpleRSTTree, _binarize)
from educe.rst_dt.corpus import (Reader as RstReader,
                                 RstRelationConverter as RstRelationConverter)
from educe.rst_dt.dep2con import (deptree_to_simple_rst_tree,
                                  DummyNuclearityClassifier,
                                  InsideOutAttachmentRanker)
from educe.rst_dt.deptree import RstDepTree, RstDtException
#
from attelo.io import load_edus
from attelo.metrics.constituency import (parseval_detailed_report,
                                         parseval_report)
from attelo.metrics.deptree import compute_uas_las
from attelo.table import UNRELATED  # for load_attelo_output_file


# RST corpus
CORPUS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', 'corpus',
    'RSTtrees-WSJ-main-1.0/'))
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
CD_TEST = os.path.join(CORPUS_DIR, 'TEST')
# relation converter (fine- to coarse-grained labels)
RELMAP_FILE = os.path.join('/home/mmorey/melodi/educe',
                           'educe', 'rst_dt',
                           'rst_112to18.txt')
REL_CONV = RstRelationConverter(RELMAP_FILE).convert_tree


# move to attelo.datasets.attelo_out_format
def load_attelo_output_file(output_file):
    """Load edges from an attelo output file.

    An attelo output file typically contains edges from several
    documents. This function indexes edges by the name of their
    document.

    Parameters
    ----------
    output_file: string
        Path to the attelo output file

    Returns
    -------
    edges_pred: dict(string, [(string, string, string)])
        Predicted edges for each document, indexed by doc name

    Notes
    -----
    See `attelo.io.load_predictions` that is almost equivalent to this
    function. They are expected to converge some day into a better,
    obvious in retrospect, function.
    """
    edges_pred = defaultdict(list)
    with open(output_file) as f:
        for line in f:
            src_id, tgt_id, lbl = line.strip().split('\t')
            if lbl != UNRELATED:
                # dirty hack: get doc name from EDU id
                # e.g. (EDU id = wsj_0601_1) => (doc id = wsj_0601)
                doc_name = tgt_id.rsplit('_', 1)[0]
                edges_pred[doc_name].append((src_id, tgt_id, lbl))

    return edges_pred


def load_deptrees_from_attelo_output(output_file, edus_file,
                                     nuc_strategy, rank_strategy,
                                     prioritize_same_unit=True,
                                     skpd_docs=None):
    """Load an RstDepTree from the output of attelo.

    Parameters
    ----------
    output_file: string
        Path to the file that contains attelo's output
    nuc_strategy: string
        Strategy to predict nuclearity
    rank_strategy: string
        Strategy to predict attachment ranking
    skpd_docs: set(string)
        Names of documents that should be skipped to compute scores

    Returns
    -------
    skipped_docs: set(string)
        Names of documents that have been skipped to compute scores
    """
    # load reference trees
    dtree_true = dict()  # dependency trees
    ctree_true = dict()  # constituency trees
    # FIXME: find ways to read the right (not necessarily TEST) section
    # and only the required documents
    rst_reader = RstReader(CD_TEST)
    rst_corpus = rst_reader.slurp()
    for doc_id, rtree_true in sorted(rst_corpus.items()):
        doc_name = doc_id.doc

        # transform into binary tree with coarse-grained labels
        coarse_rtree_true = REL_CONV(rtree_true)
        bin_rtree_true = _binarize(coarse_rtree_true)
        ctree_true[doc_name] = bin_rtree_true

        # transform into dependency tree via SimpleRSTTree
        bin_srtree_true = SimpleRSTTree.from_rst_tree(coarse_rtree_true)
        dt_true = RstDepTree.from_simple_rst_tree(bin_srtree_true)
        dtree_true[doc_name] = dt_true

    # USE TO INCORPORATE CONSTITUENCY LOSS INTO STRUCTURED CLASSIFIERS
    # load predicted trees
    dtree_pred = dict()  # predicted dtrees
    ctree_pred = dict()  # predicted ctrees
    # load EDUs as they are known to attelo (sigh)
    # and predicted edges on these EDUs
    att_edus = load_edus(edus_file)
    edges_pred = load_attelo_output_file(output_file)
    # rebuild educe EDUs from their attelo description
    # and group them by doc_name
    educe_edus = defaultdict(list)
    edu2sent_idx = defaultdict(dict)
    gid2num = dict()
    for att_edu in att_edus:
        # doc name
        doc_name = att_edu.grouping
        # EDU info
        edu_num = int(att_edu.id.rsplit('_', 1)[1])
        edu_span = EduceSpan(att_edu.start, att_edu.end)
        edu_text = att_edu.text
        educe_edus[doc_name].append(EduceEDU(edu_num, edu_span, edu_text))
        # map global id of EDU to num of EDU inside doc
        gid2num[att_edu.id] = edu_num
        # map EDU to sentence
        sent_idx = int(att_edu.subgrouping.split('_sent')[1])
        edu2sent_idx[doc_name][edu_num] = sent_idx
    # sort EDUs by num
    educe_edus = {doc_name: sorted(edus, key=lambda e: e.num)
                  for doc_name, edus in educe_edus.items()}
    # rebuild educe-style edu2sent ; prepend 0 for the fake root
    doc_name2edu2sent = {doc_name: ([0] +
                                    [edu2sent_idx[doc_name][e.num]
                                     for e in doc_educe_edus])
                         for doc_name, doc_educe_edus in educe_edus.items()}

    # re-build predicted trees from predicted edges and educe EDUs
    skipped_docs = set()  # docs skipped because non-projective structures

    # classifiers for nuclearity and ranking
    # FIXME declare, fit and predict upstream...
    X_train = []
    y_nuc_train = []
    y_rank_train = []
    for doc_name, dt in sorted(dtree_true.items()):
        X_train.append(dt)
        y_nuc_train.append(dt.nucs)
        y_rank_train.append(dt.ranks)
    # nuclearity
    nuc_classifier = DummyNuclearityClassifier(strategy=nuc_strategy)
    nuc_classifier.fit(X_train, y_nuc_train)
    # ranking classifier
    rank_classifier = InsideOutAttachmentRanker(
        strategy=rank_strategy,
        prioritize_same_unit=prioritize_same_unit)
    rank_classifier.fit(X_train, y_rank_train)

    # rebuild RstDepTrees
    for doc_name, es_pred in sorted(edges_pred.items()):
        # get educe EDUs
        doc_educe_edus = educe_edus[doc_name]
        # create pred dtree
        dt_pred = RstDepTree(doc_educe_edus)
        for src_id, tgt_id, lbl in es_pred:
            if src_id == 'ROOT':
                if lbl == 'ROOT':
                    dt_pred.set_root(gid2num[tgt_id])
                else:
                    raise ValueError('Weird root label: {}'.format(lbl))
            else:
                dt_pred.add_dependency(gid2num[src_id], gid2num[tgt_id], lbl)
        # NEW add nuclearity: heuristic baseline
        if True:
            dt_pred.nucs = nuc_classifier.predict([dt_pred])[0]
        else:  # EXPERIMENTAL use gold nuclearity
            dt_pred.nucs = dtree_true[doc_name].nucs
        # NEW add rank: some strategies require a mapping from EDU to sentence
        # EXPERIMENTAL attach array of sentence index for each EDU in tree
        edu2sent = doc_name2edu2sent[doc_name]
        dt_pred.sent_idx = edu2sent
        # end EXPERIMENTAL
        if False:  # DEBUG
            print(doc_name)
        dt_pred.ranks = rank_classifier.predict([dt_pred])[0]
        # end NEW
        dtree_pred[doc_name] = dt_pred

        # create pred ctree
        try:
            bin_srtree_pred = deptree_to_simple_rst_tree(dt_pred)
            if False:  # EXPERIMENTAL
                # currently False to run on output that already has
                # labels embedding nuclearity
                bin_srtree_pred = SimpleRSTTree.incorporate_nuclearity_into_label(
                    bin_srtree_pred)
            bin_rtree_pred = SimpleRSTTree.to_binary_rst_tree(bin_srtree_pred)
            ctree_pred[doc_name] = bin_rtree_pred
        except RstDtException as rst_e:
            print(rst_e)
            skipped_docs.add(doc_name)
            if False:
                print('\n'.join('{}: {}'.format(edu.text_span(), edu)
                                for edu in educe_edus[doc_name]))
            # raise
    # end USE TO INCORPORATE CONSTITUENCY LOSS INTO STRUCTURED CLASSIFIERS

    # compare gold with pred on doc_names
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

    # compute and print PARSEVAL scores
    if skipped_docs:
        print('Skipped {} docs over {}'.format(len(skipped_docs),
                                               len(edges_pred)))
    # also skip docs passed as argument
    if skpd_docs is not None:
        skipped_docs |= skpd_docs
    # convert dicts to aligned lists of SimpleRSTTrees, skipping docs where
    # needed
    ctree_true = [ct for doc_name, ct in sorted(ctree_true.items())
                  if doc_name not in skipped_docs]
    ctree_pred = [ct for doc_name, ct in sorted(ctree_pred.items())
                  if doc_name not in skipped_docs]

    print(parseval_report(ctree_true, ctree_pred,
                          digits=4))
    # detailed report on S+N+R
    print(parseval_detailed_report(ctree_true, ctree_pred,
                                   metric_type='S+R'))

    return skipped_docs
