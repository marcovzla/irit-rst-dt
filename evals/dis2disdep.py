"""Convert RST trees to their dependency version (.dis to .dis_dep).

TODO
----
* [ ] support intra-sentential level document parsing ; required to score
      Joty's .sen_dis files

"""
from __future__ import absolute_import, print_function
import argparse
from collections import defaultdict
from glob import glob
import os

from educe.annotation import Span
from educe.corpus import FileId
from educe.learning.disdep_format import dump_disdep_files
from educe.rst_dt.annotation import Node, RSTTree
from educe.rst_dt.codra import load_codra_output_files
from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.feng import load_feng_output_files
from educe.rst_dt.rst_wsj_corpus import (DOUBLE_FOLDER, TEST_FOLDER,
                                         TRAIN_FOLDER)


# original RST corpus
RST_CORPUS = os.path.join('/home/mmorey/corpora/rst_discourse_treebank/data')
RST_MAIN_TRAIN = os.path.join(RST_CORPUS, TRAIN_FOLDER)
RST_MAIN_TEST = os.path.join(RST_CORPUS, TEST_FOLDER)
RST_DOUBLE = os.path.join(RST_CORPUS, DOUBLE_FOLDER)
# output of Joty's parser
OUT_JOTY = os.path.join('/home/mmorey/melodi/rst/joty/Doc-level/')
# output of Feng & Hirst's parser
OUT_FENG = os.path.join('/home/mmorey/melodi/rst/feng_hirst/tmp/')
# output of Ji's parser
OUT_JI = os.path.join('/home/mmorey/melodi/rst/ji_eisenstein/test_input')


def main():
    """Main"""
    parser = argparse.ArgumentParser(
        description='Convert .dis files to .dis_dep'
    )
    parser.add_argument('--nary_enc', default='chain',
                        choices=['chain', 'tree'],
                        help="Encoding for n-ary nodes")
    parser.add_argument('--author', default='gold',
                        choices=['gold', 'silver', 'joty', 'feng', 'ji'],
                        help="Author of the version of the corpus")
    parser.add_argument('--split', default='test',
                        choices=['train', 'test', 'double'],
                        help="Relevant part of the corpus")
    parser.add_argument('--out_root', default='TMP_disdep',
                        help="Root directory for the output")
    args = parser.parse_args()
    # precise output path, by default: TMP_disdep/chain/gold/train
    out_dir = os.path.join(args.out_root, args.nary_enc, args.author, args.split)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # read RST trees
    nary_enc = args.nary_enc
    author = args.author
    corpus_split = args.split

    if author == 'gold':
        if corpus_split == 'train':
            corpus_dir = RST_MAIN_TRAIN
        elif corpus_split == 'test':
            corpus_dir = RST_MAIN_TEST
        elif corpus_split == 'double':
            raise NotImplementedError("Gold trees for 'double'")
        reader = Reader(corpus_dir)
        rtrees = reader.slurp()
        dtrees = {doc_name: RstDepTree.from_rst_tree(rtree, nary_enc=nary_enc)
                  for doc_name, rtree in rtrees.items()}
    elif author == 'silver':
        if corpus_split == 'double':
            corpus_dir = RST_DOUBLE
        else:
            raise ValueError("'silver' annotation is available for the "
                             "'double' split only")
    elif author == 'joty':
        if corpus_split != 'test':
            raise ValueError("The output of Joty's parser is available for "
                             "the 'test' split only")
        data_pred = load_codra_output_files(OUT_JOTY, level='doc')
        doc_names = data_pred['doc_names']
        rtrees = data_pred['rst_ctrees']
        dtrees = {doc_name: RstDepTree.from_rst_tree(rtree, nary_enc=nary_enc)
                  for doc_name, rtree in zip(doc_names, rtrees)}
        # set reference to the document in the RstDepTree (required by
        # dump_disdep_files)
        for doc_name, dtree in dtrees.items():
            dtree.origin = FileId(doc_name, None, None, None)
    elif author == 'feng':
        if corpus_split != 'test':
            raise ValueError("The output of Feng & Hirst's parser is "
                             "available for the 'test' split only")
        data_pred = load_feng_output_files(OUT_FENG)
        doc_names = data_pred['doc_names']
        rtrees = data_pred['rst_ctrees']
        dtrees = {doc_name: RstDepTree.from_rst_tree(rtree, nary_enc=nary_enc)
                  for doc_name, rtree in zip(doc_names, rtrees)}
        # set reference to the document in the RstDepTree (required by
        # dump_disdep_files)
        for doc_name, dtree in dtrees.items():
            dtree.origin = FileId(doc_name, None, None, None)

    elif author == 'ji':
        if corpus_split != 'test':
            raise ValueError("The output of Ji & Eisenstein's parser is "
                             "available for the 'test' split only")
        # * load the text of the EDUs
        # FIXME get the text of EDUs from the .merge files
        corpus_dir = RST_MAIN_TEST
        reader_true = Reader(corpus_dir)
        ctree_true = reader_true.slurp()
        doc_edus = {k.doc: ct_true.leaves() for k, ct_true
                    in ctree_true.items()}
        # * for each doc, load the predicted spans from the .brackets
        ctree_pred = dict()
        files_pred = os.path.join(OUT_JI, '*.brackets')
        for f_pred in sorted(glob(files_pred)):
            doc_name = os.path.splitext(os.path.basename(f_pred))[0]
            edus = {i: e for i, e in enumerate(doc_edus[doc_name], start=1)}
            origin = FileId(doc_name, None, None, None)
            # read spans
            spans_pred = defaultdict(list)  # predicted spans by length
            with open(f_pred) as f:
                for line in f:
                    # FIXME use a standard module: ast or pickle?
                    # drop surrounding brackets + opening bracket of edu span
                    line = line.strip()[2:-1]
                    edu_span, nuc_rel = line.split('), ')
                    edu_span = tuple(int(x) for x in edu_span.split(', '))
                    nuc, rel = nuc_rel.split(', ')
                    edu_span_len = edu_span[1] - edu_span[0]
                    spans_pred[edu_span_len].append((edu_span, nuc, rel))
            # bottom-up construction of the RST ctree
            # left_border -> list of RST ctree fragments, sorted by len
            tree_frags = defaultdict(list)
            for span_len, spans in sorted(spans_pred.items()):
                for edu_span, nuc, rel in spans:
                    children = []
                    edu_beg, edu_end = edu_span
                    if edu_beg == edu_end:
                        # leaf node
                        txt_span = edus[edu_beg].span
                    else:
                        # internal node
                        # * get the children (subtrees)
                        edu_cur = edu_beg
                        while edu_cur < edu_end:
                            kid_nxt = tree_frags[edu_cur][-1]
                            children.append(kid_nxt)
                            edu_cur = kid_nxt.label().edu_span[1] + 1
                        # compute properties of this node
                        txt_span = Span(children[0].label().span.char_start,
                                        children[-1].label().span.char_end)
                    # build node and RSTTree fragment
                    node = Node(nuc, edu_span, txt_span, rel,
                                context=None)  # TODO context?
                    tree_frags[edu_beg].append(
                        RSTTree(node, children, origin=origin))
            # build the top node
            edu_nums = sorted(edus.keys())
            edu_span = (edu_nums[0], edu_nums[-1])
            print(doc_name, edu_span)
            children = []
            edu_beg, edu_end = edu_span
            edu_cur = edu_beg
            while edu_cur < edu_end:
                print(edu_cur)
                kid_nxt = tree_frags[edu_cur][-1]
                children.append(kid_nxt)
                edu_cur = kid_nxt.label().edu_span[1] + 1
            txt_span = Span(children[0].label().span.char_start,
                            children[-1].label().span.char_end)
            node = Node(nuc, edu_span, txt_span, 'Root', context=None)
            tree_frags[edu_beg].append(
                RSTTree(node, children, origin=origin))
            # now we should have a spanning ctree
            ct_pred = tree_frags[1][-1]
            # DEBUG
            print(sorted(edus.keys())[0],
                  sorted(edus.keys())[-1])
            print(ct_pred.label().edu_span)  # RESUME HERE
            print(sorted(tree_frags.items()))
            # end DEBUG
            assert ct_pred.label().edu_span == (sorted(edus.keys())[0],
                                                sorted(edus.keys())[-1])
            ctree_pred[doc_name] = ct_pred
            
        raise NotImplementedError("Output of Ji's parser")
    # do dump
    dump_disdep_files(dtrees.values(), out_dir)


if __name__ == '__main__':
    main()
