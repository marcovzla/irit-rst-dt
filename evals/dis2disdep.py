"""Convert RST trees to their dependency version (.dis to .dis_dep).

TODO
----
* [ ] support the output of Ji & Eisenstein's parser ; need to convert
      .brackets to .dis_dep (via .dis?)
* [ ] support intra-sentential level document parsing ; required to score
      Joty's .sen_dis files

"""
from __future__ import absolute_import, print_function
import argparse
import os

from educe.corpus import FileId
from educe.learning.disdep_format import dump_disdep_files
from educe.rst_dt.codra import load_codra_output_files
from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree
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
        # files_glob = os.path.join(OUT_FENG, '*.txt.dis')  # FIXME
        raise NotImplementedError("Output of Feng's parser")
    elif author == 'ji':
        raise NotImplementedError("Output of Ji's parser")
    # do dump
    dump_disdep_files(dtrees.values(), out_dir)


if __name__ == '__main__':
    main()
