"""Generate .edus files for Feng's gCRF parser, with gold EDUs.

"""

from __future__ import absolute_import, print_function

import argparse
from difflib import SequenceMatcher
from glob import glob
import os

import numpy as np

TXT_MAP = [
    (' .', '.'),
    (' ,', ','),
    (' %', '%'),
    (' :', ':'),
    ('-LRB-', '('),
    ('-RRB-', ')'),
    # non-breaking space
    # FIXME switch to unicode where this is a unique char: u"\u00A0"
    ('\xc2\xa0', ' '),
    ("do n't", "don't"),
    ('...', '. . .'),
]


def dump_gcrf_edus_gold(f_gold, f_pred, f_dest):
    """Reinject gold segmentation into .edus files output by gCRF.

    Parameters
    ----------
    f_gold: str
        Path to the gold .edus file
    f_pred: str
        Path to the predicted .edus file
    f_dest: str
        Path to the output
    """
    txt_gold = f_gold.read()
    i_gold = 0  # pointer in txt_gold

    skip_toks = 0  # nb of tokens from _pred that have already been consumed

    for line in f_pred:
        tokens_pred = line.split(' ')
        # the newline character (marking the end of sentence) is appended
        # to the last token
        assert tokens_pred[-1][-1] == '\n'
        #
        for i, tok in enumerate(tokens_pred):
            if skip_toks:
                # skip tokens from _pred that have already been consumed
                skip_toks -= 1
                continue

            while txt_gold[i_gold] == ' ':
                # skip whitespaces in gold
                i_gold += 1

            if (tok[0] == '.' and tokens_pred[i - 1][-1] == '.'
                and txt_gold[i_gold] != '.'):
                # preprocessing adds an extra full stop when the last
                # token ends with one (e.g. for abbreviations:
                # "Inc." => "Inc. .")
                if len(tok) > 1:
                    # skip extra stop, resume normal matching procedure
                    tok = tok[1:]
                else:
                    # token is exactly '.' => skip it
                    continue

            if tok == 'EDU_BREAK':
                # predicted EDU break inside sentence
                if txt_gold[i_gold] == '\n':
                    # also in gold => correctly predicted => leave it
                    print(tok, end=' ', file=f_dest)
                    i_gold += 1
                    continue
                else:
                    # not in gold => erroneously predicted => delete it
                    # (this is a silent operation)
                    continue
            elif tok == '\n' and txt_gold[i_gold] == '\n':
                # happens when the token before the newline was a copy of
                # the punctuation added by preprocessing, removed above ;
                # ex: "... Inc." => "... Inc. ."
                print(tok, end='', file=f_dest)
                i_gold += 1
                continue

            if txt_gold[i_gold:i_gold + 5] == '\n    ':
                # gold EDU break inside sentence, missing from predicted
                print('EDU_BREAK', end=' ', file=f_dest)  # FIXME to f_dest
                i_gold += 5

            # match token
            # whitespaces inside tokens are non-breaking spaces:
            # \xc2\xa0 in ascii, but we should really be processing
            # them as unicode symbols...
            tok_txt_gold = (tok
                            .replace('\xc2\xa0', ' ')
                            .replace('-LRB-', '(')
                            .replace('-RRB-', ')')
                            .replace('-LCB-', '{')
                            .replace('-RCB-', '}')
                            .replace('``', '"')
                            .replace("''", '"')
                            .replace('...', '. . .')
            )
            if i < len(tokens_pred) - 1:
                # all tokens except for the last of the sentence
                if (txt_gold[i_gold:i_gold + len(tok_txt_gold)]
                    == tok_txt_gold):
                    # it is a match indeed
                    i_gold += len(tok_txt_gold)
                    # print token followed by a whitespace
                    print(tok, end=' ', file=f_dest)  # FIXME to f_dest
                    continue
                else:
                    print()
                    print('wow')
                    print(tokens_pred[i:])
                    print(repr(txt_gold[i_gold:i_gold + len(tok_txt_gold)]),
                          repr(tok))
                    raise ValueError('gni')
            else:
                # last token of the sentence
                if (txt_gold[i_gold:i_gold + len(tok_txt_gold) + 1]
                    == tok_txt_gold[:-1] + ' ' + tok[-1]):
                    # gold has an extra whitespace before the newline
                    i_gold += len(tok_txt_gold) + 1
                    # token but no following whitespace
                    print(tok, end='', file=f_dest)
                elif (txt_gold[i_gold:i_gold + 7] == '. . . .'
                      and tok == '...\n'):
                    # pre-processing replaces '[. . .] [.]' with '...' ;
                    # let's assume it's normal
                    i_gold += 7
                    print(tok, end='', file=f_dest)
                else:
                    print()
                    print('i-2', tokens_pred[i - 2])
                    print('i-1', tokens_pred[i - 1])
                    print('i', tokens_pred[i])
                    print(repr(txt_gold[i_gold:i_gold + len(tok_txt_gold)]),
                          repr(tok))
                    raise ValueError('pouet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate .edus files with gold segmentation')
    parser.add_argument('dir_gold', metavar='DIR',
                        help='folder with the gold files (.edus)')
    parser.add_argument('dir_pred', metavar='DIR',
                        help='folder with the predicted files (.edus)')
    parser.add_argument('dir_dest', metavar='DIR',
                        help='output folder')

    args = parser.parse_args()

    # setup output dir
    if not os.path.exists(args.dir_dest):
        os.makedirs(args.dir_dest)
    
    files_edus_gold = sorted(glob(os.path.join(args.dir_gold, '*.edus')))
    files_edus_pred = sorted(glob(os.path.join(args.dir_pred, '*.edus')))
    for file_gold, file_pred in zip(files_edus_gold, files_edus_pred):
        print(file_gold)
        assert os.path.basename(file_gold) == os.path.basename(file_pred)
        file_dest = os.path.join(args.dir_dest,
                                 os.path.basename(file_pred))

        with open(file_gold) as f_gold:
            with open(file_pred) as f_pred:
                with open(file_dest, mode='w') as f_dest:
                    dump_gcrf_edus_gold(f_gold, f_pred, f_dest)
