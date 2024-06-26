__all__ = ["CreateBPELexiconJob"]

import subprocess as sp
import os
import sys
import xml.etree.ElementTree as ET

from sisyphus import Job, Task, tk, setup_path

from i6_core.lib.lexicon import Lexicon, Lemma
import i6_core.util as util


class CreateBPELexiconJob(Job):
    """
    Create a Bliss lexicon from bpe transcriptions that can be used e.g, for lexicon constrained BPE search.

    DO NOT USE; THIS IS SUPERSEDED BY i6_core.lexicon.bpe.CreateBpeLexiconJob
    """
    __sis_hash_exclude__ = {"keep_special_lemmas": False}

    def __init__(
        self,
        base_lexicon_path: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: tk.Path,
        subword_nmt_repo: tk.Path,
        unk_label: str = "UNK",
        keep_special_lemmas: bool = False,
    ):
        """
        :param base_lexicon_path: base lexicon (can be phoneme based) to take the lemmas from
        :param bpe_codes: bpe codes from the ReturnnTrainBPEJob
        :param bpe_vocab: vocab file to limit which bpe splits can be created
        :param subword_nmt_repo: cloned repository
        :param unk_label:
        :param keep_special_lemmas: If special lemmas should be kept,
            usually yes for RASR search and no for Flashlight search
        """
        self.base_lexicon_path = base_lexicon_path
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = subword_nmt_repo
        self.unk_label = unk_label
        self.keep_special_lemmas = keep_special_lemmas

        self.out_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def _fill_lm_tokens(self, base_lexicon: Lexicon):
        lm_tokens = set()
        special_lemmas = []
        for l in base_lexicon.lemmata:
            if l.special is None:
                for orth in l.orth:
                    lm_tokens.add(orth)
                for token in l.synt or []:  # l.synt can be None
                    lm_tokens.add(token)
                for eval in l.eval:
                    for t in eval:
                        lm_tokens.add(t)
            else:
                special_lemmas.append(l)

        lm_tokens = list(lm_tokens)
        return lm_tokens, special_lemmas

    def _fill_vocab_and_lexicon(self):
        lexicon = Lexicon()
        vocab = set()
        with util.uopen(self.bpe_vocab.get_path(), "rt") as f, util.uopen("fake_count_vocab.txt", "wt") as vocab_file:
            for line in f:
                line = line.strip()
                if line == "{" or line == "}":
                    continue
                if line == "<s>" or line == "</s>":
                    continue
                symbol = line.split(":")[0][1:-1]
                if symbol != self.unk_label:
                    # Fake count vocab filled with -1 so that all merges possible are done
                    vocab_file.write(symbol + " -1\n")
                    symbol = symbol.replace(".", "_")
                    vocab.add(symbol)
                    lexicon.add_phoneme(symbol.replace(".", "_"))

        return vocab, lexicon

    def run(self):
        base_lexicon = Lexicon()
        base_lexicon.load(self.base_lexicon_path)

        lm_tokens, special_lemmas = self._fill_lm_tokens(base_lexicon)

        with util.uopen("words", "wt") as f:
            for t in lm_tokens:
                f.write(f"{t}\n")

        vocab, lexicon = self._fill_vocab_and_lexicon()

        # add special lemmas back to lexicon
        if self.keep_special_lemmas is True:
            for special_lemma in special_lemmas:
                for phon in special_lemma.phon:
                    lexicon.add_phoneme(phon, variation="none")
                lexicon.add_lemma(special_lemma)

        apply_binary = os.path.join(self.subword_nmt_repo.get_path(), "apply_bpe.py")
        args = [
            sys.executable,
            apply_binary,
            "--input",
            "words",
            "--codes",
            self.bpe_codes.get_path(),
            "--vocabulary",
            "fake_count_vocab.txt",
            "--output",
            "bpes",
        ]
        sp.run(args, check=True)

        with util.uopen("bpes", "rt") as f:
            bpe_tokens = [l.strip() for l in f]

        w2b = {w: b for w, b in zip(lm_tokens, bpe_tokens)}

        for w, b in w2b.items():
            b = " ".join([token if token in vocab else self.unk_label for token in b.split()])
            lexicon.add_lemma(Lemma([w], [b.replace(".", "_")]))

        elem = lexicon.to_xml()
        tree = ET.ElementTree(elem)
        util.write_xml(self.out_lexicon.get_path(), tree)
