"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any, Dict, List, Optional, Tuple

from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict, get_bliss_corpus_dict, get_ogg_zip_dict, get_bliss_lexicon

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream


from .common import get_zip, DATA_PREFIX, build_training_datasets, TrainingDatasets, TrainingDatasetSettings
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT



def get_eow_lexicon(librispeech_key: str, with_g2p=True) -> tk.Path:

    """
    get the g2p bliss lexicon with EOW tokens added
    :return:
    """
    if with_g2p:
        lex = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False, add_silence=False)[librispeech_key]
    else:
        lex = get_bliss_lexicon(use_stress_marker=False, add_silence=False)

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_eow_bliss(librispeech_key: str, train_librispeech_key:str, remove_unk_seqs=False) -> tk.Path:
    """
    get an EOW modified corpus with optional unknown removed for cross validation

    :param corpus_key: train, dev, test
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="ogg")[librispeech_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob
        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            bliss_lexicon=get_eow_lexicon(librispeech_key=train_librispeech_key, with_g2p=True),  # cv may include words from g2p
            all_unknown=False
        ).out_corpus

    # default train lexicon
    lexicon = get_eow_lexicon(librispeech_key=train_librispeech_key, with_g2p=True)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus


def get_eow_bliss_and_zip(librispeech_key: str, train_librispeech_key: str, remove_unk_seqs=False):
    """
    :param corpus_key: e.g. "train", "dev", or "test,
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_eow_bliss(librispeech_key=librispeech_key, train_librispeech_key=train_librispeech_key, remove_unk_seqs=remove_unk_seqs)
    zip_dataset = get_zip(f"{train_librispeech_key}_{librispeech_key}_filtered_eow", bliss_dataset=bliss_dataset)

    return bliss_dataset, zip_dataset


def get_eow_vocab_datastream(librispeech_key: str) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream for Tedlium-2

    :param with_blank: datastream for CTC training
    """
    lexicon = get_eow_lexicon(librispeech_key=librispeech_key)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, f"{librispeech_key}", "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_text_lexicon(librispeech_key: str) -> tk.Path:
    """

    :return:
    """
    bliss_lex = get_eow_lexicon(librispeech_key=librispeech_key, with_g2p=False)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon = BlissLexiconToWordLexicon(bliss_lex).out_lexicon
    return word_lexicon


def build_eow_phon_training_datasets(
        librispeech_key: str,
        settings: TrainingDatasetSettings,
        real_data_weight: int = 1,
        extra_bliss: Optional[List[tk.Path]] = None,
        lexicon_librispeech_key: Optional[str] = None,
        random_merge_extra_bliss=False,
) -> TrainingDatasets:
    """
    :param librispeech_key: which librispeech corpus to use
    :param settings: configuration object for the dataset pipeline
    :param real_data_weight: how often to repeat the original data (e.g. to match length of synthetic)
    :param extra_bliss: extra data (e.g. synthetic) to train with
    :param lexicon_librispeech_key: if we are using extra synthetic data, we might a lexicon with the OOV coverage of that data as well
    :param random_merge_extra_bliss: assuming all extras are the same dataset, perform equal random splitting
        This will create new OggZip files and waste some space, but as all extras might have identical
        sequence names controlling via segment lists only is difficult
    """
    label_datastream = get_eow_vocab_datastream(librispeech_key=lexicon_librispeech_key or librispeech_key)

    _, train_ogg = get_eow_bliss_and_zip(librispeech_key=librispeech_key, train_librispeech_key=librispeech_key, remove_unk_seqs=False)
    _, dev_clean_ogg = get_eow_bliss_and_zip(librispeech_key="dev-clean", train_librispeech_key=librispeech_key, remove_unk_seqs=True)
    _, dev_other_ogg = get_eow_bliss_and_zip(librispeech_key="dev-other", train_librispeech_key=librispeech_key, remove_unk_seqs=True)

    if extra_bliss:
        # because we are working with phonemes, we need to take the synthetic bliss, convert the text to phonemes
        # and build the zip here, this is annoying but caused by unfortunate design decisions
        lexicon = get_eow_lexicon(librispeech_key=lexicon_librispeech_key or librispeech_key)
        extra_zips = []

        if random_merge_extra_bliss:
            from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
            segment_file = SegmentCorpusJob(bliss_corpus=extra_bliss[0], num_segments=1).out_single_segment_files[1]
            split_ratio = 1.0/len(extra_bliss)
            segment_split_segments = ShuffleAndSplitSegmentsJob(
                segment_file=segment_file,
                split={k: split_ratio for k in range(len(extra_bliss))}
            ).out_segments
            segment_split_segments = [segment_split_segments[k] for k in range(len(extra_bliss))]  # convert dict to list

        for i, bliss in enumerate(extra_bliss):
            converted_bliss = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus
            if random_merge_extra_bliss:
                from i6_core.corpus.filter import FilterCorpusBySegmentsJob
                converted_bliss = FilterCorpusBySegmentsJob(
                    bliss_corpus=converted_bliss,
                    segment_file=segment_split_segments[i],
                    compressed=True,
                    delete_empty_recordings=True
                ).out_corpus
            ogg_zip = BlissToOggZipJob(
                bliss_corpus=converted_bliss,
                no_conversion=True,
                returnn_python_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT
            ).out_ogg_zip
            extra_zips.append(ogg_zip)
        train_oggs = [train_ogg] * real_data_weight + extra_zips
    else:
        train_oggs = train_ogg

    return build_training_datasets(
        train_ogg=train_oggs,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream
    )