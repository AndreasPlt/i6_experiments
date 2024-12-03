"""
Defines the data inputs for any RASR based LibriSpeech task
"""
from dataclasses import dataclass
from typing import Dict
from i6_core.meta import CorpusObject
from sisyphus import tk

from i6_experiments.common.datasets.librispeech import (
    get_corpus_object_dict,
    get_g2p_augmented_bliss_lexicon_dict,
    constants,
    get_arpa_lm_dict,
    get_bliss_lexicon,
)
from i6_experiments.common.setups.rasr.util import RasrDataInput

from copy import deepcopy
from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import RemoveSpeakerTagsJob


@dataclass()
class CorpusData:
    """
    Helper class to define all RasrDataInputs to be passed to the `System` class
    """

    train_data: Dict[str, RasrDataInput]
    dev_data: Dict[str, RasrDataInput]
    test_data: Dict[str, RasrDataInput]


def get_corpus_data_inputs():
    """
    :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
    """
    # lexica and lm definition
    g2p_lexica = get_g2p_augmented_bliss_lexicon_dict(
        output_prefix="corpora",
        add_unknown_phoneme_and_mapping=False,
        use_stress_marker=False,
    )
    train_lexicon = {  # lexicon for training
        "filename": g2p_lexica["train-clean-100"],
        "normalize_pronunciation": False,
    }
    test_lexicon = {  # lexicon for decoding/alignment
        "filename": g2p_lexica["train-other-960"],
        "normalize_pronunciation": False,
    }
    lexicon = {  # lexicon for decoding
        "filename": get_bliss_lexicon(
            use_stress_marker=False,
            add_unknown_phoneme_and_mapping=False,
        ),
        "normalize_pronunciation": False,
    }
    lm = {  # language model for decoding
        "filename": get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }

    # define all used corpora
    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

    train_data_inputs["train-clean-100"] = RasrDataInput(
        corpus_object=corpus_object_dict["train-clean-100"],
        concurrent=constants.concurrent["train-clean-100"],
        lexicon=train_lexicon,
        lm=None,
    )
    for dev_key in ["dev-clean", "dev-other"]:
        dev_data_inputs[dev_key] = RasrDataInput(
        corpus_object=corpus_object_dict[dev_key],
        concurrent=constants.concurrent[dev_key],
        lexicon=lexicon,
        lm=lm,
    )
    test_data_inputs["train-other-960"] = RasrDataInput(
        corpus_object=corpus_object_dict["train-other-960"],
        concurrent=constants.concurrent["train-other-960"],
        lexicon=test_lexicon,
        lm=lm,
    )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
