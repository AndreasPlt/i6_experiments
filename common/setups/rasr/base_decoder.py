__all__ = ["BaseDecoder"]

import copy
from typing import Dict, Optional, Union

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.recognition as recog

from i6_core.meta import CorpusObject


from .util.decode import (
    BaseRecognitionParameters,
    PriorArgs,
    SearchJobArgs,
    Lattice2CtmArgs,
    ScorerArgs,
    OptimizeJobArgs,
)


class BaseDecoder:
    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: "str" = "linux-x86_64-standard",
        compress: bool = False,
        append: bool = False,
        unbuffered: bool = False,
        compress_after_run: bool = True,
        search_job_class: tk.Job = recog.AdvancedTreeSearchJob,
        scorer_job_class: tk.Job = recog.ScliteJob,
        alias_output_prefix: str = "",
    ):
        self.rasr_binary_path = rasr_binary_path
        self.rasr_arch = rasr_arch

        self.search_job_class = search_job_class
        self.scorer_job_class = scorer_job_class

        self.crp = {"base": rasr.CommonRasrParameters()}
        rasr.crp_add_default_output(
            self.crp["base"], compress, append, unbuffered, compress_after_run
        )
        self.crp["base"].set_executables(rasr_binary_path, rasr_arch)

        self.eval_corpora = []

        self.alias_output_prefix = alias_output_prefix

    def init_decoder(
        self,
        *,
        acoustic_model_config: rasr.RasrConfig,
        lexicon_config: rasr.RasrConfig,
        language_model_config: rasr.RasrConfig,
        lm_lookahead_config: rasr.RasrConfig,
        model_combination_config: Optional[rasr.RasrConfig] = None,
        lm_rescoring_config: Optional[rasr.RasrConfig] = None,
    ):
        self.crp["base"].acoustic_model_config = acoustic_model_config
        self.crp["base"].lexicon_config = lexicon_config
        self.crp["base"].language_model_config = language_model_config
        self.crp["base"].lm_lookahead_config = lm_lookahead_config
        self.crp["base"].model_combination_config = model_combination_config
        self.crp["base"].lm_rescoring_config = lm_rescoring_config

    def copy_crp_from_system(self, name: str, crp: rasr.CommonRasrParameters):
        self.crp[name] = copy.deepcopy(crp)

    def init_eval_datasets(self, eval_datasets: Dict[str, CorpusObject]):
        self.eval_corpora.extend([eval_datasets.keys()])
        for corpus_key, corpus_object in eval_datasets.items():
            self.crp[corpus_key] = rasr.CommonRasrParameters(base=self.crp["base"])
            rasr.crp_set_corpus(self.crp[corpus_key], corpus_object)

    def _recog(
        self,
        name: str,
        corpus_key: str,
        search_job_args: SearchJobArgs,
        lat_2_ctm_args: Lattice2CtmArgs,
        scorer_args: ScorerArgs,
    ):
        adv_tree_search_job = self.search_job_class(**search_job_args)
        adv_tree_search_job.set_vis_name(
            f"Recog: {self.alias_output_prefix}{name}. Corpus: {corpus_key}"
        )
        adv_tree_search_job.add_alias(
            f"{self.alias_output_prefix}recog_{corpus_key}/{name}"
        )

        lat_2_ctm_job = recog.LatticeToCtmJob(**lat_2_ctm_args)
        lat_2_ctm_job.add_alias(
            f"{self.alias_output_prefix}lattice2ctm_{corpus_key}/{name}"
        )

        scorer_job = self.scorer_job_class(**scorer_args)
        scorer_job.add_alias(f"{self.alias_output_prefix}scoring_{corpus_key}/{name}")
        tk.register_output(
            f"{self.alias_output_prefix}recog_{corpus_key}/{name}.reports",
            scorer_job.out_report_dir,
        )

    def _optimize_scales(
        self, name: str, corpus_key: str, optimize_args: OptimizeJobArgs
    ) -> (float, float):
        opt_job = recog.OptimizeAMandLMScaleJob(**optimize_args)
        opt_job.add_alias(f"{self.alias_output_prefix}optimize_{corpus_key}/{name}")

        return opt_job.out_best_am_score, opt_job.out_best_lm_score

    def decode(
        self,
        name,
        recognition_parameters: Union[BaseRecognitionParameters, Dict],
        prior: PriorArgs,
        optimize_am_lm_scales: bool = False,
    ):
        for corpus_key in self.eval_corpora:
            self._recog(
                name=name,
                corpus_key=corpus_key,
                search_job_args={},
                lat_2_ctm_args={},
                scorer_args={},
            )
            if optimize_am_lm_scales:
                best_am_scale, best_lm_scale = self._optimize_scales(
                    scorer_cls=self.scorer_job_class,
                    scorer_kwargs={},
                    opt_only_lm_scale=True if best_am_scale == 1.0 else False,
                )
                search_job_args = {}
                self._recog(
                    name=f"{name}-opt",
                    corpus_key=corpus_key,
                    search_job_args={},
                    lat_2_ctm_args={},
                    scorer_args={},
                )

    def get_results(self):
        pass
