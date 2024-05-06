from dataclasses import dataclass
import dataclasses

from fairseq.models.wav2vec.wav2vec2_asr import Wav2Vec2CtcConfig
from fairseq.dataclass.configs import FairseqConfig
from fairseq.tasks.audio_finetuning import AudioFinetuningTask, AudioFinetuningConfig
from fairseq.tasks import FairseqTask
from fairseq.data import Dictionary

from omegaconf import OmegaConf

@dataclass
class ModelConfig:
    """
    Model configuration for Wav2Vec2CTC Wrapper Model.

    Args:
        model_config_updates (dict): Updates to the Wav2Vec2CtcConfig
            (see fairseq.models.wav2vec.wav2vec2_asr.Wav2Vec2CtcConfig)
        task_config_updates: dict: Updates to the AudioFinetuningConfig
            (see fairseq.tasks.audio_finetuning.AudioFinetuningConfig)
        label_target_size: int: Size of the target dictionary.
    """
    model_config_updates: dict = dataclasses.field(default_factory=dict)
    task_config_updates: dict = dataclasses.field(default_factory=dict)
    label_target_size: int = 0

    def build_full_config(self) -> FairseqConfig:
        """
        Build a full FairseqConfig object from the model and task configurations updates.
        Non-specified fields will be filled with the default values.

        FairseqConfig is needed to instantiate a FairseqModel.
        """
        fairseq_model_config = self._build_model_config()
        fairseq_task_config = self._build_task_config()
        fairseq_conf = FairseqConfig()
        setattr(fairseq_conf, 'model', fairseq_model_config)
        setattr(fairseq_conf, 'task', fairseq_task_config)

        # convert to OmegaConf to resolve interpolations in the default config fields
        return OmegaConf.structured(fairseq_conf)

    def _build_model_config(self) -> Wav2Vec2CtcConfig:
        model_cfg = Wav2Vec2CtcConfig()
        return self._update_config(model_cfg, self.model_config_updates)

    def _build_task_config(self) -> AudioFinetuningConfig:
        task_cfg = AudioFinetuningConfig()
        return self._update_config(task_cfg, self.task_config_updates)

    def _update_config(self, cfg, update_dict):
        for k, v in update_dict.items():
            setattr(cfg, k, v)
        return cfg

    def build_dummy_task(self) -> FairseqTask:
        """
        Build a dummy task with a target dictionary of size `label_target_size`.
        This is needed to instantiate the Wav2Vec2Ctc model.
        """
        task_cfg = AudioFinetuningConfig()
        task = AudioFinetuningTask(task_cfg)

        def _build_dummy_target_dict() -> Dictionary:
            target_dict = Dictionary()
            for i in range(self.label_target_size):
                target_dict.add_symbol(f'{i}')
            return target_dict

        task.state.add_factory('target_dictionary', _build_dummy_target_dict)
        return task
    
    @classmethod
    def from_dict(cls, cfg: dict):
        ret = cls(**cfg)
        print(ret)
        return ret
        

