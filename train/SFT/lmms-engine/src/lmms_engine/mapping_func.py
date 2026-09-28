from typing import Literal

from transformers import (  # AutoModelForVision2Seq,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMaskedLM,
    PretrainedConfig,
)
from transformers.modeling_utils import PreTrainedModel

DATASET_MAPPING = {}
DATAPROCESSOR_MAPPING = {}

from loguru import logger

try:
    import fla
except ImportError as e:
    logger.warning(f"Failed to import the lib 'fla'. If you do not need it, you can ignore this warning.")


# A decorator class to register processors
def register_processor(processor_type: str):
    def decorator(cls):
        if processor_type in DATAPROCESSOR_MAPPING:
            raise ValueError(f"Processor type {processor_type} is already registered.")
        DATAPROCESSOR_MAPPING[processor_type] = cls
        return cls

    return decorator


# A decorator class to register dataset
def register_dataset(dataset_type: str):
    def decorator(cls):
        if dataset_type in DATASET_MAPPING:
            raise ValueError(f"Dataset type {dataset_type} is already registered.")
        DATASET_MAPPING[dataset_type] = cls
        return cls

    return decorator


AUTO_REGISTER_MODEL_MAPPING = {
    "causal_lm": AutoModelForCausalLM,
    "masked_lm": AutoModelForMaskedLM,
    "image_text_to_text": AutoModelForImageTextToText,
    "general": AutoModel,
}


def register_model(
    model_type: str,
    model_config: PretrainedConfig,
    model_class: PreTrainedModel,
    model_general_type: Literal["causal_lm", "masked_lm", "image_text_to_text", "general"] = "causal_lm",
):
    AutoConfig.register(model_type, model_config, exist_ok=True)
    AUTO_REGISTER_MODEL_MAPPING[model_general_type].register(model_config, model_class)


def create_model_from_pretrained(load_from_pretrained_path):
    # Handle both config object and model name/path
    config = AutoConfig.from_pretrained(load_from_pretrained_path)

    return _select_auto_model_class(config, load_from_pretrained_path)


def _select_auto_model_class(config, model_name_or_path):
    model_mapping_candidates = (
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModel,
    )
    supported_candidates = [
        auto_model
        for auto_model in model_mapping_candidates
        if type(config) in auto_model._model_mapping.keys()
    ]

    # Some unified multimodal configs are also registered as text-only causal
    # LMs. Honor the checkpoint's declared architecture before using a fallback.
    architectures = set(getattr(config, "architectures", None) or [])
    for auto_model in supported_candidates:
        mapped_model = auto_model._model_mapping[type(config)]
        if mapped_model.__name__ in architectures:
            return auto_model

    if (
        AutoModelForImageTextToText in supported_candidates
        and getattr(config, "vision_config", None) is not None
    ):
        return AutoModelForImageTextToText

    if supported_candidates:
        return supported_candidates[0]
    raise ValueError(f"Model {model_name_or_path} is not supported.")


def create_model_from_config(model_type, config):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type in CONFIG_MAPPING:
        config_class = CONFIG_MAPPING[model_type]
        m_config = config_class(**config)
        model_class = _select_auto_model_class(m_config, model_type)
    else:
        raise ValueError(f"Model type '{model_type}' is not found in CONFIG_MAPPING.")
    return model_class, m_config
