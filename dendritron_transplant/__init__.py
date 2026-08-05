"""CPU-native sparse Dendritron transplant reference implementation."""

from .config import DendritronRecipientConfig, HashMemoryConfig
from .donor import DenseDonorLM, DonorTransplanter
from .falsification import build_phrase_control_banks
from .memory import FrozenPhraseBank, MemoryPayloadBuilder, MemoryPayloads
from .model import DendritronRecipientLM, RecipientOutput
from .runtime import CPUTransplantRuntime
from .vm896 import DefinitionSplit, MacslSolution, ReferenceFrame, RidgeSolution, SpectralRidgeFactory, fit_macsl, fit_nonlinear
from .vivere_macsl import (
    KMeansRouter,
    MacslMap,
    PhraseSplit,
    RidgeMap,
    VivereEncoding,
    ViverePhraseBank,
    fit_macsl_map,
    fit_mlp_map,
    fit_vivere_encoding,
)

__all__ = [
    "CPUTransplantRuntime",
    "fit_nonlinear",
    "fit_macsl",
    "SpectralRidgeFactory",
    "RidgeSolution",
    "ReferenceFrame",
    "MacslSolution",
    "DefinitionSplit",
    "DenseDonorLM",
    "DendritronRecipientConfig",
    "DendritronRecipientLM",
    "DonorTransplanter",
    "build_phrase_control_banks",
    "FrozenPhraseBank",
    "HashMemoryConfig",
    "MemoryPayloadBuilder",
    "MemoryPayloads",
    "RecipientOutput",
    "fit_vivere_encoding",
    "fit_mlp_map",
    "fit_macsl_map",
    "ViverePhraseBank",
    "VivereEncoding",
    "RidgeMap",
    "PhraseSplit",
    "MacslMap",
    "KMeansRouter",
]
