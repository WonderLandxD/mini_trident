from __future__ import annotations

from typing import Dict, List, Type

from .gpfm_models.gpfm import GPFM
from .h0_mini_models.h0_mini import H0_Mini
from .h_optimus_models.h_optimus_0 import H_optimus_0
from .h_optimus_models.h_optimus_1 import H_optimus_1
from .lunit_models.lunit_p16 import Lunit_P16
from .lunit_models.lunit_p8 import Lunit_P8
from .pathorchestra_models.pathorchestra import PathOrchestra
from .prov_gigapath_models.prov_gigapath import ProvGigaPath
from .stainnet_models.stainnet_base import StainNet_Base
from .stainnet_models.stainnet_small import StainNet_Small
from .uni_models.uni_v1 import UNI_v1
from .uni_models.uni_v2 import UNI_v2
from .virchow_models.virchow_1 import Virchow_1
from .virchow_models.virchow_2 import Virchow_2

MODEL_REGISTRY: Dict[str, Type] = {
    "gpfm": GPFM,
    "h0_mini": H0_Mini,
    "h_optimus_0": H_optimus_0,
    "h_optimus_1": H_optimus_1,
    "lunit_p8": Lunit_P8,
    "lunit_p16": Lunit_P16,
    "pathorchestra": PathOrchestra,
    "prov_gigapath": ProvGigaPath,
    "stainnet_small": StainNet_Small,
    "stainnet_base": StainNet_Base,
    "uni_v1": UNI_v1,
    "uni_v2": UNI_v2,
    "virchow_1": Virchow_1,
    "virchow_2": Virchow_2,
}


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


def create_model(name: str):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown patch encoder: {name}. Available: {', '.join(list_models())}")
    return MODEL_REGISTRY[key]()
