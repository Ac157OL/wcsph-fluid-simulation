import warp as wp
import numpy as np
from typing import Any, List, Tuple, Dict, Union

INF_SMALL = wp.constant(wp.float32(1e-6))
WC_GAMMA = wp.constant(7.0)
DT_WEIGHT = wp.constant(1000)