""" TEst for the diffusion model module."""

import numpy as np
import math
import pytest
from diffusion import calculate_stable_time_step, step_like

def test_time_step_is_float():
    time_step = calculate_stable_time_step(1,1)
    assert isinstance(time_step,float)

def test_time_step_with_zero_spacing():
    dt = calculate_stable_time_step(dx = 0.0, diffusivity = 1)
    assert dt == pytest.approx(0.0)
    assert math.isclose(dt, 0.0)

def test_step_like_length():
    z = step_like(np.arange(100))
    assert len(z) == 100