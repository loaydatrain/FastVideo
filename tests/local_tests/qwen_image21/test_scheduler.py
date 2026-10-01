# SPDX-License-Identifier: Apache-2.0
"""Independent flow schedule and Euler checks of the reused native scheduler."""

import ast
import functools
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def scheduler(monkeypatch):
    class StripImports(ast.NodeTransformer):

        def visit_ImportFrom(self, node):
            if node.module and node.module.startswith(("fastvideo", "diffusers")):
                return ast.copy_location(ast.Pass(), node)
            return node

    class Config(dict):
        __getattr__ = dict.__getitem__

    def register(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = Config({key: value for key, value in bound.arguments.items() if key != "self"})
            return function(self, *args, **kwargs)

        return wrapped

    module = ModuleType("qwen21_scheduler_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(ConfigMixin=type("ConfigMixin", (), {}), SchedulerMixin=type("SchedulerMixin", (), {}),
                           BaseScheduler=type("BaseScheduler", (), {}), BaseOutput=type("BaseOutput", (), {}),
                           register_to_config=register, init_logger=lambda _name: SimpleNamespace(warning=lambda *_a: None))
    path = ROOT / "fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py"
    tree = StripImports().visit(ast.parse(path.read_text()))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift_terminal=.02,
                                                  base_image_seq_len=256, max_image_seq_len=8192,
                                                  base_shift=.5, max_shift=.9)


@pytest.mark.parametrize("tokens", [4096, 16384])
def test_qwen_dynamic_target_schedule_and_terminal_zero(scheduler, tokens):
    mu = .5 + (.9 - .5) * (tokens - 256) / (8192 - 256)
    sigmas = np.linspace(1, 1 / 40, 40).astype(np.float32)
    shifted = np.exp(mu) / (np.exp(mu) + (1 / sigmas - 1))
    terminal = 1 - (1 - shifted) / ((1 - shifted[-1]) / (1 - .02))
    scheduler.set_timesteps(sigmas=sigmas.tolist(), device="cpu", mu=mu)
    torch.testing.assert_close(scheduler.timesteps, torch.tensor(terminal * 1000, dtype=torch.float32),
                               atol=1e-4, rtol=1e-6)
    assert scheduler.sigmas[-1].item() == 0
    assert abs(scheduler.sigmas[-2].item() - .02) < 1e-6


def test_euler_runs_in_fp32_then_restores_prediction_dtype(scheduler):
    scheduler.set_timesteps(sigmas=[1, .5, .25], device="cpu", mu=.7)
    scheduler.set_begin_index(0)
    sample = torch.randn(1, 8, 4, generator=torch.Generator().manual_seed(21), dtype=torch.bfloat16)
    noise = torch.randn(1, 8, 4, generator=torch.Generator().manual_seed(22), dtype=torch.bfloat16)
    for index, timestep in enumerate(scheduler.timesteps):
        expected = (sample.float() + (scheduler.sigmas[index + 1] - scheduler.sigmas[index]) * noise).to(noise.dtype)
        sample = scheduler.step(noise, timestep, sample, return_dict=False)[0]
        torch.testing.assert_close(sample, expected, atol=0, rtol=0)
