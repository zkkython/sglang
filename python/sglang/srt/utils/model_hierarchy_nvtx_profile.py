import gzip
import json
import logging
import os
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)
_DUMMY_CONTEXT = nullcontext()
_nvtx_events = deque()
_tmp_events = deque()

try:
    import nvtx

    _has_nvtx = True
except ImportError:
    _has_nvtx = False


class NVTXConfig:
    def __init__(self):
        self._nvtx_enable = False
        self._sync_cuda = False

    def update_state(self, nvtx_enable, sync_cuda):
        self._nvtx_enable = nvtx_enable
        self._sync_cuda = sync_cuda

    @property
    def nvtx_enable(self):
        return self._nvtx_enable

    @property
    def sync_cuda(self):
        return self._sync_cuda


nvtx_config = NVTXConfig()


@dataclass
class NvtxEvent:
    name: str = ""
    st: float = 0.0
    et: float = 0.0


class NvtxRange:
    @staticmethod
    def push(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> None:

        if _has_nvtx:
            nvtx.push_range(message, color, domain, category, payload)
        else:
            torch.cuda.nvtx.range_push(message)

        if not any(event.name == message for event in _tmp_events):
            _tmp_events.append(NvtxEvent(name=message, st=time.time() * 1000000.0))
        else:
            logger.error(f"nvtx error: {message} already exists in temporary events")

    @staticmethod
    def pop(domain: Optional[str] = None) -> None:

        if not nvtx_config.nvtx_enable:
            return

        if nvtx_config.sync_cuda:
            torch.cuda.synchronize()

        if _has_nvtx:
            nvtx.pop_range(domain)
        else:
            torch.cuda.nvtx.range_pop()

        if not _tmp_events:
            logger.error("nvtx error: no start event to pop")
            return

        event = _tmp_events.pop()
        event.et = time.time() * 1000000.0

        _nvtx_events.append(
            {
                "cat": "user_nvtx_annotation",
                "name": event.name,
                "ph": "X",
                "s": "t",
                "pid": os.getpid(),
                "tid": 1,
                "ts": event.st,
                "dur": event.et - event.st,
            }
        )

    @staticmethod
    def clear_all_events() -> None:
        """clear the queue"""
        _tmp_events.clear()
        _nvtx_events.clear()


class custom_nvtx_annotate:
    """
    Annotate code ranges using a context manager or a decorator.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        payload: Optional[Any] = None,
    ):
        """
        Annotate a code range.
        """
        self.message = message
        self.color = color
        self.domain = domain
        self.category = category
        self.payload = payload
        self._enabled = getattr(nvtx_config, "nvtx_enable", False)
        self._sync_cuda = getattr(nvtx_config, "sync_cuda", False)

    def __enter__(self):
        if not self._enabled:
            return _DUMMY_CONTEXT.__enter__()

        if self._sync_cuda:
            torch.cuda.synchronize()
        NvtxRange.push(
            self.message, self.color, self.domain, self.category, self.payload
        )

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._enabled:
            _DUMMY_CONTEXT.__exit__(exc_type, exc_value, traceback)
            return
        if self._sync_cuda:
            torch.cuda.synchronize()
        NvtxRange.pop(self.domain)

    def __call__(self, func):
        if not self._enabled:
            return func

        @wraps(func)
        def inner(*args, **kwargs):

            if self._sync_cuda:
                torch.cuda.synchronize()
            NvtxRange.push(self.message, self.color, self.domain, self.category)
            try:
                result = func(*args, **kwargs)
            finally:
                if self._sync_cuda:
                    torch.cuda.synchronize()
                NvtxRange.pop(self.domain)
            return result

        return inner


class ModelRunnerNvtxHook:

    def __init__(self):
        self.hook_handles = []
        # save module map info for forward hook
        self._name_modules = {}

    def nvtx_forward_pre_hook(self, module, _args):
        # `_args` is provided by PyTorch forward pre-hook API but intentionally unused.
        _name = (
            f"{self._name_modules[id(module)]}"
            if id(module) in self._name_modules
            else f"{module.__class__.__name__}"
        )
        NvtxRange.push(_name)

    def nvtx_forward_hook(self, module, _args, _output):
        # `_args` and `_output` are provided by PyTorch forward hook API but intentionally unused.
        NvtxRange.pop()

    def enable_nvtx(self, model: torch.nn.Module):
        def get_submodules(m):
            result = []
            for name, module in m.named_children():
                if isinstance(module, torch.nn.ModuleList):
                    for i, item in enumerate(module):
                        result.append([i, item])
                else:
                    result.append([name, module])
            return result

        modules = get_submodules(model)

        while len(modules) != 0:
            childs = []
            for item in modules:
                name = item[0]
                module = item[1]
                if id(module) in self._name_modules:
                    continue
                self._name_modules[id(module)] = name
                self.hook_handles.append(
                    module.register_forward_pre_hook(self.nvtx_forward_pre_hook)
                )
                self.hook_handles.append(
                    module.register_forward_hook(self.nvtx_forward_hook)
                )

                childs.extend(get_submodules(module))

            modules.clear()
            modules = childs

    def disable_nvtx(self):
        for handle in self.hook_handles:
            handle.remove()

        self._name_modules.clear()

    @staticmethod
    def shift_of_nvtx(profiler_path: str):
        """
        Try to compute a timestamp shift so NVTX events align with
        the profiler trace. We read the profiler trace we just
        exported and find its minimum 'ts' value, then compute
        shift = profiler_min_ts - nvtx_min_ts and apply it when
        flushing NVTX events.
        """

        global _nvtx_events
        try:
            merged_start_time = 0.0
            # 1. align nvtx timestamp with profiler timestamp

            with gzip.open(profiler_path, "rt") as f:
                profiler_data = json.load(f)
                nvtx_time = 0.0
                for event in profiler_data["traceEvents"]:
                    if (
                        event["name"].startswith("nvtx")
                        and event["name"].endswith("push_range")
                    ) or (
                        event["name"].startswith("torch/cuda/nvtx")
                        and event["name"].endswith("range_push")
                    ):
                        nvtx_time = event["ts"] + event["dur"]
                        merged_start_time = nvtx_time
                        continue
                    if nvtx_time > 1e-06:
                        if event["ts"] > nvtx_time:
                            break
                        if event["name"].startswith("nvtx") or event["name"].startswith(
                            "torch/cuda/nvtx"
                        ):
                            merged_start_time = event["ts"] + event["dur"]
            if merged_start_time < 1e-06:
                return 0, 0

            nvtx_start_time = 0
            if _nvtx_events:
                nvtx_start_time = min(e.get("ts", 0.0) for e in _nvtx_events)
            return merged_start_time, float(merged_start_time - nvtx_start_time)
        except Exception as e:
            logger.error(f"Error when parsing profiler file {profiler_path}: {e}")
            return 0, 0

    @staticmethod
    def flush_nvtx(
        output_name="",
        ts_shift_us: float = 0.0,
        nvtx_thread_name: str = "nvtx_thread",
        profiler_start_time=0,
    ):
        """
        flush nvtx cache data, support .gz,  match torch profiler output format
        """
        global _nvtx_events, _tmp_events
        if len(_nvtx_events) == 0:
            logger.warning("capture none nvtx events.")
            return False

        # Optionally apply a timestamp shift
        if ts_shift_us:
            shifted_events = []
            for ev in _nvtx_events:
                ev_copy = ev.copy()
                ev_copy["ts"] = float(ev_copy.get("ts", 0.0) + ts_shift_us)
                shifted_events.append(ev_copy)
            trace_events = shifted_events
        else:
            trace_events = list(_nvtx_events)

        thread_name = {
            "name": "thread_name",
            "ph": "M",
            "ts": profiler_start_time,
            "pid": os.getpid(),
            "tid": 1,
            "args": {"name": nvtx_thread_name},
        }
        thread_sort_index = {
            "name": "thread_sort_index",
            "ph": "M",
            "ts": profiler_start_time,
            "pid": os.getpid(),
            "tid": 1,
            "args": {"sort_index": 0},
        }
        trace_events.append(thread_name)
        trace_events.append(thread_sort_index)

        output = {
            "displayTimeUnit": "ms",
            "traceEvents": trace_events,
        }

        if output_name.endswith(".gz"):
            fp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            try:
                with open(fp.name, "w") as f:
                    json.dump(output, f)
                with open(fp.name, "rb") as fin:
                    with gzip.open(output_name, "wb") as fout:
                        fout.writelines(fin)
            finally:
                os.remove(fp.name)
        else:
            with open(output_name, "w") as f:
                json.dump(output, f)

        NvtxRange.clear_all_events()
        return True
