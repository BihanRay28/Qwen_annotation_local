import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cue_annotation.config import Config
from cue_annotation.qwen_backend import FatalGPUError, QwenBackend, SplitRequired, model_identity


class FakeOOM(RuntimeError):
    pass


class Batch(dict):
    def to(self, device):
        return self


class Processor:
    def __init__(self, length):
        self.length = length

    def apply_chat_template(self, messages, **kwargs):
        return "Synthetic processor contract"

    def __call__(self, **kwargs):
        return Batch(input_ids=np.zeros((1, self.length), dtype=np.int64))

    def batch_decode(self, generated, **kwargs):
        return ['{"frames":[]}']


def backend(length=4, mode="normal"):
    result = QwenBackend.__new__(QwenBackend)
    result.config = Config(input_token_budget=100, max_new_tokens=10)
    result.processor = Processor(length)
    result.process_vision_info = lambda messages: ([], None)
    result.last_metrics = {}
    result.torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            OutOfMemoryError=FakeOOM,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 100,
        ),
    )

    def generate(**kwargs):
        if mode == "oom":
            raise FakeOOM("synthetic out of memory")
        if mode == "fatal":
            raise RuntimeError("CUDA illegal memory access")
        tokens = np.ones((1, length + 10), dtype=np.int64)
        if mode != "truncated":
            tokens[0, -1] = 2
        return tokens

    result.model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=2))
    return result


class BackendContractTests(unittest.TestCase):
    def test_actual_encoded_length_controls_split_before_inference(self):
        with self.assertRaises(SplitRequired):
            backend(length=101).generate([])

    def test_generation_limit_without_eos_is_not_accepted(self):
        with self.assertRaises(SplitRequired):
            backend(mode="truncated").generate([])
        self.assertEqual(backend().generate([]), '{"frames":[]}')

    def test_oom_is_retryable_but_device_failure_is_fatal(self):
        with self.assertRaises(SplitRequired):
            backend(mode="oom").generate([])
        with self.assertRaises(FatalGPUError):
            backend(mode="fatal").generate([])

    def test_local_3b_checkpoint_cannot_be_labelled_7b(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2_5_vl",
                        "hidden_size": 2048,
                        "num_hidden_layers": 36,
                        "max_position_embeddings": 128000,
                    }
                )
            )
            (root / "model.safetensors").write_bytes(b"synthetic checksum fixture, not model weights")
            with self.assertRaises(ValueError):
                model_identity(root, Config())
            identity = model_identity(root, Config(model_id="Qwen/Qwen2.5-VL-3B-Instruct"))
            self.assertEqual(identity["model_id"], "Qwen/Qwen2.5-VL-3B-Instruct")


if __name__ == "__main__":
    unittest.main()
