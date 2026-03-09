from __future__ import annotations

from types import SimpleNamespace

from intra_model_memval.adapters import GPT2SmallAdapter


class FakeTensor:
    def __init__(self, values, shape):
        self._values = values
        self.shape = shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, _device):
        return self

    def tolist(self):
        return self._values


class FakeTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token_id = None
    pad_token = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True, return_tensors=None):
        values = {"input_ids": [11, 12], "attention_mask": [1, 1]}
        if return_tensors == "pt":
            return {
                "input_ids": FakeTensor([[11, 12]], [1, 2]),
                "attention_mask": FakeTensor([[1, 1]], [1, 2]),
            }
        return values

    @staticmethod
    def convert_ids_to_tokens(token_ids):
        return [f"tok-{token_id}" for token_id in token_ids]


class FakeModel:
    config = SimpleNamespace(n_layer=2, n_head=2, n_embd=4, vocab_size=8, n_positions=32)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(
            logits=FakeTensor([[[0.1, 0.2, 0.8, 0.0]]], [1, 1, 4]),
            hidden_states=(
                FakeTensor([[[0.0, 0.0, 0.0, 0.0]]], [1, 1, 4]),
                FakeTensor([[[1.0, 1.0, 1.0, 1.0]]], [1, 1, 4]),
            ),
            attentions=None,
        )


class FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed_all(_seed):
        return None


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    __version__ = "fake-torch"
    cuda = FakeCuda()

    @staticmethod
    def manual_seed(_seed):
        return None

    @staticmethod
    def inference_mode():
        return FakeInferenceMode()


def test_gpt2_adapter_loads_and_runs_with_mocked_dependencies(monkeypatch):
    monkeypatch.setattr(
        GPT2SmallAdapter,
        "_load_dependencies",
        staticmethod(lambda: (FakeTorch, FakeModel, FakeTokenizer, "fake-transformers")),
    )
    adapter = GPT2SmallAdapter(seed=7)
    adapter.load()

    tokenized = adapter.tokenize("France")
    forward = adapter.forward("France")
    spec = adapter.describe()

    assert tokenized.input_ids == [11, 12]
    assert forward.token_ids == [11, 12]
    assert len(forward.hidden_states) == 2
    assert forward.metadata["seed"] == 7
    assert spec.family == "gpt2"
    assert spec.metadata["n_layer"] == 2
