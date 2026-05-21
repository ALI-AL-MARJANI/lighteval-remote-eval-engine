# MIT License

# Copyright (c) 2025 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from lighteval.models.model_input import GenerationParameters


class TestGenerationParameters:
    @pytest.mark.parametrize(
        "model_args, expected",
        [
            (
                "generation_parameters={temperature: 0.7,top_p: 0.95},pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,max_model_length=32768,gpu_memory_utilisation=0.8",
                {"temperature": 0.7, "top_p": 0.95},
            ),
            (
                "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,generation_parameters={temperature: 0.7,top_p: 0.95},max_model_length=32768,gpu_memory_utilisation=0.8",
                {"temperature": 0.7, "top_p": 0.95},
            ),
            (
                "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,max_model_length=32768,gpu_memory_utilisation=0.8,generation_parameters={temperature: 0.7,top_p: 0.95}",
                {"temperature": 0.7, "top_p": 0.95},
            ),
        ],
    )
    def test_extract_num_samples(self, model_args: str, expected):
        gen = GenerationParameters.from_model_args(model_args)
        for k, v in expected.items():
            assert getattr(gen, k) == v


class TestToLitellmTextCompletionDict:
    """Tests for GenerationParameters.to_litellm_text_completion_dict()."""

    def test_all_none_returns_empty_dict(self):
        gen = GenerationParameters()
        result = gen.to_litellm_text_completion_dict()
        assert result == {}

    def test_seed_included_when_set(self):
        gen = GenerationParameters(seed=42)
        assert gen.to_litellm_text_completion_dict()["seed"] == 42

    def test_stop_tokens_included_when_set(self):
        gen = GenerationParameters(stop_tokens=["\n", "END"])
        assert gen.to_litellm_text_completion_dict()["stop"] == ["\n", "END"]

    def test_top_p_included_when_set(self):
        gen = GenerationParameters(top_p=0.9)
        assert gen.to_litellm_text_completion_dict()["top_p"] == pytest.approx(0.9)

    def test_frequency_penalty_included(self):
        gen = GenerationParameters(frequency_penalty=0.5)
        assert gen.to_litellm_text_completion_dict()["frequency_penalty"] == pytest.approx(0.5)

    def test_presence_penalty_included(self):
        gen = GenerationParameters(presence_penalty=0.3)
        assert gen.to_litellm_text_completion_dict()["presence_penalty"] == pytest.approx(0.3)

    def test_max_new_tokens_not_included(self):
        """max_new_tokens belongs to the caller (hardcoded to 1 for loglikelihood)."""
        gen = GenerationParameters(max_new_tokens=256)
        assert "max_new_tokens" not in gen.to_litellm_text_completion_dict()
        assert "max_tokens" not in gen.to_litellm_text_completion_dict()
        assert "max_completion_tokens" not in gen.to_litellm_text_completion_dict()

    def test_temperature_not_included(self):
        """temperature is hardcoded to 0.0 by the caller for deterministic scoring."""
        gen = GenerationParameters(temperature=0.7)
        assert "temperature" not in gen.to_litellm_text_completion_dict()

    def test_chat_only_params_not_included(self):
        """repetition_penalty is chat-specific and absent from text_completion."""
        gen = GenerationParameters(repetition_penalty=1.2)
        assert "repetition_penalty" not in gen.to_litellm_text_completion_dict()

    def test_full_config_only_returns_non_none(self):
        gen = GenerationParameters(seed=1, top_p=0.95, stop_tokens=["\n"])
        result = gen.to_litellm_text_completion_dict()
        assert set(result.keys()) == {"seed", "top_p", "stop"}
        assert result["seed"] == 1
        assert result["top_p"] == pytest.approx(0.95)
        assert result["stop"] == ["\n"]
