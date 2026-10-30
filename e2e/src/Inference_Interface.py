"""
Unified inference interface wrapper.

Exposes :class:`InferenceInterface`, a thin dispatcher that picks between
the local VLLM-backed :class:`VLLM_Interface` and the closed-source
:class:`Remote_Inference_Interface` based on the engine name.

Callers can write::

    llm = InferenceInterface(engine="claude-sonnet-4-6")
    llm = InferenceInterface(engine="gpt-5.4")
    llm = InferenceInterface(engine="Qwen/Qwen3-VL-8B-Instruct", client=client)

and then use the familiar ``generate`` / ``MCQ_probabilities`` /
``yes_no_probabilities`` / ``VQA`` methods regardless of backend.
"""

from __future__ import annotations

from typing import Any, Optional

from Remote_Inference_Interface import (
	Remote_Inference_Interface,
	is_anthropic_engine,
	is_closed_source_engine,
	is_openai_engine,
)
from VLLM_Interface import VLLM_Interface


class InferenceInterface:
	"""Dispatcher that forwards calls to the correct underlying backend.

	Attributes
	----------
	engine:
		The model identifier used to select the backend.
	backend:
		Either ``"remote"`` (OpenAI / Anthropic) or ``"local"`` (VLLM).
	is_closed_source:
		True if the backend is a closed-source remote API.
	impl:
		The concrete interface instance, useful for backend-specific
		access if needed.
	"""

	def __init__(self, engine: str, client: Optional[Any] = None, **kwargs: Any):
		self.engine = engine
		if is_closed_source_engine(engine):
			self.backend = "remote"
			self.is_closed_source = True
			self.impl = Remote_Inference_Interface(engine, **kwargs)
		else:
			self.backend = "local"
			self.is_closed_source = False
			self.impl = VLLM_Interface(engine=engine, client=client)

	# Any attribute access (generate, MCQ_probabilities, yes_no_probabilities,
	# VQA, get_llm_call_count, etc.) is transparently forwarded to the
	# underlying implementation.
	def __getattr__(self, name: str) -> Any:
		# ``__getattr__`` is only called when the attribute is not found
		# on the instance itself, so this avoids infinite recursion for
		# ``engine`` / ``backend`` / ``impl``.
		return getattr(self.impl, name)


__all__ = [
	"InferenceInterface",
	"is_anthropic_engine",
	"is_closed_source_engine",
	"is_openai_engine",
]
