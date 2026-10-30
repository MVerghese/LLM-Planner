"""
Remote inference interface for OpenAI and Anthropic closed-source models.

Mirrors the public surface of ``VLLM_Interface`` so the rest of the
planning code can interact with either local or closed-source models
without code branches. See ``Inference_Interface.py`` for the wrapper
that selects between local (VLLM) and remote (OpenAI/Anthropic) backends.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from openai import OpenAI

try:
	import anthropic
except ImportError:  # pragma: no cover - handled at runtime
	anthropic = None


# Default path for the API key files. Can be overridden via env vars or
# via ``Remote_Inference_Interface(...)`` kwargs.
_CLAUDE_KEY_PATH = "/home/mverghese/LLM_Grounding/Claude_API_Key.txt"
_GPT_KEY_PATH = "/home/mverghese/LLM_Grounding/GPT_API_Key.txt"

# Maximum number of top log-probs that can be requested from the OpenAI
# chat completions API. The API currently caps this at 20 per token for
# most models, but the GPT-5 reasoning family only accepts up to 5.
OPENAI_MAX_TOP_LOGPROBS = 20
OPENAI_GPT5_MAX_TOP_LOGPROBS = 5

# Fill value used when a requested token is missing from the returned
# top-logprobs (closed-source APIs only return a limited top-k set).
# log(1e-6) ≈ -13.8 — small enough to effectively zero out the option
# after renormalization while still being finite.
MISSING_TOKEN_PROB = 1e-6
MISSING_TOKEN_LOGPROB = float(np.log(MISSING_TOKEN_PROB))


def _read_key_file(path: str) -> Optional[str]:
	"""Read an API key from ``path``, returning ``None`` if unavailable."""
	try:
		with open(path, "r", encoding="utf-8") as f:
			key = f.read().strip()
		return key or None
	except OSError:
		return None


def is_openai_engine(engine: str) -> bool:
	"""Return True if ``engine`` identifies an OpenAI model."""
	if not engine:
		return False
	engine_lower = engine.lower()
	return (
		engine_lower.startswith("gpt-")
		or engine_lower.startswith("o1")
		or engine_lower.startswith("o3")
		or engine_lower.startswith("o4")
		or engine_lower.startswith("openai/")
	)


def is_anthropic_engine(engine: str) -> bool:
	"""Return True if ``engine`` identifies an Anthropic model."""
	if not engine:
		return False
	return engine.lower().startswith("claude")


def is_closed_source_engine(engine: str) -> bool:
	"""Return True if ``engine`` is served by a closed-source API."""
	return is_openai_engine(engine) or is_anthropic_engine(engine)


class _LogprobToken:
	"""Minimal stand-in for the ``ChatCompletionTokenLogprob`` object.

	Returned by Anthropic code paths (which lack real logprobs) so that
	existing callers can iterate with the same attribute access pattern
	as on the OpenAI / VLLM client.
	"""

	__slots__ = ("token", "logprob", "top_logprobs")

	def __init__(self, token: str, logprob: float, top_logprobs: List["_LogprobToken"]):
		self.token = token
		self.logprob = logprob
		self.top_logprobs = top_logprobs


def _match_tokens_in_output(tokens: List[str], output: str) -> Dict[str, float]:
	"""Return a ``{token: logprob}`` map given a free-text model response.

	When a closed-source model does not expose real logprobs we fall back
	to generating an answer and then matching it against the requested
	tokens. This helper makes that match robust to common answer
	prefixes / suffixes emitted by models such as Claude (e.g.
	``"## Answer: **A**"``).
	"""
	import re

	result: Dict[str, float] = {token: MISSING_TOKEN_LOGPROB for token in tokens}
	if not output:
		return result
	stripped = output.strip()
	# Direct equality first (fast path for Yes/No or single-letter answers).
	for token in tokens:
		if token.strip() == stripped:
			result[token] = 0.0
			return result
	# Whole-word match is always preferred: it avoids picking up the
	# letter 'E' out of "answer" when the real answer is 'A'.
	def _search_positions(whole_word: bool) -> Dict[str, int]:
		positions: Dict[str, int] = {}
		for token in tokens:
			needle = token.strip()
			if not needle:
				continue
			if whole_word:
				pattern = re.compile(r"\b" + re.escape(needle) + r"\b", re.IGNORECASE)
				match = pattern.search(stripped)
				if match is not None:
					positions[token] = match.start()
			else:
				idx = stripped.lower().find(needle.lower())
				if idx >= 0:
					positions[token] = idx
		return positions

	matches = _search_positions(whole_word=True)
	if not matches:
		matches = _search_positions(whole_word=False)
	if matches:
		best_token = min(matches, key=matches.get)
		result[best_token] = 0.0
	return result


def _synthesize_logprobs(text: str) -> List[_LogprobToken]:
	"""Build a plausible logprobs sequence for a generated ``text``.

	Anthropic does not return logprobs, so we approximate the structure
	by treating each whitespace-delimited token as having probability
	1.0 (logprob 0) with no alternates.
	"""
	tokens = []
	# Split while preserving whitespace; keeps downstream token-matching
	# code (which uses ``strip``/``rstrip``/``lstrip``) happy.
	for chunk in text.split(" "):
		if not chunk:
			continue
		tokens.append(_LogprobToken(chunk + " ", 0.0, []))
	if tokens:
		# Drop the trailing space from the last token.
		tokens[-1] = _LogprobToken(tokens[-1].token.rstrip(), 0.0, [])
	return tokens


class Remote_Inference_Interface:
	"""Closed-source counterpart of :class:`VLLM_Interface`.

	Supports both OpenAI (GPT / o-series) and Anthropic (Claude) models.
	Methods mirror the local interface so callers can swap backends.
	"""

	def __init__(
		self,
		engine: str,
		openai_api_key: Optional[str] = None,
		anthropic_api_key: Optional[str] = None,
		openai_api_key_path: str = _GPT_KEY_PATH,
		anthropic_api_key_path: str = _CLAUDE_KEY_PATH,
	):
		self.engine = engine
		self.llm_call_count = 0

		if is_openai_engine(engine):
			self.backend = "openai"
			api_key = (
				openai_api_key
				or os.environ.get("OPENAI_API_KEY")
				or _read_key_file(openai_api_key_path)
			)
			if not api_key:
				raise RuntimeError(
					f"Could not find an OpenAI API key. Set OPENAI_API_KEY or place a key at {openai_api_key_path}."
				)
			self.openai_client = OpenAI(api_key=api_key)
			self.anthropic_client = None
		elif is_anthropic_engine(engine):
			if anthropic is None:
				raise ImportError(
					"The 'anthropic' package is not installed. Run `pip install anthropic`."
				)
			self.backend = "anthropic"
			api_key = (
				anthropic_api_key
				or os.environ.get("ANTHROPIC_API_KEY")
				or _read_key_file(anthropic_api_key_path)
			)
			if not api_key:
				raise RuntimeError(
					f"Could not find an Anthropic API key. Set ANTHROPIC_API_KEY or place a key at {anthropic_api_key_path}."
				)
			self.anthropic_client = anthropic.Anthropic(api_key=api_key)
			self.openai_client = None
		else:
			raise ValueError(
				f"Engine '{engine}' is not a supported remote model. Expected OpenAI (gpt-*/o*) or Anthropic (claude-*)."
			)

	# ------------------------------------------------------------------
	# Bookkeeping
	# ------------------------------------------------------------------
	def get_llm_call_count(self) -> int:
		return self.llm_call_count

	def get_tokenization(self, prompt: str) -> List[str]:
		raise NotImplementedError(
			"Closed-source engines do not expose tokenization over the API."
		)

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------
	def _openai_supports_logprobs(self) -> bool:
		"""Heuristic: only a subset of OpenAI models accept ``logprobs``."""
		engine_lower = self.engine.lower()
		# o-series reasoning models do not expose logprobs.
		if engine_lower.startswith("o1") or engine_lower.startswith("o3") or engine_lower.startswith("o4"):
			return False
		return True

	def _openai_needs_reasoning_off(self) -> bool:
		"""Return True if we need ``reasoning={'effort': 'none'}`` to unlock logprobs."""
		engine_lower = self.engine.lower()
		# GPT-5.x reasoning models require reasoning effort = 'none' to
		# accept ``temperature`` / ``top_p`` / ``logprobs``.
		return engine_lower.startswith("gpt-5")

	def _openai_max_top_logprobs(self) -> int:
		"""Return the API's per-model cap on ``top_logprobs``."""
		if self.engine.lower().startswith("gpt-5"):
			return OPENAI_GPT5_MAX_TOP_LOGPROBS
		return OPENAI_MAX_TOP_LOGPROBS

	def _build_openai_content(self, prompt: str, images: Optional[List[str]]):
		content: List[Dict[str, Any]] = []
		if images:
			for image in images:
				content.append(
					{
						"type": "image_url",
						"image_url": {"url": f"data:image/jpeg;base64,{image}"},
					}
				)
		content.append({"type": "text", "text": prompt})
		return content

	def _build_anthropic_content(self, prompt: str, images: Optional[List[str]]):
		content: List[Dict[str, Any]] = []
		if images:
			for image in images:
				content.append(
					{
						"type": "image",
						"source": {
							"type": "base64",
							"media_type": "image/jpeg",
							"data": image,
						},
					}
				)
		content.append({"type": "text", "text": prompt})
		return content

	def _openai_extra_kwargs(self, return_logprobs: bool, num_logprobs: int) -> Dict[str, Any]:
		extras: Dict[str, Any] = {}
		if self._openai_needs_reasoning_off():
			# GPT-5.x family requires reasoning_effort='none' to enable
			# ``temperature`` / ``top_p`` / ``logprobs`` parameters.
			extras["reasoning_effort"] = "none"
		if return_logprobs and self._openai_supports_logprobs():
			extras["logprobs"] = True
			extras["top_logprobs"] = min(max(num_logprobs, 1), self._openai_max_top_logprobs())
		return extras

	# ------------------------------------------------------------------
	# Core generation path
	# ------------------------------------------------------------------
	def generate(
		self,
		prompt: str,
		max_tokens: int = 300,
		temperature: float = 0.0,
		images: Optional[List[str]] = None,
		n: int = 1,
		return_logprobs: bool = False,
		num_logprobs: int = 5,
		logit_bias: Optional[Dict[str, float]] = None,
		add_generation_prompt: bool = True,
		continue_final_message: bool = False,
	):
		"""Generate one or more completions.

		Matches the ``VLLM_Interface.generate`` signature. Closed-source
		APIs do not support ``add_generation_prompt`` or
		``continue_final_message``; these are silently ignored.
		"""
		self.llm_call_count += 1
		if temperature == 0.0 and n > 1:
			print(
				"Warning: Requesting multiple outputs with temperature set to 0.0 will "
				"result in identical outputs. Setting n to 1 but this is probably not what you want"
			)
			n = 1
		if not add_generation_prompt or continue_final_message:
			# Closed-source APIs cannot control these flags, so just note it.
			pass

		if self.backend == "openai":
			return self._openai_generate(
				prompt=prompt,
				max_tokens=max_tokens,
				temperature=temperature,
				images=images,
				n=n,
				return_logprobs=return_logprobs,
				num_logprobs=num_logprobs,
				logit_bias=logit_bias or {},
			)
		return self._anthropic_generate(
			prompt=prompt,
			max_tokens=max_tokens,
			temperature=temperature,
			images=images,
			n=n,
			return_logprobs=return_logprobs,
		)

	def _openai_generate(
		self,
		prompt: str,
		max_tokens: int,
		temperature: float,
		images: Optional[List[str]],
		n: int,
		return_logprobs: bool,
		num_logprobs: int,
		logit_bias: Dict[str, float],
	):
		content = self._build_openai_content(prompt, images)
		extras = self._openai_extra_kwargs(return_logprobs, num_logprobs)
		kwargs: Dict[str, Any] = {
			"model": self.engine,
			"messages": [{"role": "user", "content": content}],
			"n": n,
			**extras,
		}
		# GPT-5.x expects ``max_completion_tokens`` instead of ``max_tokens``.
		if self._openai_needs_reasoning_off():
			kwargs["max_completion_tokens"] = max_tokens
		else:
			kwargs["max_tokens"] = max_tokens
			kwargs["temperature"] = temperature
		if logit_bias:
			kwargs["logit_bias"] = logit_bias

		response = self.openai_client.chat.completions.create(**kwargs)
		generated = [choice.message.content or "" for choice in response.choices]
		if return_logprobs:
			logprobs = []
			for choice in response.choices:
				if choice.logprobs is not None and getattr(choice.logprobs, "content", None):
					logprobs.append(choice.logprobs.content)
				else:
					logprobs.append(_synthesize_logprobs(choice.message.content or ""))
			return generated, logprobs
		return generated

	def _anthropic_generate(
		self,
		prompt: str,
		max_tokens: int,
		temperature: float,
		images: Optional[List[str]],
		n: int,
		return_logprobs: bool,
	):
		content = self._build_anthropic_content(prompt, images)
		generated: List[str] = []
		for _ in range(n):
			response = self.anthropic_client.messages.create(
				model=self.engine,
				max_tokens=max_tokens,
				temperature=temperature,
				messages=[{"role": "user", "content": content}],
			)
			text_parts = [
				block.text
				for block in response.content
				if getattr(block, "type", "") == "text"
			]
			generated.append("".join(text_parts))
		if return_logprobs:
			logprobs = [_synthesize_logprobs(t) for t in generated]
			return generated, logprobs
		return generated

	# ------------------------------------------------------------------
	# Token-level probability helpers
	# ------------------------------------------------------------------
	def get_token_logprobs(
		self,
		prompt: str,
		tokens: Iterable[str],
		images: Optional[List[str]] = None,
		requested_logprobs: int = OPENAI_MAX_TOP_LOGPROBS,
	) -> Dict[str, float]:
		"""Return a dict ``{token: logprob}`` for each token in ``tokens``.

		Closed-source APIs only return logprobs for the top-k most likely
		tokens (k<=20 for OpenAI, and none for Anthropic). If a requested
		token is not in that set we fill it with ``MISSING_TOKEN_LOGPROB``.
		"""
		self.llm_call_count += 1
		tokens = list(tokens)
		if self.backend == "openai":
			return self._openai_get_token_logprobs(prompt, tokens, images, requested_logprobs)
		return self._anthropic_get_token_logprobs(prompt, tokens, images)

	def _openai_get_token_logprobs(
		self,
		prompt: str,
		tokens: List[str],
		images: Optional[List[str]],
		requested_logprobs: int,
	) -> Dict[str, float]:
		if not self._openai_supports_logprobs():
			# Model does not return logprobs at all — fall back to a
			# single-token generation and treat that as a hard choice.
			return self._openai_choice_based_token_logprobs(prompt, tokens, images)

		content = self._build_openai_content(prompt, images)
		kwargs: Dict[str, Any] = {
			"model": self.engine,
			"messages": [{"role": "user", "content": content}],
			"logprobs": True,
			"top_logprobs": min(max(requested_logprobs, 1), self._openai_max_top_logprobs()),
		}
		if self._openai_needs_reasoning_off():
			kwargs["reasoning_effort"] = "none"
			# GPT-5.x sometimes emits multi-token responses even for
			# single-letter answers, so give it a little headroom. The
			# logprobs we care about come from the first generated token.
			kwargs["max_completion_tokens"] = 16
		else:
			kwargs["max_tokens"] = 1
		try:
			response = self.openai_client.chat.completions.create(**kwargs)
		except Exception as exc:
			# The API sometimes rejects logprobs for particular models
			# even when our heuristic allows it. Fall back to a generated
			# single-token response.
			print(f"[Remote_Inference] OpenAI logprobs request failed ({exc}); falling back to choice-based probs.")
			return self._openai_choice_based_token_logprobs(prompt, tokens, images)

		response_logprobs = (
			response.choices[0].logprobs.content[0].top_logprobs
			if response.choices[0].logprobs and response.choices[0].logprobs.content
			else []
		)
		logprobs = {tok.token: tok.logprob for tok in response_logprobs}
		return {token: logprobs.get(token, MISSING_TOKEN_LOGPROB) for token in tokens}

	def _openai_choice_based_token_logprobs(
		self, prompt: str, tokens: List[str], images: Optional[List[str]]
	) -> Dict[str, float]:
		"""Fallback for OpenAI models that do not expose logprobs."""
		generated = self._openai_generate(
			prompt=prompt,
			max_tokens=16,
			temperature=0.0,
			images=images,
			n=1,
			return_logprobs=False,
			num_logprobs=0,
			logit_bias={},
		)
		output = (generated[0] if generated else "").strip()
		return _match_tokens_in_output(tokens, output)

	def _anthropic_get_token_logprobs(
		self, prompt: str, tokens: List[str], images: Optional[List[str]]
	) -> Dict[str, float]:
		"""Fallback for Anthropic: generate an answer and use it."""
		# Give Claude room to emit its usual preamble (``## Answer: **``
		# etc.) then we fish out the intended token from the response.
		generated = self._anthropic_generate(
			prompt=prompt,
			max_tokens=16,
			temperature=0.0,
			images=images,
			n=1,
			return_logprobs=False,
		)
		output = (generated[0] if generated else "").strip()
		return _match_tokens_in_output(tokens, output)

	# ------------------------------------------------------------------
	# Convenience wrappers
	# ------------------------------------------------------------------
	def yes_no_probabilities(self, prompt: str, images: Optional[List[str]] = None) -> np.ndarray:
		logprobs_dict = self.get_token_logprobs(
			prompt, tokens=["Yes", "No"], images=images, requested_logprobs=OPENAI_MAX_TOP_LOGPROBS
		)
		logprobs = np.array([logprobs_dict["Yes"], logprobs_dict["No"]], dtype=np.float64)
		probs = np.exp(logprobs)
		total = float(np.sum(probs))
		if total == 0.0:
			probs = np.array([0.5, 0.5])
		else:
			probs = probs / total
		return probs

	def MCQ_probabilities(
		self,
		prompt: str,
		images: Optional[List[str]] = None,
		options: Optional[List[str]] = None,
	) -> np.ndarray:
		if options is None:
			options = ["A", "B", "C", "D", "E"]
		logprobs_dict = self.get_token_logprobs(
			prompt, tokens=options, images=images, requested_logprobs=OPENAI_MAX_TOP_LOGPROBS
		)
		logprobs = np.array([logprobs_dict[option] for option in options], dtype=np.float64)
		probs = np.exp(logprobs)
		total = float(np.sum(probs))
		if total == 0.0:
			probs = np.full(len(options), 1.0 / len(options))
		else:
			probs = probs / total
		return probs

	def VQA(
		self,
		question: str,
		images: List[str],
		num_candidate_images: int = 1,
		recency_bias: float = 0.5,
	) -> str:
		"""Visual Question Answering mirror of the VLLM implementation."""
		if not images:
			answer_prompt = (
				"Answer the following question in a single complete sentence:\n"
				f"{question}{'?' if not question.endswith('?') else ''}"
			)
			return self.generate(answer_prompt, images=[], n=1)[0]

		image_selection_prompt = (
			f"Can the question: {question} be answered using information in this image?"
		)
		image_scores = [
			self.yes_no_probabilities(image_selection_prompt, images=[image])[0]
			for image in images
		]
		image_times = np.linspace(0, 1, len(images))
		image_recency_weights = np.exp(recency_bias * (image_times - 1))
		image_scores_weighted = np.array(image_scores) * image_recency_weights
		top_k_images = np.argsort(image_scores_weighted)[-num_candidate_images:][::-1]
		answer_prompt = (
			"Based on the information in the images, answer the following question "
			f"in a single complete sentence:\n {question}"
			f"{'?' if not question.endswith('?') else ''}"
		)
		return self.generate(
			answer_prompt,
			images=[images[i] for i in top_k_images],
			n=1,
		)[0]
