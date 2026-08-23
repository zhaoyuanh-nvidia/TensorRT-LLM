import bisect
import contextlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, TypeAlias

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.llmapi.llm_args import (
    BaseSparseAttentionConfig,
    DecodingBaseConfig,
    DSparkDecodingConfig,
    SeqLenAwareSparseAttentionConfig,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend.trtllm import TrtllmAttentionMetadata
from ..distributed import Distributed
from ..expert_statistic import ExpertStatistic
from ..memory_buffer_utils import Buffers, get_memory_buffers
from ..modules.multi_stream_utils import with_multi_stream
from ..speculative.eagle3 import Eagle3ResourceManager
from ..speculative.interface import SpecMetadata
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from ..speculative.utils import get_draft_kv_cache_manager
from ..utils import make_weak_ref, piecewise_cuda_graph
from .llm_request import LlmRequest, get_draft_token_length
from .resource_manager import BaseResourceManager, ResourceManager, ResourceManagerType
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests

# A large prime number used for dummy request IDs to avoid collisions
CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 64) - 1
# Gen dummies get prompt_len = token_num - 1. Before capturing enc-dec decode
# graphs, prepare_cross_batch temporarily runs each dummy generation request
# as a one-token context chunk to write its cross-KV cache, so enc-dec
# dummies need one prompt token plus one generated token.
ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM = 2


class KeyType(NamedTuple):
    batch_size: int
    draft_len: int
    is_first_draft: bool
    short_seq_len_mode: bool = False
    is_all_greedy_sample: bool = True
    # Primarily used for mixed batches of encoder-decoder models.
    num_contexts: int = 0
    context_query_len: int = 0
    num_encoder_tokens: int = 0
    peft_cache_data_type: Optional[torch.dtype] = None
    # Total packed target-verifier token capacity. Zero retains the legacy
    # per-request ``draft_len + 1`` token count.
    verifier_num_tokens: int = 0


def _save_spec_decode_capture_state(
        attn_metadata: Any, enable_spec_decode: bool) -> Optional[torch.Tensor]:
    if not enable_spec_decode or not hasattr(attn_metadata, 'kv_lens_cuda'):
        return None
    return attn_metadata.kv_lens_cuda[:attn_metadata.num_seqs].clone()


def _restore_spec_decode_capture_state(
        attn_metadata: Any, saved_kv_lens_cuda: Optional[torch.Tensor]) -> None:
    if saved_kv_lens_cuda is None:
        return
    # Speculative decoding updates kv_lens_cuda in-place during every forward.
    # CUDA graph warmup reuses one dummy request for multiple eager forwards, so
    # letting those updates accumulate would make later warmups/capture advertise
    # more KV tokens than the dummy request actually allocated. Restore the
    # single-step input state outside the graph after each forward instead.
    batch_size = saved_kv_lens_cuda.shape[0]
    attn_metadata.kv_lens_cuda[:batch_size].copy_(saved_kv_lens_cuda)
    attn_metadata.on_update_kv_lens()


@dataclass
class CUDAGraphRunnerConfig:
    """Configuration for the CUDAGraphRunner, passed from the ModelEngine."""
    use_cuda_graph: bool
    """
    Master switch controlling the model's execution path.

    This flag determines one of three distinct execution paths for the
    model engine:

    1.  **`False` (Pure Eager Path):**
        * Forces all execution to be in eager mode.
        * The `CUDAGraphRunner` instance is mostly dormant
        * Methods like `maybe_get_cuda_graph` and `pad_batch`
            will return immediately, signaling the model engine to
            run in eager mode.

    2.  **`True` (Eager Fallback Path):**
        * The runner is active and checks for graph eligibility.
        * If a batch is ineligible (e.g., it's a prefill batch,
            stats collection is on, or it's an unsupported batch size),
            the runner signals a fallback to eager mode for that batch.

    3.  **`True` (CUDA Graph Path):**
        * The runner finds an eligible batch and a matching graph.
        * The graph is then captured (if new) or replayed.

    Note: As of this implementation, the model engine *always* calls
    `cuda_graph_runner.pad_batch` and `cuda_graph_runner.maybe_get_cuda_graph`
    even when this is `False`. This could be refactored in the future
    so that the engine bypasses the `CUDAGraphRunner` entirely in Case 1.
    """
    cuda_graph_padding_enabled: bool
    cuda_graph_batch_sizes: list[int]
    max_cuda_graph_batch_size: int
    max_beam_width: int
    max_num_tokens: int
    spec_config: Optional[DecodingBaseConfig]
    cuda_graph_mem_pool: Any
    use_mrope: bool
    original_max_draft_len: int
    original_max_total_draft_tokens: int
    is_draft_model: bool
    enable_attention_dp: bool
    is_encoder_decoder: bool
    batch_size: int
    mapping: Optional[Mapping]
    dist: Optional[Distributed]
    kv_cache_manager_key: Any
    dynamic_draft_len_mapping: Optional[Dict[int, int]] = None
    sparse_attention_config: Optional[BaseSparseAttentionConfig] = None
    enable_encoder_decoder_mixed_cuda_graph: bool = False


class CUDAGraphRunner:
    """
    Manages the lifecycle and execution of CUDA graphs for the model engine.

    This unified class handles high-level orchestration (padding, eligibility)
    and low-level execution (capturing, resource management, replaying) for
    multiple graphs, keyed by (batch size, draft_len, is_first_draft).
    """
    WARMUP_STEPS = 1

    def __init__(self, config: CUDAGraphRunnerConfig):
        self.config = config

        # High-level configuration from the config object
        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = config.cuda_graph_batch_sizes
        self.max_supported_batch_size = config.max_cuda_graph_batch_size
        self.max_beam_width = config.max_beam_width
        self.spec_config = config.spec_config
        self.sparse_config = config.sparse_attention_config
        self.is_encoder_decoder = config.is_encoder_decoder
        self.enable_encoder_decoder_mixed_cuda_graph = (
            config.enable_encoder_decoder_mixed_cuda_graph)

        self.graphs: Dict[KeyType, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[KeyType,
                                 Callable[[], Optional[torch.Tensor]]] = {}
        self.graph_metadata: Dict[KeyType, Dict[str, Any]] = {}
        self.memory_pool = config.cuda_graph_mem_pool
        # Alternative verifier budgets have different live allocation
        # layouts. Keep each (BS, physical DL, V) shape in its own graph pool
        # so later captures cannot recycle addresses held by an earlier graph.
        self._confidence_memory_pools: Dict[Tuple[int, int, int], Any] = {}
        self.padding_dummy_requests: Dict[int, LlmRequest] = {}
        self.confidence_adp_plan_ready = False
        self.confidence_adp_execution_batch_size = 0
        self.confidence_adp_verifier_token_budget = 0
        self.confidence_adp_route_epoch = 0
        self.confidence_force_full_k_route = False
        self.confidence_device_layout = None
        self.confidence_query_lens_host = None
        self.confidence_device_layout_row_map_valid = False
        self.confidence_discarded_device_layout = False
        self.confidence_engine_generation = 0
        self.dynamic_draft_len_mapping = config.dynamic_draft_len_mapping

        self.shared_static_tensors: Dict[str, torch.Tensor] = {}
        if self.enabled:
            self._create_shared_static_tensors()
        self.cuda_graph_meta_buffers = get_memory_buffers()

        # On-the-fly capture is disabled by default to prevent workspace
        # tensor reallocation from invalidating addresses baked into existing
        # CUDA graphs.  Use allow_capture() context manager during warmup.
        self._capture_allowed = False
        self.is_warmup_only = False

    def _create_shared_static_tensors(self):
        """Allocates static tensors sized for the largest possible batch."""
        runtime_draft_token_buffer_width = (
            self.config.original_max_total_draft_tokens
            if self.config.spec_config is not None else 0)
        token_per_request = runtime_draft_token_buffer_width + 1
        max_total_tokens = (self.max_supported_batch_size *
                            self.max_beam_width * token_per_request)
        if self.enable_encoder_decoder_mixed_cuda_graph:
            # A mixed encoder-decoder batch can contain multiple decoder
            # context tokens per request, unlike a pure generation batch.
            max_total_tokens = self.config.max_num_tokens
        max_total_tokens = min(max_total_tokens, self.config.max_num_tokens)

        self.shared_static_tensors = {
            "input_ids":
            torch.ones((max_total_tokens, ), device="cuda", dtype=torch.int32),
            "position_ids":
            torch.zeros((1, max_total_tokens), device="cuda",
                        dtype=torch.int32),
        }
        if self.config.use_mrope:
            self.shared_static_tensors["position_ids"] = torch.zeros(
                (3, 1, max_total_tokens), device="cuda", dtype=torch.int32)
            self.shared_static_tensors[
                "mrope_delta_read_seq_slots"] = torch.zeros(
                    (max_total_tokens, ), device="cuda", dtype=torch.long)

    def _get_static_encoder_hidden_states(
        self,
        encoder_hidden_states: torch.Tensor,
        num_encoder_tokens: int,
        *,
        allow_allocate: bool,
    ) -> torch.Tensor:
        """Return the stable mixed-graph encoder input, allocating at warmup."""
        if encoder_hidden_states.ndim != 2:
            raise RuntimeError(
                "Mixed encoder-decoder CUDA graphs require rank-2 packed "
                "encoder hidden states.")

        static_encoder_hidden_states = self.shared_static_tensors.get(
            "encoder_hidden_states")
        if static_encoder_hidden_states is None:
            if not allow_allocate:
                raise RuntimeError(
                    "Mixed encoder-decoder CUDA graph replay requires the "
                    "encoder hidden-state buffer initialized during warmup.")
            if self.graphs:
                raise RuntimeError(
                    "Mixed encoder-decoder CUDA graph encoder hidden-state "
                    "buffer cannot be allocated after graph capture.")
            static_encoder_hidden_states = encoder_hidden_states.new_empty(
                (num_encoder_tokens, encoder_hidden_states.shape[1]))
            self.shared_static_tensors[
                "encoder_hidden_states"] = static_encoder_hidden_states

        if static_encoder_hidden_states.shape[0] < num_encoder_tokens:
            raise RuntimeError(
                "Mixed encoder-decoder CUDA graph encoder hidden-state buffer "
                f"has capacity {static_encoder_hidden_states.shape[0]}, but "
                f"{num_encoder_tokens} tokens were requested.")
        return static_encoder_hidden_states[:num_encoder_tokens]

    def _is_mixed_encoder_decoder_batch(self, batch: ScheduledRequests) -> bool:
        return (self.is_encoder_decoder and batch.num_context_requests > 0
                and batch.num_generation_requests > 0)

    def _can_run_cuda_graph_batch(self, batch: ScheduledRequests) -> bool:
        return (batch.can_run_cuda_graph
                or (self.enable_encoder_decoder_mixed_cuda_graph
                    and self._is_mixed_encoder_decoder_batch(batch)))

    def _trace_dspark_graph_reject(self, reason: str, **fields: Any) -> None:
        """Emit bounded diagnostics only after graph dispatch is live."""
        if (not self.enabled or os.environ.get("TLLM_DSPARK_TRACE_GRAPH_REJECT",
                                               "0") != "1"):
            return
        count = getattr(self, "_dspark_graph_reject_trace_count", 0)
        if count >= 512:
            return
        details = ", ".join(f"{name}={value}" for name, value in fields.items())
        logger.info(
            f"DSpark graph route reject: index={count}, reason={reason}" +
            (f", {details}" if details else ""))
        self._dspark_graph_reject_trace_count = count + 1

    def _get_seq_len_mode(
        self,
        batch: ScheduledRequests,
        new_tensors_device: Optional[SampleStateTensors] = None,
        promoted_context_request_ids: frozenset[int] = frozenset()
    ) -> bool:
        """Select the sparse-attention graph family for the execution view.

        ``promoted_context_request_ids`` contains semantic final-context rows
        that the model engine temporarily placed in the generation list. It is
        empty for the existing generation-only path.
        """
        if (isinstance(self.sparse_config, SeqLenAwareSparseAttentionConfig)
                and self.sparse_config.needs_separate_short_long_cuda_graphs()):
            # Some sparse attention algorithms need to use different forward paths for short and long sequences.
            # For example, the DSA can skip the MQA and Top-K in the indexer for short sequences to reduce the
            # computational overhead. To support this feature, we need to capture separate CUDA graphs for short
            # and long sequences. We need to first collect the sequence length of the requests and then determine
            # the sequence length mode. For long sequences, use the default maximum sequence length. For short
            # sequences, use the sequence length threshold as the maximum sequence length.
            total_seq_lens = []
            new_tokens_device, next_draft_tokens_device = None, None
            if new_tensors_device is not None:
                new_tokens_device = new_tensors_device.new_tokens
                if isinstance(new_tensors_device, SampleStateTensorsSpec):
                    next_draft_tokens_device = new_tensors_device.next_draft_tokens
            overlap_scheduler_enabled = new_tokens_device is not None
            for request in batch.generation_requests:
                is_spec_request = get_draft_token_length(
                    request) > 0 or next_draft_tokens_device is not None
                num_draft_tokens = self.spec_config.max_draft_len if is_spec_request else 0
                if request.py_request_id in promoted_context_request_ids:
                    # A promoted context row may retain overlap bookkeeping
                    # such as py_batch_idx from an earlier context chunk. That
                    # state describes the previous batch, not the sequence
                    # length of the final prompt token executed by this graph.
                    # Use the authoritative context cursor so graph keying
                    # matches the decode-shaped input prepared for this row.
                    total_seq_len = request.context_current_position + 1
                # First draft
                elif request.py_is_first_draft:
                    # get_num_tokens is O(1); len(get_tokens(0)) marshals the
                    # whole O(seq_len) VecTokens into a Python list just for len.
                    total_seq_len = request.get_num_tokens(0)
                # With overlap scheduler disabled or dummy request or not assigned to a batch,
                elif not overlap_scheduler_enabled or request.is_dummy or request.py_batch_idx is None:
                    total_seq_len = request.max_beam_num_tokens + num_draft_tokens
                # Other cases
                else:
                    total_seq_len = request.max_beam_num_tokens + num_draft_tokens + 1
                total_seq_lens.append(total_seq_len)
            # Determine the sequence length mode.
            from ..speculative import get_num_extra_kv_tokens
            num_extra_kv_tokens = get_num_extra_kv_tokens(self.spec_config)
            max_seq_len = max(total_seq_lens)
            if max_seq_len <= self.sparse_config.seq_len_threshold - num_extra_kv_tokens:
                short_seq_len_mode = True
            else:
                short_seq_len_mode = False
        else:
            # For non-sparse attention or sparse attention that does not need separate short and long CUDA graphs,
            # use the default sequence length mode.
            short_seq_len_mode = False
        return short_seq_len_mode

    def get_graph_key(
        self,
        batch: ScheduledRequests,
        new_tensors_device: Optional[SampleStateTensors] = None,
        spec_resource_manager: Optional[BaseResourceManager] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        peft_cache_data_type: Optional[torch.dtype] = None,
    ) -> Optional[KeyType]:
        batch_size = batch.batch_size

        # Promoted IDs correct the sequence length observed by sparse
        # short/long graph selection.
        short_seq_len_mode = self._get_seq_len_mode(
            batch, new_tensors_device, promoted_context_request_ids)

        # Spec one-engine sampler has two code paths (argmax fast-path vs
        # advanced sampling kernel). Include this in the key so we capture
        # both variants and dispatch at replay based on actual batch state.
        # Default to True (greedy fast-path) when the metadata doesn't carry
        # this field (non-one-engine paths or non-spec batches).
        is_all_greedy_sample = bool(
            getattr(spec_metadata, "is_all_greedy_sample", True))

        if self.config.is_draft_model and spec_resource_manager is not None and isinstance(
                spec_resource_manager, Eagle3ResourceManager):
            # If 'is_first_draft' is True, even with tree decoding, the length of draft_len will only be 'max_draft_len', not 'max_total_draft_token'.
            # Because we will pad the input to 'max_draft_len' length for the first draft layer.
            draft_len = self.config.original_max_draft_len if spec_resource_manager.is_first_draft else 0
            key = KeyType(batch_size=batch_size,
                          draft_len=draft_len,
                          is_first_draft=spec_resource_manager.is_first_draft,
                          short_seq_len_mode=short_seq_len_mode,
                          is_all_greedy_sample=is_all_greedy_sample,
                          peft_cache_data_type=peft_cache_data_type)
        else:
            # With dynamic spec decode, the draft length may be zero even when enable_spec_decode is True,
            # so we need to get the draft length from the batch instead of using enable_spec_decode.
            draft_len_list = []
            for request in batch.generation_requests:
                draft_len_list.append(len(request.py_draft_tokens))
            draft_len = max(draft_len_list)
            assert len(
                set(draft_len_list)) == 1, "All draft lengths must be the same"
            context_requests = batch.context_requests
            num_contexts = len(context_requests)
            context_query_len = 0
            if num_contexts:
                context_query_len = int(context_requests[0].context_chunk_size)
                if any(
                        int(request.context_chunk_size) != context_query_len
                        for request in context_requests[1:]):
                    self._trace_dspark_graph_reject(
                        "context_query_len_mismatch",
                        batch_size=batch_size,
                        context_query_len=context_query_len)
                    return None
            num_encoder_tokens = sum(
                int(request.encoder_output_len) for request in context_requests
                if not request.py_skip_cross_kv_projection)
            verifier_num_tokens = 0
            if (not self.config.is_draft_model
                    and isinstance(self.spec_config, DSparkDecodingConfig)
                    and self.spec_config.is_confidence_budget_enabled
                    and is_all_greedy_sample):
                force_full_k_route = bool(
                    getattr(spec_metadata, "confidence_force_full_k_route",
                            False))
                carried_budget = int(
                    getattr(spec_metadata,
                            "confidence_verifier_token_budget", 0) or 0)
                configured_budget = (
                    None if force_full_k_route else
                    (carried_budget if carried_budget > 0 else self.spec_config.
                     resolve_confidence_verifier_token_budget(batch_size)))
                is_synthetic_capture = (
                    self._capture_allowed and num_contexts == 0
                    and bool(batch.generation_requests) and all(
                        getattr(request, "is_dummy", False)
                        for request in batch.generation_requests))
                if is_synthetic_capture and configured_budget is not None:
                    retained_budget = (configured_budget -
                                       len(batch.generation_requests))
                    base_retained, extra_retained = divmod(
                        retained_budget, len(batch.generation_requests))
                    synthetic_effective_lens = [
                        base_retained + (index < extra_retained)
                        for index in range(len(batch.generation_requests))
                    ]
                    if any(effective_len > physical_len
                           for effective_len, physical_len in zip(
                               synthetic_effective_lens, draft_len_list)):
                        self._trace_dspark_graph_reject(
                            "synthetic_effective_len_exceeds_physical",
                            batch_size=batch_size,
                            configured_budget=configured_budget)
                        return None
                    for request, effective_len in zip(batch.generation_requests,
                                                      synthetic_effective_lens):
                        request.py_draft_tokens_effective_len = effective_len

                carried_layout = getattr(self, "confidence_device_layout", None)
                carried_layout_route = bool(
                    not is_synthetic_capture and carried_layout is not None
                    and configured_budget is not None)
                carried_compact_route = bool(
                    carried_layout_route
                    and configured_budget < batch_size * (draft_len + 1))
                effective_draft_lens = []
                for request in batch.generation_requests:
                    effective_len = getattr(request,
                                            "py_draft_tokens_effective_len",
                                            None)
                    if effective_len is None:
                        effective_len = len(request.py_draft_tokens)
                    effective_draft_lens.append(int(effective_len))
                if any(effective_len < 0 or effective_len > physical_len
                       for effective_len, physical_len in zip(
                           effective_draft_lens, draft_len_list)):
                    self._trace_dspark_graph_reject(
                        "invalid_effective_draft_len",
                        batch_size=batch_size,
                        effective_lens=effective_draft_lens,
                        physical_lens=draft_len_list)
                    return None

                is_compact_step = (
                    carried_compact_route if carried_layout_route else any(
                        effective_len != physical_len
                        for effective_len, physical_len in zip(
                            effective_draft_lens, draft_len_list)))
                has_live_dummy = (not is_synthetic_capture and any(
                    getattr(request, "is_dummy", False)
                    for request in batch.generation_requests))
                anchor_only_live_padding = (
                    has_live_dummy and
                    (carried_layout_route or all(
                        (not getattr(request, "is_dummy", False)) or
                        (getattr(request, "is_cuda_graph_dummy", False)
                         and effective_len == 0)
                        for request, effective_len in zip(
                            batch.generation_requests,
                            effective_draft_lens))))
                confidence_shape_active = (not force_full_k_route and (
                    configured_budget is not None or bool(
                        getattr(spec_metadata, "confidence_fixed_budget_active",
                                False)) or anchor_only_live_padding))
                if num_contexts and (is_compact_step
                                     or confidence_shape_active):
                    self._trace_dspark_graph_reject(
                        "confidence_context_unsupported",
                        batch_size=batch_size,
                        num_contexts=num_contexts)
                    return None

                # Live compact padding is executable only in the anchor-only
                # layout: every padding row is the runner-owned CUDA-graph
                # dummy, has effective draft length zero, and therefore needs
                # no previous overlap row. Synthetic all-dummy capture remains
                # a separate trusted path above and can use nonzero lengths.
                compact_shape_ready = (not has_live_dummy
                                       or anchor_only_live_padding)
                if has_live_dummy and not anchor_only_live_padding:
                    self._trace_dspark_graph_reject(
                        "confidence_live_dummy_fallback_full_k",
                        batch_size=batch_size)
                if is_compact_step:
                    # A compact key is valid only when input preparation can
                    # stage the matching packed rows in this same iteration.
                    # Under the overlap scheduler, request-local retained
                    # lengths can outlive the device tensors / batch-index
                    # mapping that produced them during roster transitions.
                    # Keying a compact graph in that state would replay the
                    # previous compact step's row maps and KV extents.
                    compact_inputs_ready = (
                        carried_compact_route or
                        (new_tensors_device is not None and all(
                            (getattr(request, "is_dummy", False)
                             and anchor_only_live_padding) or
                            (getattr(request, "py_draft_tokens_effective_len",
                                     None) is not None and getattr(
                                         request, "py_batch_idx", None)
                             is not None)
                            for request in batch.generation_requests)))
                    compact_shape_ready = (compact_shape_ready
                                           and (is_synthetic_capture
                                                or compact_inputs_ready))
                    if not compact_shape_ready:
                        self._trace_dspark_graph_reject(
                            "compact_inputs_fallback_full_k",
                            batch_size=batch_size,
                            has_new_tensors=new_tensors_device is not None)
                if confidence_shape_active and compact_shape_ready:
                    # Every confidence-selected target shape, including the
                    # physical full-K tier, needs an explicit V key.  Otherwise
                    # the full-K confidence capture aliases the ordinary DSpark
                    # graph while their ragged row maps / attention metadata
                    # can differ.  Dynamic mode deliberately does not infer
                    # confidence activity from physical request lengths alone:
                    # generic model-memory warmup also presents full-K dummy
                    # requests but prepares the ordinary (non-confidence)
                    # metadata path.
                    actual_budget = (
                        configured_budget if carried_layout_route else sum(
                            1 + effective_len
                            for effective_len in effective_draft_lens))
                    if configured_budget is None:
                        configured_budget = self.spec_config.resolve_confidence_verifier_token_budget(
                            batch_size, effective_draft_lens)
                    if (configured_budget is None
                            or actual_budget != configured_budget):
                        self._trace_dspark_graph_reject(
                            "confidence_budget_not_executable",
                            batch_size=batch_size,
                            actual_budget=actual_budget,
                            configured_budget=configured_budget,
                            candidates=self.spec_config.
                            resolve_confidence_verifier_token_budget_candidates(
                                batch_size),
                            effective_lens=effective_draft_lens)
                        return None
                    verifier_num_tokens = configured_budget
                elif is_compact_step and not confidence_shape_active:
                    self._trace_dspark_graph_reject(
                        "compact_confidence_fallback_full_k",
                        batch_size=batch_size,
                        effective_lens=effective_draft_lens)
            key = KeyType(batch_size=batch_size,
                          draft_len=draft_len,
                          is_first_draft=False,
                          short_seq_len_mode=short_seq_len_mode,
                          is_all_greedy_sample=is_all_greedy_sample,
                          num_contexts=num_contexts,
                          context_query_len=context_query_len,
                          num_encoder_tokens=num_encoder_tokens,
                          peft_cache_data_type=peft_cache_data_type,
                          verifier_num_tokens=verifier_num_tokens)
        return key

    def _get_compatible_mixed_encoder_decoder_key(self,
                                                  key: KeyType) -> KeyType:
        """Round the packed encoder extent up to a captured graph key."""
        if (not self.padding_enabled or self._capture_allowed
                or key in self.graph_metadata or key.num_encoder_tokens == 0):
            return key

        key_without_encoder_extent = key._replace(num_encoder_tokens=0)
        compatible_keys = [
            captured_key for captured_key in self.graph_outputs
            if isinstance(captured_key, KeyType) and captured_key._replace(
                num_encoder_tokens=0) == key_without_encoder_extent
            and captured_key.num_encoder_tokens >= key.num_encoder_tokens
        ]
        if not compatible_keys:
            return key
        return min(compatible_keys,
                   key=lambda captured_key: captured_key.num_encoder_tokens)

    @staticmethod
    def _get_mrope_position_delta(request: Any) -> Optional[Any]:
        mrope_position_delta = getattr(request, "py_mrope_position_delta", None)
        if mrope_position_delta is not None:
            return mrope_position_delta

        multimodal_data = getattr(request, "py_multimodal_data", None)
        if not multimodal_data:
            return None

        mrope_config = multimodal_data.get("mrope_config")
        if mrope_config is None:
            return None
        return mrope_config.get("mrope_position_deltas")

    @staticmethod
    def _needs_mrope_delta_cache_update(request: Any) -> bool:
        if request.py_seq_slot is None or request.is_dummy:
            return False

        if getattr(request, "py_mrope_delta_cache_slot",
                   None) == request.py_seq_slot:
            return False

        return CUDAGraphRunner._get_mrope_position_delta(request) is not None

    def __del__(self):
        self.clear()

    def maybe_get_cuda_graph(
        self,
        batch: ScheduledRequests,
        enable_spec_decode: bool,
        attn_metadata: Any,
        spec_metadata: Optional[SpecMetadata] = None,
        draft_tokens_cuda: Optional[torch.Tensor] = None,
        new_tensors_device: Optional[SampleStateTensors] = None,
        spec_resource_manager: Optional[BaseResourceManager] = None,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        peft_cache_data_type: Optional[torch.dtype] = None,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[KeyType]]:
        """
        Determines if the current batch can be run with a CUDA graph.

        Returns a tuple containing:
        - The attn_metadata for the graph, if applicable.
        - The spec_metadata for the graph, if applicable.
        - The key for the graph, if applicable.

        ``promoted_context_request_ids`` is execution-view metadata. It does
        not change request state or type and is used only to build a graph key
        consistent with the final-context token that will be executed.
        """
        # disable when doing statistic
        if ExpertStatistic.should_record():
            self._trace_dspark_graph_reject("expert_statistics_active")
            return None, None, None

        is_mixed_encoder_decoder = self._is_mixed_encoder_decoder_batch(batch)
        can_run_cuda_graph = self._can_run_cuda_graph_batch(batch)
        batch_size = batch.batch_size
        key = (self.get_graph_key(
            batch, new_tensors_device, spec_resource_manager, spec_metadata,
            promoted_context_request_ids, peft_cache_data_type)
               if self.enabled and can_run_cuda_graph else None)
        if self.enabled and self.config.enable_attention_dp:
            # A compact verifier graph has a second static shape axis: V.  ADP
            # peers must therefore agree on both G and V (and on physical K),
            # not merely on graph eligibility and batch size.  In particular,
            # a rank whose overlap tensors are not ready returns key=None above;
            # all peers then fall back together instead of mixing a compact
            # replay with a full-K path across one TP/EP collective group.
            live_compact_inputs = (
                not getattr(self.config, "is_draft_model", False)
                and not self._capture_allowed
                and new_tensors_device is not None
                and (getattr(self, "confidence_device_layout", None)
                     is not None or any(
                         getattr(request, "py_draft_tokens_effective_len", None)
                         is not None
                         and int(request.py_draft_tokens_effective_len)
                         != len(request.py_draft_tokens)
                         for request in batch.generation_requests)))
            graph_batch_info = self.config.dist.tp_allgather([
                can_run_cuda_graph,
                batch_size,
                -1 if key is None else int(key.verifier_num_tokens),
                -1 if key is None else int(key.draft_len),
                live_compact_inputs,
            ])
            all_can_run_cuda_graph = all(rank_info[0]
                                         for rank_info in graph_batch_info)
            all_batch_sizes_equal = all(rank_info[1] == graph_batch_info[0][1]
                                        for rank_info in graph_batch_info)
            all_verifier_shapes_equal = all(
                rank_info[2] == graph_batch_info[0][2]
                for rank_info in graph_batch_info)
            all_draft_lens_equal = all(rank_info[3] == graph_batch_info[0][3]
                                       for rank_info in graph_batch_info)

            unsafe_late_compact_fallback = any(
                rank_info[4]
                and (rank_info[2] <= 0 or not rank_info[0])
                for rank_info in graph_batch_info)
            if unsafe_late_compact_fallback:
                raise RuntimeError(
                    "DSpark attention-DP compact inputs cannot be converted "
                    "to full-K after draft tensors were generated; the common "
                    f"graph route was unavailable: {graph_batch_info}")
            if (not all_can_run_cuda_graph or not all_batch_sizes_equal
                    or not all_draft_lens_equal or key is None):
                self._trace_dspark_graph_reject(
                    "attention_dp_shape_gate",
                    local_key=key,
                    graph_batch_info=graph_batch_info)
                return None, None, None
            if not all_verifier_shapes_equal:
                self._trace_dspark_graph_reject(
                    "attention_dp_verifier_shape_mismatch",
                    local_key=key,
                    graph_batch_info=graph_batch_info)
                raise RuntimeError(
                    "DSpark attention-DP peers selected different verifier "
                    f"shapes after draft generation: {graph_batch_info}")

        if not self.enabled or not can_run_cuda_graph:
            self._trace_dspark_graph_reject(
                "runner_disabled_or_batch_ineligible",
                enabled=self.enabled,
                can_run_cuda_graph=can_run_cuda_graph,
                batch_size=batch_size,
                batch_can_run_cuda_graph=batch.can_run_cuda_graph)
            return None, None, None
        if self.config.use_mrope and any(
                self._needs_mrope_delta_cache_update(request)
                for request in batch.generation_requests):
            # Some MRoPE paths have no per-request delta (for example,
            # Qwen3.5 configs normalized to text-only decoding). Requests that
            # do carry a delta must first populate the model-side cache for their
            # current seq slot before graph replay.
            return None, None, None
        # Propagate the execution-view identity through graph lookup. Existing
        # callers pass the empty default and retain generation-only behavior.
        if key is None:
            self._trace_dspark_graph_reject("graph_key_unresolved",
                                            batch_size=batch_size)
            return None, None, None
        if is_mixed_encoder_decoder:
            key = self._get_compatible_mixed_encoder_decoder_key(key)

        if key in self.graph_metadata:
            return self.graph_metadata[key][
                "attn_metadata"], self.graph_metadata[key]["spec_metadata"], key

        # Capturing a mixed graph on a live batch would execute its KV-cache
        # writes during graph warmup/capture and could resize shared attention
        # workspace after older graph pointers have been fixed. Only shapes
        # captured by the two-pass startup warmup may replay.
        if not self._capture_allowed:
            self._trace_dspark_graph_reject("key_not_captured",
                                            key=key,
                                            captured_keys=tuple(
                                                self.graph_metadata.keys()))
            return None, None, None

        if batch_size not in self.supported_batch_sizes:
            self._trace_dspark_graph_reject(
                "unsupported_capture_batch_size",
                batch_size=batch_size,
                supported_batch_sizes=self.supported_batch_sizes)
            return None, None, None

        num_sequences_in_batch = batch_size * self.max_beam_width
        graph_attn_metadata = attn_metadata.create_cuda_graph_metadata(
            num_sequences_in_batch, False, key.draft_len,
            self.cuda_graph_meta_buffers)
        if key.verifier_num_tokens:
            # create_cuda_graph_metadata initializes every generation row to
            # K+1, but a confidence capture packs exactly V tokens.  Seed the
            # graph-resident metadata with that same V before the first model
            # warmup; later live replays overwrite the per-row lengths in
            # ModelEngine._prepare_inputs while retaining this total extent.
            self._initialize_generation_graph_metadata_extent(
                graph_attn_metadata, key, self.max_beam_width)
        if is_mixed_encoder_decoder:
            generation_query_len = key.draft_len + 1
            graph_attn_metadata.seq_lens = torch.tensor(
                (key.context_query_len, ) * key.num_contexts +
                (generation_query_len, ) *
                (num_sequences_in_batch - key.num_contexts),
                dtype=torch.int,
            )
            graph_attn_metadata.num_contexts = key.num_contexts
        assert graph_attn_metadata.is_cuda_graph

        if enable_spec_decode:
            graph_spec_metadata = spec_metadata.create_cuda_graph_metadata(
                num_sequences_in_batch)
            graph_spec_metadata.draft_tokens = draft_tokens_cuda
        else:
            graph_spec_metadata = None
        return graph_attn_metadata, graph_spec_metadata, key

    def clear_capture_only_spec_state(self) -> int:
        """Clear capture-scoped state from every cached graph SpecMetadata.

        ``create_cuda_graph_metadata`` shallow-copies the live SpecMetadata, so a
        copy made while ``_run_capture_pass(force_non_greedy=True)`` is active
        inherits ``_force_non_greedy_for_capture=True``. That copy is cached here
        and reseated as the live spec_metadata on every later replay of its graph,
        while the capture pass clears the flag on the base object only. Without
        this cleanup the copies keep the flag forever and
        ``_scan_one_model_sampling`` rewrites EVERY serving request's sampling
        params to the synthetic capture values (temperature 0.7 / top_k 50 /
        top_p 0.9), silently ignoring what the client asked for.

        The flag must NOT be cleared at copy time instead: it is load-bearing
        *during* capture. It is what makes the pass-2 populate scan non-greedy on
        parameter-less warmup requests, so that the advanced-sampling branch (not
        the argmax fast path, and with the top-k/top-p kernels present) is the one
        recorded into the graph. Clearing it here -- after the pass has captured
        every graph -- keeps capture correct and serving clean.

        Returns the number of cached metadata objects cleared.
        """
        cleared = 0
        for stored in self.graph_metadata.values():
            spec_metadata = stored.get("spec_metadata")
            if spec_metadata is not None and getattr(
                    spec_metadata, "_force_non_greedy_for_capture", False):
                spec_metadata._force_non_greedy_for_capture = False
                cleared += 1
        return cleared

    def needs_capture(self, key: KeyType):
        return self._capture_allowed and key not in self.graph_outputs

    @contextlib.contextmanager
    def allow_capture(self):
        """Context manager that enables CUDA graph capture.

        Capture is disabled by default.  On-the-fly captures outside this
        context are prevented because they can resize the shared
        cuda_graph_workspace tensor, invalidating addresses baked into
        previously captured graphs.
        """
        self._capture_allowed = True
        try:
            yield
        finally:
            self._capture_allowed = False

    @staticmethod
    def _get_memory_pool_key(key: KeyType) -> Tuple[int, int, int]:
        return (int(key.batch_size), int(key.draft_len),
                int(key.verifier_num_tokens))

    def get_graph_pool(self, key: Optional[KeyType] = None):
        """Returns the CUDA memory pool used by this graph runner.

        Returns:
            The CUDA memory pool associated with captured graphs, or None if
            no graphs have been captured yet.
        """
        if key is None:
            return self.memory_pool
        pool_key = self._get_memory_pool_key(KeyType(*key))
        if pool_key[2] == 0:
            return self.memory_pool
        return self._confidence_memory_pools.get(pool_key)

    def _get_num_tokens_for_key(self, key: KeyType) -> int:
        if key.verifier_num_tokens:
            assert key.num_contexts == 0
            return key.verifier_num_tokens
        token_per_generation = key.draft_len + 1
        return (key.num_contexts * key.context_query_len +
                (key.batch_size * self.max_beam_width - key.num_contexts) *
                 token_per_generation)

    @staticmethod
    def _get_generation_seq_lens_for_key(
            key: KeyType, max_beam_width: int) -> List[int]:
        """Return a deterministic query layout with the exact graph extent."""
        num_sequences = key.batch_size * max_beam_width
        if not key.verifier_num_tokens:
            return [key.draft_len + 1] * num_sequences
        if key.num_contexts != 0 or max_beam_width != 1:
            raise ValueError(
                "Compact verifier graphs require generation-only beam-one "
                "batches")
        retained_tokens = key.verifier_num_tokens - num_sequences
        max_retained_tokens = num_sequences * key.draft_len
        if retained_tokens < 0 or retained_tokens > max_retained_tokens:
            raise ValueError(
                "Compact verifier token budget is outside the physical "
                f"[G, G*(K+1)] extent: G={num_sequences}, "
                f"K={key.draft_len}, V={key.verifier_num_tokens}")
        base, extra = divmod(retained_tokens, num_sequences)
        return [base + 1 + (index < extra)
                for index in range(num_sequences)]

    @classmethod
    def _initialize_generation_graph_metadata_extent(
            cls, graph_attn_metadata: Any, key: KeyType,
            max_beam_width: int) -> None:
        """Bind every capture-time metadata consumer to the graph's exact V."""
        seq_lens = cls._get_generation_seq_lens_for_key(
            key, max_beam_width)
        graph_attn_metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
        expected_num_tokens = (key.verifier_num_tokens
                               or len(seq_lens) * (key.draft_len + 1))
        if graph_attn_metadata.num_tokens != expected_num_tokens:
            raise RuntimeError(
                "CUDA graph attention metadata token extent does not match "
                f"its graph key: metadata={graph_attn_metadata.num_tokens}, "
                f"key={expected_num_tokens}")

    @staticmethod
    def _assert_generation_model_token_extent(
            key: KeyType, model_inputs: Dict[str, Any]) -> None:
        """Keep embedding/residual rows and attention rows on one graph extent."""
        if key.num_contexts != 0:
            return
        expected = (key.verifier_num_tokens
                    or key.batch_size * (key.draft_len + 1))
        actual = {
            "input_ids": int(model_inputs["input_ids"].shape[0]),
            "position_ids": int(model_inputs["position_ids"].shape[-1]),
            "attention_metadata": int(model_inputs["attn_metadata"].num_tokens),
        }
        if any(extent != expected for extent in actual.values()):
            raise RuntimeError(
                "CUDA graph model-facing token extents disagree before forward: "
                f"key={expected}, extents={actual}")

    def _get_capture_static_model_inputs(
            self, key: KeyType,
            initial_inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Return graph-static model inputs sliced to the finalized key extent."""
        num_tokens = self._get_num_tokens_for_key(key)
        static_inputs = {
            "input_ids": self.shared_static_tensors["input_ids"][:num_tokens],
            "position_ids": self.shared_static_tensors["position_ids"][:,
                                                                        :num_tokens],
        }
        if self.config.use_mrope:
            static_inputs["position_ids"] = self.shared_static_tensors[
                "position_ids"][:, :, :num_tokens]
            if "mrope_delta_read_seq_slots" in initial_inputs:
                static_inputs[
                    "mrope_delta_read_seq_slots"] = self.shared_static_tensors[
                        "mrope_delta_read_seq_slots"][:key.batch_size *
                                                      self.max_beam_width]
        return static_inputs

    def capture(self,
                key: KeyType,
                forward_fn: Callable,
                initial_inputs: Dict[str, Any],
                enable_spec_decode: bool = False,
                postprocess_fn: Optional[Callable] = None) -> Any:
        """Warm up and/or capture the forward pass for a graph key."""
        # Preserve compatibility with direct callers that still pass the
        # original three-field generation-only tuple.
        key = KeyType(*key)
        batch_size = key.batch_size
        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        sliced_static_tensors = self._get_capture_static_model_inputs(
            key, initial_inputs)

        capture_inputs = initial_inputs.copy()
        capture_inputs.update(sliced_static_tensors)
        if (isinstance(self.spec_config, DSparkDecodingConfig)
                and self.spec_config.is_confidence_budget_enabled):
            self._assert_generation_model_token_extent(key, capture_inputs)
        num_encoder_tokens = key.num_encoder_tokens
        if num_encoder_tokens:
            encoder_hidden_states = initial_inputs.get("encoder_hidden_states")
            if encoder_hidden_states is None:
                raise RuntimeError("Mixed encoder-decoder CUDA graph capture "
                                   "requires encoder hidden states.")
            static_encoder_hidden_states = (
                self._get_static_encoder_hidden_states(
                    encoder_hidden_states,
                    num_encoder_tokens,
                    allow_allocate=True,
                ))
            actual_num_encoder_tokens = encoder_hidden_states.shape[0]
            if actual_num_encoder_tokens > num_encoder_tokens:
                raise RuntimeError(
                    "Mixed encoder-decoder CUDA graph capture received "
                    f"{actual_num_encoder_tokens} encoder tokens for a "
                    f"{num_encoder_tokens}-token graph.")
            static_encoder_hidden_states[:actual_num_encoder_tokens].copy_(
                encoder_hidden_states)
            static_encoder_hidden_states[actual_num_encoder_tokens:].zero_()
            capture_inputs[
                "encoder_hidden_states"] = static_encoder_hidden_states
        attn_metadata = capture_inputs["attn_metadata"]
        saved_kv_lens_cuda = _save_spec_decode_capture_state(
            attn_metadata, enable_spec_decode)

        self.graph_metadata[key] = {
            "attn_metadata": attn_metadata,
            "spec_metadata": initial_inputs.get("spec_metadata", None),
        }

        def _setup_spec_decoding_and_forward(key: KeyType, forward_fn: Callable,
                                             capture_inputs: Dict[str, Any]):
            is_first_draft = key.is_first_draft
            needs_kv_cache_recompute = True if enable_spec_decode and self.config.spec_config.spec_dec_mode.needs_kv_cache_recompute(
            ) else False
            if is_first_draft and self.config.is_draft_model and needs_kv_cache_recompute:
                capture_inputs['attn_metadata'].use_spec_decoding = True
            return forward_fn(capture_inputs)

        output = None
        with with_multi_stream(True), piecewise_cuda_graph(False):
            # We have to do a warmup run to initialize PyTorch's internal
            # states according to the docs:
            # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
            # This also lets us initialize states in the attn_metadata and
            # resize the shared attention workspace before any graph is captured.
            for _ in range(self.WARMUP_STEPS):
                output = _setup_spec_decoding_and_forward(
                    key, forward_fn, capture_inputs)
                if postprocess_fn is not None:
                    postprocess_fn(capture_inputs)
                _restore_spec_decode_capture_state(attn_metadata,
                                                   saved_kv_lens_cuda)

            if self.is_warmup_only:
                return output

            graph = torch.cuda.CUDAGraph()
            pool_key = self._get_memory_pool_key(key)
            capture_pool = (self.memory_pool if pool_key[2] == 0 else
                            self._confidence_memory_pools.get(pool_key))
            with torch.cuda.graph(graph, pool=capture_pool):
                output = _setup_spec_decoding_and_forward(
                    key, forward_fn, capture_inputs)
            if postprocess_fn is not None:
                postprocess_fn(capture_inputs)
            _restore_spec_decode_capture_state(attn_metadata,
                                               saved_kv_lens_cuda)

        self.graphs[key] = graph
        graph_output = make_weak_ref(output)
        self.graph_outputs[key] = graph_output
        captured_pool = graph.pool()
        if pool_key[2] == 0:
            self.memory_pool = captured_pool
        else:
            self._confidence_memory_pools[pool_key] = captured_pool
        return graph_output

    def replay(self, key: KeyType,
               current_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Replays a previously captured graph."""
        key = KeyType(*key)
        stored_meta = self.graph_metadata[key]
        assert current_inputs["attn_metadata"] is stored_meta["attn_metadata"]
        if stored_meta["spec_metadata"] is not None:
            assert current_inputs.get(
                "spec_metadata") is stored_meta["spec_metadata"]

        static_tensors = self.shared_static_tensors

        input_ids = current_inputs["input_ids"]
        seqlen = input_ids.shape[0]
        static_tensors["input_ids"][:seqlen].copy_(input_ids)

        position_ids = current_inputs["position_ids"]
        if self.config.use_mrope:
            static_tensors["position_ids"][:, :, :seqlen].copy_(position_ids)
            mrope_delta_read_seq_slots = current_inputs.get(
                'mrope_delta_read_seq_slots')
            if mrope_delta_read_seq_slots is not None:
                static_tensors[
                    'mrope_delta_read_seq_slots'][:mrope_delta_read_seq_slots.
                                                  shape[0]].copy_(
                                                      mrope_delta_read_seq_slots,
                                                      non_blocking=True)
        else:
            static_tensors["position_ids"][:, :seqlen].copy_(position_ids)

        num_encoder_tokens = key.num_encoder_tokens
        if num_encoder_tokens:
            encoder_hidden_states = current_inputs.get("encoder_hidden_states")
            if encoder_hidden_states is None:
                raise RuntimeError("Mixed encoder-decoder CUDA graph replay "
                                   "requires encoder hidden states.")
            actual_num_encoder_tokens = encoder_hidden_states.shape[0]
            if actual_num_encoder_tokens > num_encoder_tokens:
                raise RuntimeError(
                    "Mixed encoder-decoder CUDA graph replay received "
                    f"{actual_num_encoder_tokens} encoder tokens for a "
                    f"{num_encoder_tokens}-token graph.")
            static_encoder_hidden_states = (
                self._get_static_encoder_hidden_states(
                    encoder_hidden_states,
                    num_encoder_tokens,
                    allow_allocate=False,
                ))
            static_encoder_hidden_states[:actual_num_encoder_tokens].copy_(
                encoder_hidden_states)
            static_encoder_hidden_states[actual_num_encoder_tokens:].zero_()

        self.graphs[key].replay()
        output_ref = self.graph_outputs[key]

        return output_ref

    def _get_confidence_adp_common_batch_size(
            self, batch: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors]) -> Optional[int]:
        """Restore or choose one confidence bucket for the actual ADP group.

        ``None`` means this is not the DSpark confidence/ADP path. ``0`` means
        every peer must use eager full-K and skip confidence planning this
        iteration. A positive value is the exact execution G all peers pad to.
        A compact route is restored only from the matching iteration-owned
        device layout; ordinary full-K input chooses a new common G for the
        next plan. The exchange precedes request padding and graph keying.
        """
        self.confidence_adp_execution_batch_size = 0
        self.confidence_adp_verifier_token_budget = 0
        self.confidence_adp_route_epoch = 0
        self.confidence_force_full_k_route = False
        self.confidence_device_layout = None
        self.confidence_query_lens_host = None
        self.confidence_device_layout_row_map_valid = False
        self.confidence_discarded_device_layout = False
        if (not self.enabled or not self.padding_enabled
                or not self.config.enable_attention_dp
                or not isinstance(self.spec_config, DSparkDecodingConfig)
                or not self.spec_config.is_confidence_budget_enabled):
            return None

        generation_requests = batch.generation_requests
        all_dummy_generation = (
            batch.num_context_requests == 0 and bool(generation_requests)
            and all(request.is_dummy for request in generation_requests))
        if self._capture_allowed:
            capture_budget_active, capture_budget = (
                self.spec_config.get_confidence_capture_verifier_token_budget())
            physical_draft_lens = [
                len(request.py_draft_tokens) for request in generation_requests
            ]
            capture_k = (physical_draft_lens[0]
                         if len(set(physical_draft_lens)) == 1 else -1)
            local_capture = bool(all_dummy_generation and capture_budget_active)
            local_capture_info = [
                local_capture,
                len(generation_requests),
                capture_k,
                capture_budget is not None,
                0 if capture_budget is None else int(capture_budget),
            ]
            rank_capture_info = list(
                self.config.dist.tp_allgather(local_capture_info))
            if not rank_capture_info:
                raise RuntimeError(
                    "DSpark synthetic capture group exchange returned no peers")
            if any(
                    not isinstance(info, (list, tuple)) or len(info) != 5
                    for info in rank_capture_info):
                self.confidence_force_full_k_route = True
                raise RuntimeError(
                    "DSpark synthetic capture group exchange returned malformed "
                    f"peer payloads: {rank_capture_info}")
            if any(tuple(info) != tuple(local_capture_info)
                   for info in rank_capture_info):
                self.confidence_force_full_k_route = True
                raise RuntimeError(
                    "DSpark attention-DP peers selected different synthetic "
                    f"capture shapes: {rank_capture_info}")
            if not local_capture:
                self.confidence_force_full_k_route = True
                raise RuntimeError(
                    "DSpark synthetic capture requires an all-dummy generation "
                    "batch and an explicit capture-owned verifier budget: "
                    f"{rank_capture_info}")

            capture_g = len(generation_requests)
            if (capture_k < 0 or capture_g != batch.batch_size
                    or capture_g > self.max_supported_batch_size
                    or capture_g > self.config.batch_size):
                raise RuntimeError(
                    "DSpark synthetic capture shape exceeds runtime capacity: "
                    f"G={capture_g}, K={capture_k}")
            if capture_budget is not None:
                capture_budget = int(capture_budget)
                candidates = self.spec_config.resolve_confidence_verifier_token_budget_candidates(
                    capture_g)
                if (capture_budget not in candidates
                        or capture_budget < capture_g
                        or capture_budget > capture_g * (capture_k + 1)):
                    raise RuntimeError(
                        "DSpark synthetic capture budget is not a configured "
                        f"physical shape: G={capture_g}, K={capture_k}, "
                        f"V={capture_budget}, candidates={candidates}")
            else:
                self.confidence_force_full_k_route = True
            self.confidence_adp_execution_batch_size = capture_g
            return capture_g

        carried_layout = getattr(new_tensors_device,
                                 "dspark_confidence_layout", None)
        carried_epoch = getattr(new_tensors_device,
                                "dspark_confidence_route_epoch", None)
        if carried_layout is not None or carried_epoch is not None:
            carried_g = int(
                getattr(new_tensors_device,
                        "dspark_confidence_execution_batch_size", 0) or 0)
            carried_k = int(
                getattr(new_tensors_device,
                        "dspark_confidence_physical_draft_len", 0) or 0)
            carried_epoch = int(carried_epoch or 0)
            carried_engine_generation = int(
                getattr(new_tensors_device,
                        "dspark_confidence_engine_generation", 0) or 0)
            carried_v = getattr(
                new_tensors_device,
                "dspark_confidence_verifier_token_budget", None)
            budget_event = getattr(
                new_tensors_device,
                "dspark_confidence_budget_ready_event", None)
            budget_host = getattr(
                new_tensors_device,
                "dspark_confidence_verifier_token_budget_host", None)
            semantics_host = getattr(
                new_tensors_device,
                "dspark_confidence_semantics_host", None)
            query_lens_host = getattr(
                new_tensors_device,
                "dspark_confidence_query_lens_host", None)
            if (budget_event is not None
                    and (budget_host is not None
                         or semantics_host is not None
                         or query_lens_host is not None)):
                # The device-selected V, semantic scalars, and query lengths
                # are copied into sampler-owned pinned storage beside the
                # layout. Their small
                # D2H overlaps scheduling/input preparation; wait only here,
                # immediately before Python graph keying.
                budget_event.synchronize()
            if carried_v is None and budget_host is not None:
                carried_v = int(budget_host)
            carried_v = int(carried_v or 0)
            semantic_values = (0, 0, 0, 0, 0, 0, 0)
            semantic_payload_ready = False
            if semantics_host is not None:
                flattened_semantics = tuple(
                    int(value) for value in semantics_host.reshape(-1).tolist())
                if len(flattened_semantics) == 7:
                    semantic_values = flattened_semantics
                    semantic_payload_ready = True
            (semantic_valid, row_map_valid, retained_count, query_count,
             cu_query_count, semantic_real_count, declared_v) = semantic_values
            current_ids = tuple(request.py_request_id
                                for request in generation_requests
                                if not request.is_dummy)
            current_slots = tuple(request.py_seq_slot
                                  for request in generation_requests
                                  if not request.is_dummy)
            planned_ids = tuple(
                getattr(new_tensors_device,
                        "dspark_confidence_request_ids", ()) or ())
            planned_slots = tuple(
                getattr(new_tensors_device,
                        "dspark_confidence_seq_slots", ()) or ())
            real_count = len(current_ids)
            host_query_lens_exact = False
            if (isinstance(query_lens_host, torch.Tensor)
                    and query_lens_host.device.type == "cpu"
                    and query_lens_host.dtype == torch.int32
                    and query_lens_host.numel() == carried_g
                    and carried_g >= real_count):
                flattened_query_lens = query_lens_host.reshape(-1)
                real_query_lens = flattened_query_lens[:real_count]
                dummy_query_lens = flattened_query_lens[real_count:]
                host_query_lens_exact = bool(
                    torch.all(real_query_lens >= 1)
                    and torch.all(real_query_lens <= carried_k + 1)
                    and (dummy_query_lens.numel() == 0
                         or torch.all(dummy_query_lens == 1))
                    and int(flattened_query_lens.sum()) == declared_v)
            roster_matches = (current_ids == planned_ids
                              and current_slots == planned_slots)
            physical_lens = [
                len(request.py_draft_tokens) for request in generation_requests
                if not request.is_dummy
            ]
            physical_block_recoverable = bool(
                real_count == 0
                or (
                    physical_lens
                    and all(length == carried_k for length in physical_lens)
                )
            )
            configured_budgets = (
                self.spec_config.
                resolve_confidence_verifier_token_budget_candidates(carried_g)
                if carried_g > 0 else ())
            base_layout_ready = bool(
                batch.num_context_requests == 0 and carried_layout is not None
                and carried_g >= real_count and real_count > 0
                and declared_v >= carried_g and carried_k > 0
                and declared_v <= carried_g * (carried_k + 1)
                and declared_v in configured_budgets
                and carried_g <= self.max_supported_batch_size
                and carried_g <= self.config.batch_size
                and carried_epoch > int(
                    getattr(self, "_confidence_last_route_epoch", 0))
                and carried_engine_generation == self.confidence_engine_generation
                and roster_matches
                and len(set(planned_ids)) == real_count
                and len(set(planned_slots)) == real_count
                and all(slot is not None for slot in planned_slots)
                and physical_lens
                and all(length == carried_k for length in physical_lens)
                and carried_layout.retained_lens.shape == (carried_g, )
                and carried_layout.query_lens.shape == (carried_g, )
                and carried_layout.cu_query_lens.shape == (carried_g + 1, )
                and carried_layout.real_request_mask.shape == (carried_g, )
                and carried_layout.source_batch_indices.shape == (carried_g, )
                and carried_layout.row_map_valid.shape == ()
                and carried_layout.packed_to_dense.shape[0] >= declared_v
                and carried_layout.packed_draft_indices.shape[0] >=
                declared_v - carried_g)
            semantic_exact = bool(
                semantic_payload_ready and semantic_valid == 1
                and row_map_valid == 1
                and host_query_lens_exact
                and carried_v == declared_v
                and retained_count == declared_v - carried_g
                and query_count == declared_v
                and cu_query_count == declared_v
                and semantic_real_count == real_count)
            layout_shapes_ready = base_layout_ready and semantic_exact
            local_route = [
                layout_shapes_ready, base_layout_ready, semantic_exact,
                carried_g, carried_v, carried_k, carried_epoch, real_count,
                semantic_valid, row_map_valid, retained_count, query_count,
                cu_query_count, semantic_real_count, declared_v
            ]
            rank_routes = list(self.config.dist.tp_allgather(local_route))
            if any(
                    not isinstance(route, (list, tuple)) or len(route) != 15
                    for route in rank_routes):
                self.confidence_force_full_k_route = True
                raise RuntimeError(
                    "DSpark confidence semantic exchange returned malformed "
                    f"peer payloads: {rank_routes}")
            unanimous = bool(
                rank_routes and all(bool(route[0]) for route in rank_routes)
                and all(tuple(route[3:]) == tuple(rank_routes[0][3:])
                        for route in rank_routes))
            if not unanimous:
                common_semantic_provenance = bool(
                    rank_routes and all(bool(route[1]) for route in rank_routes)
                    and all((route[3], route[5], route[6], route[7], route[14])
                            == (rank_routes[0][3], rank_routes[0][5],
                                rank_routes[0][6], rank_routes[0][7],
                                rank_routes[0][14])
                            for route in rank_routes))
                semantic_tuples_agree = bool(
                    rank_routes and all(
                        (route[4], *route[8:15]) ==
                        (rank_routes[0][4], *rank_routes[0][8:15])
                        for route in rank_routes))
                if (common_semantic_provenance
                        and (not all(bool(route[2]) for route in rank_routes)
                             or not semantic_tuples_agree)):
                    # The same physical-K proposal block is still available.
                    # Reject the compact payload uniformly before padding or
                    # index_select, and let this iteration run ordinary V0.
                    self.confidence_discarded_device_layout = True
                    self.confidence_force_full_k_route = True
                    for request in generation_requests:
                        if not request.is_dummy:
                            request.py_draft_tokens_effective_len = len(
                                request.py_draft_tokens)
                            request.py_dspark_confidence_route_epoch = None
                            request.py_dspark_confidence_execution_batch_size = None
                            request.py_dspark_confidence_verifier_token_budget = None
                    return 0
                compact_payload = any(
                    int(route[14]) > 0 and int(route[14]) <
                    int(route[3]) * (int(route[5]) + 1)
                    for route in rank_routes) if rank_routes else declared_v > 0
                if compact_payload:
                    # A completed/reordered roster may safely discard the old
                    # compact plan because the carrier still owns a physical-K
                    # proposal block. Stale or asymmetric provenance for the
                    # same roster is still an unrecoverable late-route error.
                    common_carried_shape = bool(
                        rank_routes and all(
                            (route[3], route[5], route[6], route[14]) ==
                            (rank_routes[0][3], rank_routes[0][5],
                             rank_routes[0][6], rank_routes[0][14])
                            for route in rank_routes))
                    peer_roster_transition = bool(
                        common_carried_shape and
                        (len({int(route[7]) for route in rank_routes}) > 1
                         or any(bool(route[1]) != bool(rank_routes[0][1])
                                 for route in rank_routes)))
                    local_discardable = bool(
                        (not roster_matches or peer_roster_transition)
                        and batch.num_context_requests == 0
                        and carried_engine_generation ==
                        self.confidence_engine_generation
                        and carried_epoch > int(
                            getattr(self, "_confidence_last_route_epoch", 0))
                        and physical_block_recoverable)
                    discard_info = list(
                        self.config.dist.tp_allgather(
                            [local_discardable, real_count]))
                    if (common_carried_shape and discard_info
                            and all(bool(info[0]) for info in discard_info)):
                        self.confidence_discarded_device_layout = True
                        for request in generation_requests:
                            if not request.is_dummy:
                                request.py_draft_tokens_effective_len = len(
                                    request.py_draft_tokens)
                                request.py_dspark_confidence_route_epoch = None
                                request.py_dspark_confidence_execution_batch_size = None
                                request.py_dspark_confidence_verifier_token_budget = None
                    else:
                        self.confidence_force_full_k_route = True
                        raise RuntimeError(
                            "DSpark compact device layout provenance is missing, "
                            "stale, or asymmetric across ADP peers: "
                            f"{rank_routes}")
                else:
                    self.confidence_force_full_k_route = True
                    return 0
            else:
                self.confidence_device_layout = carried_layout
                self.confidence_query_lens_host = query_lens_host
                self.confidence_device_layout_row_map_valid = True
                self.confidence_adp_execution_batch_size = carried_g
                self.confidence_adp_verifier_token_budget = carried_v
                self.confidence_adp_route_epoch = carried_epoch
                self._confidence_last_route_epoch = carried_epoch
                return carried_g

        real_requests = [
            request for request in batch.generation_requests
            if not request.is_dummy
        ]
        local_compact_request_state = any(
            getattr(request, "py_draft_tokens_effective_len", None)
            is not None
            and int(request.py_draft_tokens_effective_len)
            != len(request.py_draft_tokens) for request in real_requests)
        incomplete_device_plan = bool(
            new_tensors_device is not None
            and getattr(new_tensors_device, "next_draft_lens", None) is not None
            and not self.confidence_discarded_device_layout)
        local_ready = (
            batch.num_context_requests == 0 and bool(real_requests)
            and all(len(request.py_draft_tokens) ==
                    self.spec_config.max_draft_len
                    for request in real_requests)
            and not local_compact_request_state
            and not incomplete_device_plan)
        rank_info = list(
            self.config.dist.tp_allgather(
                [local_ready, len(real_requests),
                 local_compact_request_state or incomplete_device_plan]))
        if not rank_info:
            raise RuntimeError(
                "DSpark attention-DP group exchange returned no peers")

        if any(bool(info[2]) for info in rank_info):
            self.confidence_force_full_k_route = True
            raise RuntimeError(
                "DSpark compact draft lengths arrived without their exact "
                f"iteration-owned device layout provenance: {rank_info}")

        if not all(bool(info[0]) for info in rank_info):
            self.confidence_force_full_k_route = True
            return 0

        max_real_requests = max(int(info[1]) for info in rank_info)
        # Partial confidence coverage must not inflate every smaller ordinary
        # graph into the next configured confidence bucket.  For example, a
        # profile that enables compact verification only at G=128 still owns
        # ordinary full-K graphs at G=16/32/64.  Pick the nearest captured G
        # first; the sampler will plan a compact successor only when that exact
        # G has configured verifier-budget candidates.
        configured_batches = sorted(self.supported_batch_sizes)
        common_batch = next(
            (graph_batch_size for graph_batch_size in configured_batches
             if graph_batch_size >= max_real_requests), 0)
        if (common_batch <= 0
                or common_batch > self.max_supported_batch_size
                or common_batch > self.config.batch_size
                or batch.batch_size > common_batch):
            return 0
        self.confidence_adp_execution_batch_size = common_batch
        self.confidence_force_full_k_route = True
        return common_batch

    def _get_padded_batch(
            self, batch: ScheduledRequests, resource_manager: ResourceManager,
            runtime_draft_len: int,
            new_tensors_device: Optional[SampleStateTensors] = None) -> int:
        can_run_cuda_graph = self._can_run_cuda_graph_batch(batch)
        batch_size = batch.batch_size
        new_batch_size = batch_size
        common_confidence_batch = (
            CUDAGraphRunner._get_confidence_adp_common_batch_size(
                self, batch, new_tensors_device))
        self.confidence_adp_plan_ready = common_confidence_batch != 0

        if common_confidence_batch is not None:
            if common_confidence_batch == 0:
                return 0
            # A generation-only DSpark confidence batch can be ineligible only
            # because its real N is not itself a captured graph size. The
            # readiness exchange above proves every peer can instead use the
            # same captured G before any confidence plan/draft is made.
            can_run_cuda_graph = True
            new_batch_size = common_confidence_batch
        elif self.enabled and self.config.enable_attention_dp:
            graph_batch_info = self.config.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            all_can_run_cuda_graph = all(rank_info[0]
                                         for rank_info in graph_batch_info)
            if all_can_run_cuda_graph:
                new_batch_size = max(rank_info[1]
                                     for rank_info in graph_batch_info)

        if (not self.enabled or not self.padding_enabled
                or not can_run_cuda_graph
                or new_batch_size > self.max_supported_batch_size):
            return 0

        # Pad the batch size up to the nearest existing graph for the runtime
        # draft length. With dynamic draft length (one-model path) the runtime
        # draft length is the source of truth for which graphs are eligible;
        # otherwise this reduces to plain batch-size rounding.
        padded_batch_size = self._round_up_batch_size_with_draft_len(
            new_batch_size, runtime_draft_len)
        if (common_confidence_batch is not None
                and padded_batch_size != common_confidence_batch):
            raise RuntimeError(
                "DSpark common ADP batch did not resolve to the same captured "
                f"batch: agreed={common_confidence_batch}, "
                f"resolved={padded_batch_size}")

        if batch_size == padded_batch_size:
            return 0

        padding_size = padded_batch_size - batch_size
        if padding_size <= 0:
            return 0
        if padding_size + batch.batch_size > self.config.batch_size:
            return 0

        # No padding if it would create too many concurrent requests.
        # This is not strictly required, but we should probably
        # respect the requirement just in case that changes in the future.
        # Use per-draft-len dummy requests for dynamic draft length support.
        padding_dummy_request = self._get_or_create_padding_dummy(
            resource_manager, runtime_draft_len)
        if padding_dummy_request is None:
            logger.warning_once(
                "Failed to allocate the CUDA graph padding dummy request "
                f"(draft_len={runtime_draft_len}) from a saturated KV cache; "
                "falling back to eager mode for padded batches.",
                key=f"cuda_graph_padding_dummy_fallback_{runtime_draft_len}")
            return 0

        if (not self.config.is_draft_model
                and isinstance(self.spec_config, DSparkDecodingConfig)
                and self.spec_config.is_confidence_budget_enabled):
            # Runtime padding contributes one mandatory verifier anchor but no
            # draft tokens. Reset on every use because the shared dummy object
            # is realigned to physical K whenever the final route falls back to
            # V=0/eager. Synthetic capture does not use this padding path and
            # retains its separately generated nonzero compact lengths.
            padding_dummy_request.py_draft_tokens_effective_len = (
                len(padding_dummy_request.py_draft_tokens)
                if self.confidence_force_full_k_route else 0)

        batch.generation_requests.extend([padding_dummy_request] * padding_size)
        return padding_size

    def _get_or_create_padding_dummy(
            self, resource_manager: ResourceManager,
            runtime_draft_len: int) -> Optional[LlmRequest]:
        """Returns the padding dummy request for the given draft length,
        allocating it from the KV cache manager on first use.

        Returns None when the KV cache manager cannot allocate the dummy
        (e.g. saturated cache); the batch cannot be padded until a retry
        succeeds.
        """
        if runtime_draft_len in self.padding_dummy_requests:
            return self.padding_dummy_requests[runtime_draft_len]

        kv_cache_manager = resource_manager.get_resource_manager(
            self.config.kv_cache_manager_key)

        runtime_tokens_per_gen_step = (
            self.spec_config.get_runtime_tokens_per_gen_step(runtime_draft_len)
            if self.spec_config is not None else 1 + runtime_draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1

        dummy_encoder_output_len = None
        if self.is_encoder_decoder:
            cross_kv_cache_manager = resource_manager.get_resource_manager(
                ResourceManagerType.CROSS_KV_CACHE_MANAGER)
            if cross_kv_cache_manager is None:
                return None
            dummy_encoder_output_len = self._get_padding_dummy_encoder_output_len(
                cross_kv_cache_manager)

        # Get draft KV cache manager only for one-model speculative decoding.
        # In two-model mode, each model has its own KV cache manager, so
        # draft_kv_cache_manager should be None.
        draft_kv_cache_manager = get_draft_kv_cache_manager(
            self.spec_config, resource_manager)

        # Use unique dummy request ID per draft length
        dummy_request_id = CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len
        dummy_request = kv_cache_manager.add_dummy_requests(
            [dummy_request_id],
            token_nums=[ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM]
            if self.is_encoder_decoder else None,
            is_gen=True,
            max_num_draft_tokens=runtime_draft_token_buffer_width,
            use_mrope=self.config.use_mrope,
            max_beam_width=self.config.max_beam_width,
            encoder_output_lens=[dummy_encoder_output_len]
            if dummy_encoder_output_len is not None else None,
            draft_kv_cache_manager=draft_kv_cache_manager)

        if dummy_request is None:
            return None
        dummy_request = dummy_request[0]
        dummy_request.is_cuda_graph_dummy = True
        if self.is_encoder_decoder:
            if not self._add_cross_dummy_request(
                    dummy_request, resource_manager, dummy_encoder_output_len,
                    draft_kv_cache_manager):
                return None

        spec_res_mgr = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if spec_res_mgr:
            spec_res_mgr.add_dummy_requests([dummy_request_id])
        self.padding_dummy_requests[runtime_draft_len] = dummy_request
        return dummy_request

    def _padding_dummy_managers(
            self,
            resource_manager: ResourceManager) -> List[BaseResourceManager]:
        """The managers ``_get_or_create_padding_dummy`` registers a dummy with.

        Kept next to the creation path so the two stay in step.  Duplicates are
        dropped by identity: freeing the same manager twice for one request is
        not safe in general.
        """
        candidates = [
            resource_manager.get_resource_manager(
                self.config.kv_cache_manager_key),
            get_draft_kv_cache_manager(self.spec_config, resource_manager),
            resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER),
        ]
        if self.is_encoder_decoder:
            candidates.append(
                resource_manager.get_resource_manager(
                    ResourceManagerType.CROSS_KV_CACHE_MANAGER))

        managers: List[BaseResourceManager] = []
        for manager in candidates:
            if manager is not None and not any(manager is seen
                                               for seen in managers):
                managers.append(manager)
        return managers

    def release_padding_dummy(self, resource_manager: ResourceManager,
                              runtime_draft_len: int) -> bool:
        """Releases the padding dummy for ``runtime_draft_len`` from every
        manager that allocated part of it, and drops it from the runner so a
        later padded step re-creates it.

        One dummy request ID is spread across up to four managers -- the main
        KV cache manager, the one-model draft KV cache manager, the
        speculative resource manager slot and the encoder-decoder cross-KV
        cache manager.  Releasing only the main one leaves the others holding
        the ID, and re-creation reuses the same
        ``CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len``.

        Returns True if a dummy was held for that draft length.
        """
        dummy_request = self.padding_dummy_requests.pop(runtime_draft_len, None)
        if dummy_request is None:
            return False
        for manager in self._padding_dummy_managers(resource_manager):
            manager.free_resources(dummy_request)
        return True

    def _can_pad_any_batch(self, runtime_draft_len: int) -> bool:
        """Returns True when _get_padded_batch can pad at least one feasible
        batch size for the given draft length (mirrors its rounding and
        capacity guards). For example, with graph batch sizes [1, 2, 4] and
        max batch size 1, every batch already matches a graph size and no
        padding dummy is ever needed.
        """
        max_unpadded_batch_size = min(self.config.batch_size,
                                      self.max_supported_batch_size)
        for batch_size in range(1, max_unpadded_batch_size + 1):
            padded = self._round_up_batch_size_with_draft_len(
                batch_size, runtime_draft_len)
            if batch_size < padded <= self.config.batch_size:
                return True
        return False

    def preallocate_padding_dummies(self,
                                    resource_manager: ResourceManager) -> None:
        """Eagerly allocates the padding dummies while the KV cache still has
        free blocks (called at the end of ModelEngine.warmup, after graph
        capture).

        _get_padded_batch otherwise allocates the dummies lazily at the first
        padded step; when the KV cache is already saturated by then, the
        allocation fails on every step and padded batches permanently fall
        back to eager mode.

        Only the draft lengths of captured graphs are preallocated — those are
        exactly the ones runtime padding requests — and only when padding can
        actually occur for that draft length. Anything else would permanently
        hold KV blocks (and spec/hybrid cache slots) that the lazy path never
        consumes, regressing tightly-sized deployments.
        """
        if not (self.enabled and self.padding_enabled):
            return
        kv_cache_manager = resource_manager.get_resource_manager(
            self.config.kv_cache_manager_key)
        if kv_cache_manager is None or getattr(kv_cache_manager,
                                               'is_estimating_kv_cache', False):
            # The estimation-phase KV cache is sized with no headroom for
            # retained dummies; holding blocks there can leave the estimation
            # requests unschedulable. That executor is discarded anyway, so
            # preallocation only matters for the final one.
            return
        for draft_len in sorted({key[1] for key in self.graphs}):
            if not self._can_pad_any_batch(draft_len):
                continue
            if self._get_or_create_padding_dummy(resource_manager,
                                                 draft_len) is None:
                logger.warning(
                    "Could not pre-allocate the CUDA graph padding dummy "
                    f"request (draft_len={draft_len}) at warmup; allocation "
                    "will be retried at the first padded step.")
            else:
                logger.info(
                    "Pre-allocated the CUDA graph padding dummy request "
                    f"(draft_len={draft_len}) at warmup.")

    def _add_cross_dummy_request(
            self, dummy_request: LlmRequest, resource_manager: ResourceManager,
            encoder_output_len: int,
            draft_kv_cache_manager: Optional[BaseResourceManager]) -> bool:
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER)
        if cross_kv_cache_manager is None:
            return False

        dummy_request.py_encoder_output = None
        dummy_request.py_skip_cross_kv_projection = True

        encoder_output_lens = [encoder_output_len]
        cross_dummy_requests = cross_kv_cache_manager.add_dummy_requests(
            request_ids=[dummy_request.py_request_id],
            token_nums=encoder_output_lens,
            is_gen=True,
            max_beam_width=self.config.max_beam_width,
            encoder_output_lens=encoder_output_lens)
        if cross_dummy_requests is not None:
            return True

        kv_cache_manager = resource_manager.get_resource_manager(
            self.config.kv_cache_manager_key)
        kv_cache_manager.free_resources(dummy_request)
        if draft_kv_cache_manager is not None:
            draft_kv_cache_manager.free_resources(dummy_request)
        return False

    @staticmethod
    def _get_padding_dummy_encoder_output_len(
            cross_kv_cache_manager: Any) -> int:
        encoder_output_len = 1
        max_seq_len = getattr(cross_kv_cache_manager, "max_seq_len", None)
        if max_seq_len is not None:
            encoder_output_len = min(encoder_output_len, int(max_seq_len))
        return encoder_output_len

    def _round_up_batch_size(self, batch_size: int) -> int:
        """Finds the smallest supported graph batch size >= the given size."""
        if not self.supported_batch_sizes:
            return 0
        idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        if idx == len(self.supported_batch_sizes):
            return 0
        return self.supported_batch_sizes[idx]

    def _round_up_batch_size_with_draft_len(self, batch_size: int,
                                            draft_len: int) -> int:
        """Finds the smallest graph batch size >= batch_size that also matches the given draft_len.

        The dynamic draft length mapping exists exactly when the dynamic draft
        length feature is active (see _compute_dynamic_draft_len_mapping);
        without it, this ignores draft_len and reduces to plain batch-size
        rounding.
        """
        if not self.dynamic_draft_len_mapping:
            return self._round_up_batch_size(batch_size)

        start_idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        # Negate the list to make it non-decreasing for bisect
        # (draft_len decreases as batch_size increases in the schedule)
        draft_lens = [
            self.dynamic_draft_len_mapping.get(self.supported_batch_sizes[i], 0)
            for i in range(start_idx, len(self.supported_batch_sizes))
        ]
        idx = bisect.bisect_left(draft_lens, -draft_len, key=lambda x: -x)
        if idx < len(draft_lens) and draft_lens[idx] == draft_len:
            return self.supported_batch_sizes[start_idx + idx]
        # No suitable graph found
        return 0

    @contextlib.contextmanager
    def pad_batch(
            self, scheduled_requests: ScheduledRequests,
            resource_manager: ResourceManager, runtime_draft_len: int = 0,
            new_tensors_device: Optional[SampleStateTensors] = None):
        """Context manager to pad a batch to a graph-compatible size."""
        padding_size = self._get_padded_batch(
            scheduled_requests, resource_manager, runtime_draft_len,
            new_tensors_device)
        try:
            yield scheduled_requests
        finally:
            if padding_size > 0:
                scheduled_requests.generation_requests = scheduled_requests.generation_requests[:
                                                                                                -padding_size]

    def clear(self):
        """Releases all captured graphs and the associated memory pool."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        self.padding_dummy_requests = {}
        self._confidence_memory_pools.clear()
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()


EncoderKeyType: TypeAlias = Tuple[int, int, int]
_ENCODER_SOURCE_SEQ_LENS = "_encoder_source_seq_lens"
_ENCODER_SOURCE_TO_SLOT = "_encoder_source_to_slot"


@dataclass
class EncoderCUDAGraphRunnerConfig:
    """Configuration for EncoderCUDAGraphRunner."""
    use_cuda_graph: bool
    cuda_graph_padding_enabled: bool
    cuda_graph_batch_sizes: List[int]
    cuda_graph_num_tokens: List[int]
    cuda_graph_seq_lens: List[int]
    max_cuda_graph_batch_size: int
    max_cuda_graph_num_tokens: int
    max_num_tokens: int
    max_seq_len: int
    cuda_graph_mem_pool: Any
    is_encoder_decoder: bool = False
    use_fixed_sequence_slots: bool = False


class EncoderCUDAGraphRunner:
    """CUDA graph runner for no-cache encoder forward passes.

    Designed for encoder inputs with `input_ids` (flat [total_tokens]) and
    `seq_lens` ([batch_size]). Encoder CUDA graphs are keyed on the 3-tuple
    (padded_batch_size, padded_total_tokens, max_seq_len_bucket) for dynamic
    encoder-decoder batches when padding is enabled.

    Restricted to `TrtllmAttentionMetadata`: FlashInfer's per-batch planner
    state is not compatible with CUDA graph capture/replay.
    """

    WARMUP_STEPS = 1

    def __init__(self, config: EncoderCUDAGraphRunnerConfig):
        self.config = config

        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = sorted(config.cuda_graph_batch_sizes)
        self.max_supported_batch_size = config.max_cuda_graph_batch_size
        self.supported_num_tokens = sorted(config.cuda_graph_num_tokens)
        self.max_supported_num_tokens = config.max_cuda_graph_num_tokens
        self.supported_seq_lens = sorted(config.cuda_graph_seq_lens)
        self.is_encoder_decoder = config.is_encoder_decoder
        self.use_fixed_sequence_slots = config.use_fixed_sequence_slots
        self.capture_keys: frozenset[EncoderKeyType] = frozenset()
        self._capture_sequence_lengths: Dict[EncoderKeyType, List[int]] = {}
        if self.is_encoder_decoder:
            self._capture_sequence_lengths = (
                self._build_encoder_decoder_capture_layouts())
            self.capture_keys = frozenset(self._capture_sequence_lengths)
        self._capture_keys_by_batch_size: Dict[int, List[EncoderKeyType]] = {}
        for key in sorted(self.capture_keys):
            self._capture_keys_by_batch_size.setdefault(key[0], []).append(key)

        self.graphs: Dict[EncoderKeyType, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[EncoderKeyType, Callable[[],
                                                          Optional[Any]]] = {}
        self.graph_metadata: Dict[EncoderKeyType, Dict[str, Any]] = {}
        self.memory_pool = config.cuda_graph_mem_pool

        self.shared_static_tensors: Dict[str, torch.Tensor] = {}
        self.shared_static_tensors_cpu: Dict[str, torch.Tensor] = {}
        if self.enabled:
            self._create_shared_static_tensors()
        self.cuda_graph_meta_buffers = (Buffers() if self.is_encoder_decoder
                                        else get_memory_buffers())

        self._capture_allowed = False
        self.is_warmup_only = False
        self._staging_retirement_event: Optional[torch.cuda.Event] = None

        # CUDA graph H2D memcpy nodes require pinned host sources. In CC mode
        # prefer_pinned() is false: pageable host buffers are preferred, so the
        # H2D copies must be issued before graph replay instead of captured.
        self._capture_h2d_copy = prefer_pinned()

    def _create_shared_static_tensors(self):
        """Allocates static tensors sized for the largest supported num_tokens."""
        max_total_tokens = (
            self.config.max_num_tokens if self.is_encoder_decoder else min(
                self.max_supported_num_tokens, self.config.max_num_tokens))
        max_batch_size = self.max_supported_batch_size

        self.shared_static_tensors = {
            "input_ids":
            torch.ones((max_total_tokens, ), device="cuda", dtype=torch.int32),
            "position_ids":
            torch.zeros((1, max_total_tokens), device="cuda",
                        dtype=torch.int32),
        }
        self.shared_static_tensors_cpu = {
            "input_ids":
            torch.ones((max_total_tokens, ),
                       device="cpu",
                       dtype=torch.int32,
                       pin_memory=prefer_pinned()),
            "position_ids":
            torch.zeros((1, max_total_tokens),
                        device="cpu",
                        dtype=torch.int32,
                        pin_memory=prefer_pinned()),
            # Pinned static buffer for seq_lens. Each captured graph's attn_metadata._seq_lens
            # is reseated (in maybe_get_cuda_graph) to a stable slice of this buffer, and the
            # corresponding H2D copy into _seq_lens_cuda is captured inside the graph itself.
            "seq_lens":
            torch.ones((max_batch_size, ),
                       device="cpu",
                       dtype=torch.int32,
                       pin_memory=prefer_pinned()),
        }

        # Cached arange used by replay() to build packed position_ids in-place via slice copies.
        self._arange_max = torch.arange(max_total_tokens, dtype=torch.int32)

    @staticmethod
    def _round_up(value: int, supported: List[int]) -> int:
        """Smallest element of `supported` >= value, or 0 if none exists."""
        if not supported:
            return 0
        idx = bisect.bisect_left(supported, value)
        if idx == len(supported):
            return 0
        return supported[idx]

    @staticmethod
    def build_capture_sequence_lengths(batch_size: int, num_tokens: int,
                                       max_seq_len: int) -> Optional[List[int]]:
        """Build a real sequence layout for a configured encoder bucket."""
        if (batch_size <= 0 or num_tokens < batch_size
                or num_tokens > batch_size * max_seq_len):
            return None

        if batch_size == 1:
            return [num_tokens]

        if num_tokens >= max_seq_len + batch_size - 1:
            remaining_tokens = num_tokens - max_seq_len
            base, extra = divmod(remaining_tokens, batch_size - 1)
            return ([max_seq_len] + [base + 1] * extra + [base] *
                    (batch_size - 1 - extra))

        return [num_tokens - batch_size + 1] + [1] * (batch_size - 1)

    def _build_encoder_decoder_capture_layouts(
            self) -> Dict[EncoderKeyType, List[int]]:
        """Map each reachable graph key to one physical sequence-slot layout.

        Sequence layout is deliberately not part of ``EncoderKeyType``. Multiple
        configured shapes may normalize to the same three-dimensional key, so
        the first deterministic layout becomes that graph's fixed slot
        capacities. Runtime sequences are assigned to those slots during replay.
        """
        capture_layouts: Dict[EncoderKeyType, List[int]] = {}
        for batch_size in self.supported_batch_sizes:
            for num_tokens in self.supported_num_tokens:
                for max_seq_len in self.supported_seq_lens:
                    sequence_lengths = self.build_capture_sequence_lengths(
                        batch_size, num_tokens, max_seq_len)
                    if sequence_lengths is None:
                        continue

                    key, _, is_valid = self.get_graph_key(
                        {"seq_lens": sequence_lengths})
                    if is_valid:
                        # Capture at most one graph/layout for a normalized key;
                        # alternative runtime layouts do not create more keys.
                        capture_layouts.setdefault(key, sequence_lengths)

        return capture_layouts

    def _get_dynamic_capture_key(
        self,
        sequence_lengths: List[int],
        allow_batch_padding: bool,
    ) -> Optional[EncoderKeyType]:
        """Return the smallest key whose token bucket and slots fit the batch.

        Keys are ordered from smaller to larger buckets. For fixed-slot replay,
        aggregate token and maximum-length checks are insufficient: every
        runtime sequence (including batch-padding dummies) must also fit in a
        distinct capture-time slot. An incompatible key is skipped in favor of
        a larger existing key; no new layout-specific key is created.
        """
        batch_size = len(sequence_lengths)
        max_seq_len = max(sequence_lengths) if sequence_lengths else 0
        candidate_batch_sizes = (self.supported_batch_sizes
                                 if allow_batch_padding else [batch_size])
        for padded_batch_size in candidate_batch_sizes:
            if padded_batch_size < batch_size:
                continue

            padded_sequence_lengths = (sequence_lengths + [1] *
                                       (padded_batch_size - batch_size))
            required_num_tokens = sum(padded_sequence_lengths)
            for key in self._capture_keys_by_batch_size.get(
                    padded_batch_size, []):
                _, padded_num_tokens, padded_max_seq_len = key
                if (padded_num_tokens < required_num_tokens
                        or padded_num_tokens > self.max_supported_num_tokens
                        or padded_max_seq_len < max_seq_len
                        or padded_max_seq_len not in self.supported_seq_lens
                        or padded_num_tokens
                        > padded_batch_size * padded_max_seq_len):
                    continue
                if (self.use_fixed_sequence_slots
                        and self._get_sequence_slot_mapping(
                            key, padded_sequence_lengths) is None):
                    # The batch fits the aggregate bucket but not this key's
                    # individual slot capacities. Try the next captured key.
                    continue
                return key

        return None

    def _get_sequence_slot_mapping(
        self,
        key: EncoderKeyType,
        sequence_lengths: List[int],
    ) -> Optional[List[int]]:
        """Assign each runtime sequence to one compatible physical graph slot.

        The returned list maps source request index to capture slot index. It
        changes only physical placement: source request order is retained
        separately and restored after replay.
        """
        capture_lengths = self._capture_sequence_lengths.get(key)
        if (capture_lengths is None
                or len(capture_lengths) != len(sequence_lengths)):
            return None

        # Preserve physical order when every request already fits its
        # corresponding slot, avoiding unnecessary scatter/gather permutation.
        if all(sequence_length <= capture_length
               for sequence_length, capture_length in zip(
                   sequence_lengths, capture_lengths)):
            return list(range(len(sequence_lengths)))

        # Largest-to-largest matching is sufficient for one-to-one scalar
        # capacities: if any sorted request exceeds its paired slot, no
        # permutation can make the layout fit.
        sequence_order = sorted(range(len(sequence_lengths)),
                                key=lambda index:
                                (-sequence_lengths[index], index))
        capture_order = sorted(range(len(capture_lengths)),
                               key=lambda index:
                               (-capture_lengths[index], index))
        source_to_slot = [0] * len(sequence_lengths)
        for source_index, slot_index in zip(sequence_order, capture_order):
            if sequence_lengths[source_index] > capture_lengths[slot_index]:
                return None
            source_to_slot[source_index] = slot_index
        return source_to_slot

    def get_capture_warmup_sequence_lengths(
            self, key: EncoderKeyType) -> Optional[List[int]]:
        """Return the representative sequence layout for a capture key."""
        sequence_lengths = self._capture_sequence_lengths.get(key)
        return list(sequence_lengths) if sequence_lengths is not None else None

    def _get_capture_sequence_offsets(self, key: EncoderKeyType) -> List[int]:
        """Return cumulative fixed-slot offsets for a capture layout."""
        offsets = [0]
        for sequence_length in self._capture_sequence_lengths[key]:
            offsets.append(offsets[-1] + sequence_length)
        if offsets[-1] != key[1]:
            raise ValueError(
                f"Encoder CUDA graph layout for key {key} contains "
                f"{offsets[-1]} tokens.")
        return offsets

    def _get_valid_graph_key(self, batch_size: int, num_tokens: int,
                             max_seq_len: int) -> EncoderKeyType:
        num_tokens_idx = bisect.bisect_left(self.supported_num_tokens,
                                            num_tokens)
        seq_len_idx = bisect.bisect_left(self.supported_seq_lens, max_seq_len)

        while (num_tokens_idx < len(self.supported_num_tokens)
               and seq_len_idx < len(self.supported_seq_lens)):
            padded_num_tokens = self.supported_num_tokens[num_tokens_idx]
            padded_max_seq_len = self.supported_seq_lens[seq_len_idx]

            if padded_num_tokens > batch_size * padded_max_seq_len:
                seq_len_idx += 1
            elif padded_max_seq_len > padded_num_tokens:
                num_tokens_idx += 1
            else:
                return batch_size, padded_num_tokens, padded_max_seq_len

        return batch_size, 0, 0

    def get_graph_key(
            self, inputs: Dict[str, Any]) -> Tuple[EncoderKeyType, bool, bool]:
        """Compute the (bs, padded_num_tokens, padded_max_seq_len) bucket.

        `inputs['seq_lens']` must already be padded to padded_batch_size via
        `pad_batch(...)` before calling this. Dummy entries are 1-token each
        and do not raise max_seq_len since real requests dominate.
        """
        seq_lens = inputs['seq_lens']

        num_tokens = sum(
            seq_lens
        )  # Can't use len(inputs['input_ids']) because it's not padded
        batch_size = len(seq_lens)
        max_seq_len = max(seq_lens) if batch_size > 0 else 0

        if self.is_encoder_decoder:
            if self.padding_enabled and self.capture_keys:
                padded_key = self._get_dynamic_capture_key(
                    seq_lens,
                    allow_batch_padding=False,
                )
                if padded_key is None:
                    return (batch_size, 0, 0), False, False
                is_padding_performed = (padded_key[1] != num_tokens
                                        or padded_key[2] != max_seq_len)
                return padded_key, is_padding_performed, True

            max_seq_len_bucket = self._round_up(max_seq_len,
                                                self.supported_seq_lens)
            key: EncoderKeyType = (batch_size, num_tokens, max_seq_len_bucket)
            is_valid = (num_tokens <= self.max_supported_num_tokens
                        and max_seq_len_bucket > 0)
            return key, False, is_valid

        key = self._get_valid_graph_key(batch_size, num_tokens, max_seq_len)
        padded_num_tokens = key[1]
        padded_max_seq_len = key[2]

        is_padding_performed = (padded_num_tokens != num_tokens
                                or padded_max_seq_len != max_seq_len)
        is_padding_successful = (padded_num_tokens != 0
                                 and padded_max_seq_len != 0)

        return key, is_padding_performed, is_padding_successful

    @contextlib.contextmanager
    def allow_capture(self):
        """Context manager that enables CUDA graph capture.

        All encoder graphs are captured during explicit startup warmup through
        this context. Unseen runtime keys fall back to eager execution.
        """
        self._capture_allowed = True
        try:
            yield
        finally:
            self._capture_allowed = False

    @contextlib.contextmanager
    def pad_batch(self, inputs: Dict[str, Any],
                  batch_size: int) -> Iterator[Dict[str, Any]]:
        if not self.enabled or not self.padding_enabled:
            yield inputs
            return

        if self.is_encoder_decoder and self.capture_keys:
            seq_lens = inputs['seq_lens']
            padded_key = self._get_dynamic_capture_key(
                seq_lens,
                allow_batch_padding=True,
            )
            padded_batch_size = padded_key[0] if padded_key is not None else 0
        else:
            padded_batch_size = self._round_up(batch_size,
                                               self.supported_batch_sizes)
        if padded_batch_size == 0 or padded_batch_size == batch_size:
            yield inputs
            return

        padding_size = padded_batch_size - batch_size
        # Should not pad inputs if it would exceed the max supported number of tokens
        # maybe_get_cuda_graph will check this and fall back to eager if batch size is not in the supported list
        if len(inputs['input_ids']
               ) + padding_size > self.max_supported_num_tokens:
            yield inputs
            return

        # Only seq_lens is padded — that's all the attention metadata needs.
        # Token-shaped inputs (input_ids, position_ids, ...) are padded implicitly
        # by zero-filling the static buffer in `replay`.
        padded_inputs = dict(inputs)
        padded_inputs['seq_lens'] = list(
            inputs['seq_lens']) + [1] * padding_size

        yield padded_inputs

    def prepare_encoder_decoder_inputs(
        self,
        inputs: Dict[str, Any],
        key: EncoderKeyType,
        source_sequence_lengths: List[int],
    ) -> Dict[str, Any]:
        """Arrange runtime sequence metadata in capture-time slot order."""
        if not self.is_encoder_decoder:
            return inputs

        if not self.use_fixed_sequence_slots:
            prepared_inputs = dict(inputs)
            prepared_inputs[_ENCODER_SOURCE_SEQ_LENS] = list(
                source_sequence_lengths)
            return prepared_inputs

        sequence_lengths = inputs["seq_lens"]
        if (sequence_lengths[:len(source_sequence_lengths)]
                != source_sequence_lengths):
            raise ValueError("Encoder source sequence lengths must be the "
                             "unpadded prefix of graph sequence lengths.")

        source_to_slot = self._get_sequence_slot_mapping(key, sequence_lengths)
        if source_to_slot is None:
            raise ValueError(
                f"Encoder sequence lengths {sequence_lengths} are not "
                f"compatible with CUDA graph key {key}.")

        # Attention metadata follows physical slot order, while the packed
        # source tensors and the final returned output remain in request order.
        slot_sequence_lengths = [0] * len(sequence_lengths)
        for source_index, slot_index in enumerate(source_to_slot):
            slot_sequence_lengths[slot_index] = sequence_lengths[source_index]

        prepared_inputs = dict(inputs)
        prepared_inputs["seq_lens"] = slot_sequence_lengths
        prepared_inputs[_ENCODER_SOURCE_SEQ_LENS] = list(
            source_sequence_lengths)
        prepared_inputs[_ENCODER_SOURCE_TO_SLOT] = source_to_slot[:len(
            source_sequence_lengths)]
        return prepared_inputs

    def maybe_get_cuda_graph(
        self,
        inputs: Dict[str, Any],
        attn_metadata: Any,
    ) -> Tuple[Optional[Any], Optional[EncoderKeyType]]:
        """
        Decide whether the batch can use a CUDA graph.

        Returns (graph_attn_metadata, key) when a graph can be used, else
        (None, None). On graph hit, the returned `attn_metadata` is the
        graph-resident metadata whose `_seq_lens` is permanently aliased to
        a slice of the runner's pinned `seq_lens` buffer; per-replay seq_lens
        updates are pure CPU memcpys into that buffer (the H2D copy that
        feeds `_seq_lens_cuda` is captured inside the graph).
        """
        if not self.enabled:
            return None, None

        # Only TRTLLM attention backend supports encoder CUDA graphs. Other
        # backends (FlashInfer) have per-batch planner state that breaks
        # graph replay.
        if not isinstance(attn_metadata, TrtllmAttentionMetadata):
            logger.warning_once(
                "Encoder CUDA graph only supports TrtllmAttentionMetadata; "
                "falling back to eager.",
                key="encoder_cuda_graph_backend_warning")
            return None, None

        if ExpertStatistic.should_record():
            return None, None

        seq_lens = inputs['seq_lens']
        padded_batch_size = len(seq_lens)
        if padded_batch_size not in self.supported_batch_sizes:
            return None, None

        key, is_padding_performed, is_padding_successful = self.get_graph_key(
            inputs)
        if self.is_encoder_decoder and key not in self.capture_keys:
            return None, None
        padded_max_seq_len = key[2]
        if (not self.padding_enabled and is_padding_performed) \
                or not is_padding_successful:
            return None, None

        if key in self.graph_metadata:
            # Every graph key aliases the same host staging buffers. Retire a
            # prior graph's captured reads before the caller updates them.
            self.retire_staging()
            return self.graph_metadata[key]["attn_metadata"], key

        # New key not yet captured. Only create graph metadata during explicit
        # startup warmup; unseen runtime keys fall back to eager execution.
        if not self._capture_allowed:
            return None, None

        if "multi_item_part_lens" in inputs:
            # See model_engine.py for more details
            logger.warning_once(
                "Encoder CUDA graph does not support multi-item scoring; "
                "falling back to eager.",
                key="encoder_cuda_graph_multi_item_scoring_warning")
            return None, None

        if attn_metadata.has_cross_sub_metadata:
            logger.warning_once(
                "Encoder CUDA graph does not support cross-attention metadata; "
                "falling back to eager.",
                key="encoder_cuda_graph_cross_attention_warning")
            return None, None

        # First sighting of this key: create graph-resident metadata and bind
        # it to stable pinned seq_lens storage for future replays.
        graph_attn_metadata = attn_metadata.create_cuda_graph_metadata(
            padded_batch_size,
            False,
            0,
            self.cuda_graph_meta_buffers,
            encode_only=True,
        )
        assert graph_attn_metadata.is_cuda_graph

        # Lock FMHA kernel launch params to the padded max_seq_len so the
        # cubin + grid dims stay constant across replays for this key.
        graph_attn_metadata.max_context_q_len_override = padded_max_seq_len

        # Bind graph metadata to stable host seq_lens storage. The storage may
        # be pinned or pageable; only captured H2D copies require pinned memory.
        graph_attn_metadata.bind_encoder_cuda_graph_seq_lens(
            self.shared_static_tensors_cpu["seq_lens"], padded_batch_size)
        if self.use_fixed_sequence_slots:
            # CUDA graph replay keeps each request in its capture-time token
            # slot. Explicit boundaries let attention combine those fixed
            # offsets with the per-replay logical sequence lengths above.
            capture_offsets = self._get_capture_sequence_offsets(key)
            capture_offsets_cuda = torch.tensor(capture_offsets,
                                                dtype=torch.int32,
                                                device="cuda")
            graph_attn_metadata.cu_q_seqlens = capture_offsets_cuda
            graph_attn_metadata.cu_kv_seqlens = capture_offsets_cuda
        graph_attn_metadata.max_seq_len = self.config.max_seq_len
        graph_attn_metadata.request_ids = list(range(padded_batch_size))

        self.retire_staging()
        return graph_attn_metadata, key

    def _contains_nested_tensor(self, x: Any) -> bool:
        if isinstance(x, torch.Tensor):
            return x.is_nested
        if isinstance(x, dict):
            return any(self._contains_nested_tensor(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return any(self._contains_nested_tensor(v) for v in x)
        return False

    def needs_capture(self, key: EncoderKeyType) -> bool:
        return self._capture_allowed and key not in self.graphs

    def _stage_inputs(self, key: EncoderKeyType, inputs: Dict[str,
                                                              Any]) -> None:
        """Stage input and position IDs for capture or replay."""
        padded_num_tokens = key[1]

        # Captured H2D nodes read pinned host buffers. In CC mode, where H2D
        # is not captured, stage directly into the graph-resident CUDA buffers.
        static_tensors = self.shared_static_tensors_cpu if self._capture_h2d_copy else self.shared_static_tensors

        if self.is_encoder_decoder and _ENCODER_SOURCE_TO_SLOT in inputs:
            self._stage_encoder_decoder_inputs(key, inputs, static_tensors)
            return

        input_ids = inputs["input_ids"]
        if isinstance(input_ids, list):
            actual_tokens = len(input_ids)
            static_tensors["input_ids"][:actual_tokens].copy_(
                torch.tensor(input_ids, dtype=torch.int32))
        elif isinstance(input_ids, torch.Tensor):
            actual_tokens = int(input_ids.shape[0])
            static_tensors["input_ids"][:actual_tokens].copy_(input_ids)
        else:
            raise TypeError(f"Unsupported input_ids type: {type(input_ids)}")
        static_tensors["input_ids"][actual_tokens:padded_num_tokens].fill_(0)

        # Auto-generate packed position IDs without allocating one concatenated
        # tensor, or copy caller-provided values into the stable staging buffer.
        staged_position_ids = static_tensors["position_ids"][0]
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            offset = 0
            for seq_len in inputs["seq_lens"]:
                staged_position_ids[offset:offset + seq_len].copy_(
                    self._arange_max[:seq_len])
                offset += seq_len
        else:
            if isinstance(position_ids, list):
                staged_position_ids[:actual_tokens].copy_(
                    torch.tensor(position_ids, dtype=torch.int32))
            elif isinstance(position_ids, torch.Tensor):
                staged_position_ids[:actual_tokens].copy_(
                    position_ids.flatten())
            else:
                raise TypeError(
                    f"Unsupported position_ids type: {type(position_ids)}")
            offset = actual_tokens

        staged_position_ids[offset:padded_num_tokens].fill_(0)

    def _stage_encoder_decoder_inputs(
        self,
        key: EncoderKeyType,
        inputs: Dict[str, Any],
        static_tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Scatter packed request inputs into fixed capture-time slots."""
        source_sequence_lengths = inputs[_ENCODER_SOURCE_SEQ_LENS]
        source_to_slot = inputs[_ENCODER_SOURCE_TO_SLOT]

        input_ids = inputs["input_ids"]
        if isinstance(input_ids, list):
            source_input_ids = torch.tensor(input_ids, dtype=torch.int32)
        elif isinstance(input_ids, torch.Tensor):
            source_input_ids = input_ids
        else:
            raise TypeError(f"Unsupported input_ids type: {type(input_ids)}")

        actual_num_tokens = sum(source_sequence_lengths)
        if int(source_input_ids.shape[0]) != actual_num_tokens:
            raise ValueError(
                "Packed encoder input IDs must match source sequence lengths.")

        position_ids = inputs.get("position_ids")
        if isinstance(position_ids, list):
            source_position_ids = torch.tensor(position_ids, dtype=torch.int32)
        elif isinstance(position_ids, torch.Tensor):
            source_position_ids = position_ids.flatten()
        elif position_ids is None:
            source_position_ids = None
        else:
            raise TypeError(
                f"Unsupported position_ids type: {type(position_ids)}")
        if (source_position_ids is not None
                and int(source_position_ids.shape[0]) != actual_num_tokens):
            raise ValueError("Packed encoder position IDs must match source "
                             "sequence lengths.")

        static_tensors["input_ids"][:key[1]].zero_()
        static_tensors["position_ids"][:, :key[1]].zero_()

        capture_offsets = self._get_capture_sequence_offsets(key)

        source_offset = 0
        for source_index, sequence_length in enumerate(source_sequence_lengths):
            slot_index = source_to_slot[source_index]
            destination_offset = capture_offsets[slot_index]
            source_slice = slice(source_offset, source_offset + sequence_length)
            destination_slice = slice(destination_offset,
                                      destination_offset + sequence_length)
            static_tensors["input_ids"][destination_slice].copy_(
                source_input_ids[source_slice])
            if source_position_ids is None:
                static_tensors["position_ids"][0, destination_slice].copy_(
                    self._arange_max[:sequence_length])
            else:
                static_tensors["position_ids"][0, destination_slice].copy_(
                    source_position_ids[source_slice])
            source_offset += sequence_length

    def restore_encoder_decoder_output(
        self,
        key: EncoderKeyType,
        output: torch.Tensor,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """Compact fixed-slot graph output back into request order."""
        source_sequence_lengths = inputs[_ENCODER_SOURCE_SEQ_LENS]
        if _ENCODER_SOURCE_TO_SLOT not in inputs:
            return output[:sum(source_sequence_lengths)].clone()

        source_to_slot = inputs[_ENCODER_SOURCE_TO_SLOT]

        capture_offsets = self._get_capture_sequence_offsets(key)

        output_slices = []
        for source_index, sequence_length in enumerate(source_sequence_lengths):
            source_offset = capture_offsets[source_to_slot[source_index]]
            output_slices.append(output[source_offset:source_offset +
                                        sequence_length])
        return torch.cat(output_slices, dim=0)

    def capture(
        self,
        key: EncoderKeyType,
        forward_fn: Callable[[Dict[str, Any]], Any],
        inputs: Dict[str, Any],
    ) -> Any:
        """Warm up and/or capture the forward pass for a graph key."""
        padded_num_tokens = key[1]

        sliced_static_tensors = {
            "input_ids":
            self.shared_static_tensors["input_ids"][:padded_num_tokens],
            "position_ids":
            self.shared_static_tensors["position_ids"][:, :padded_num_tokens],
        }
        sliced_static_tensors_cpu = {
            "input_ids":
            self.shared_static_tensors_cpu["input_ids"][:padded_num_tokens],
            "position_ids":
            self.shared_static_tensors_cpu["position_ids"]
            [:, :padded_num_tokens],
        }

        capture_inputs = dict(inputs)
        capture_inputs.update(sliced_static_tensors)

        attn_md = capture_inputs["attn_metadata"]

        self.graph_metadata[key] = {"attn_metadata": attn_md}

        # Warmup must see the same runtime data as capture. In particular,
        # graph metadata initializes _seq_lens_cuda to ones, while
        # prepare_encoder_cuda_graph_replay updates its stable host buffer.
        # Populate every device input before warmup so packed-token counts and
        # sequence boundaries are consistent.
        self._stage_inputs(key, inputs)
        if self._capture_h2d_copy:
            capture_inputs["input_ids"].copy_(
                sliced_static_tensors_cpu["input_ids"], non_blocking=True)
            capture_inputs["position_ids"].copy_(
                sliced_static_tensors_cpu["position_ids"], non_blocking=True)
        attn_md._seq_lens_cuda.copy_(attn_md._seq_lens, non_blocking=True)
        torch.cuda.current_stream().synchronize()

        output = None
        with with_multi_stream(True), piecewise_cuda_graph(False):
            # Warmup runs required by CUDA graph semantics. See
            # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
            # Warmups initialize PyTorch and attention metadata state, and
            # resize the shared attention workspace before any graph is captured.
            for _ in range(self.WARMUP_STEPS):
                output = forward_fn(capture_inputs)

            if self.is_warmup_only:
                return output

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,
                                  pool=self.memory_pool,
                                  capture_error_mode="thread_local"):
                if self._capture_h2d_copy:
                    # H2D copies for captured inside the graph: at replay
                    # time it re-issues from the pinned static buffer without
                    # an eager driver call.
                    capture_inputs["input_ids"].copy_(
                        sliced_static_tensors_cpu["input_ids"],
                        non_blocking=True)
                    capture_inputs["position_ids"].copy_(
                        sliced_static_tensors_cpu["position_ids"],
                        non_blocking=True)
                    attn_md._seq_lens_cuda.copy_(attn_md._seq_lens,
                                                 non_blocking=True)
                output = forward_fn(capture_inputs)

        if self._contains_nested_tensor(output):
            raise TypeError(
                "Encoder CUDA graph does not support nested tensor outputs. "
                "Disable encoder CUDA graphs for models with ragged outputs.")
        self.graphs[key] = graph
        graph_output = make_weak_ref(output)
        self.graph_outputs[key] = graph_output
        self.memory_pool = graph.pool()
        return graph_output

    def retire_staging(self) -> None:
        """Wait until a prior replay no longer reads shared staging buffers."""
        if self._staging_retirement_event is not None:
            self._staging_retirement_event.synchronize()
            self._staging_retirement_event = None

    def replay(
        self,
        key: EncoderKeyType,
        inputs: Dict[str, Any],
    ) -> Any:
        """Replay a captured graph with current inputs."""
        self.retire_staging()

        stored_meta = self.graph_metadata[key]
        assert inputs["attn_metadata"] is stored_meta["attn_metadata"]

        self._stage_inputs(key, inputs)

        if not self._capture_h2d_copy:
            stored_meta["attn_metadata"]._seq_lens_cuda.copy_(
                stored_meta["attn_metadata"]._seq_lens, non_blocking=True)

        self.graphs[key].replay()
        self._staging_retirement_event = torch.cuda.Event()
        self._staging_retirement_event.record(torch.cuda.current_stream())

        return self.graph_outputs[key]

    def get_graph_pool(self):
        return self.memory_pool

    def clear(self):
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()
