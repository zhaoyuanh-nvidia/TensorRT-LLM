# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sampler for one-model speculative decoding.

Every one-model speculative mode (MTP, Eagle3, SA, DraftTarget, PARD, DFlash,
DSpark) shares a single sampler: the worker's fused kernel already performs
drafting, target verification and acceptance, so the sampler only scatters that
output into slot-indexed buffers, starts the async D2H copy, and updates
requests host-side. Buffer shapes derive entirely from ``TorchSampler.Args``.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.sampler import (
    DEFAULT_BEAM_IDX,
    AsyncWorkerMixin,
    Sampler,
    SampleState,
    SampleStateTensors,
    TorchSampler,
    add_token,
    int_tensor,
)
from ..pyexecutor.scheduler import ScheduledRequests
from .dspark_confidence import (
    DSparkConfidenceDeviceLayout,
    validate_confidence_device_layout_row_map,
)


def _clone_dspark_confidence_layout(
        layout: DSparkConfidenceDeviceLayout
) -> DSparkConfidenceDeviceLayout:
    """Detach graph-owned layout tensors from the next replay's lifetime."""
    return DSparkConfidenceDeviceLayout(*(tensor.clone() for tensor in layout))


@dataclass(kw_only=True)
class SampleStateTensorsSpec(SampleStateTensors):
    """Tensors for speculative decoding sample state."""

    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor
    next_draft_lens: Optional[torch.Tensor] = None
    verified_draft_lens: Optional[torch.Tensor] = None
    dspark_confidence_layout: Optional[DSparkConfidenceDeviceLayout] = None
    dspark_confidence_execution_batch_size: Optional[int] = None
    dspark_confidence_verifier_token_budget: Optional[int] = None
    dspark_confidence_verifier_token_budget_host: Optional[torch.Tensor] = None
    dspark_confidence_semantics_host: Optional[torch.Tensor] = None
    dspark_confidence_query_lens_host: Optional[torch.Tensor] = None
    dspark_confidence_budget_ready_event: Optional[torch.cuda.Event] = None
    dspark_confidence_route_epoch: Optional[int] = None
    dspark_confidence_physical_draft_len: Optional[int] = None
    dspark_confidence_engine_generation: Optional[int] = None
    dspark_confidence_request_ids: Optional[tuple[int, ...]] = None
    dspark_confidence_seq_slots: Optional[tuple[int, ...]] = None
    dspark_confidence_native_uniform: bool = False
    dspark_confidence_native_uniform_draft_len_host: Optional[
        torch.Tensor] = None
    dspark_confidence_native_uniform_ready_event: Optional[
        torch.cuda.Event] = None


@dataclass(kw_only=True)
class SampleStateSpec(SampleState):
    """Sample state for speculative decoding."""

    device: SampleStateTensorsSpec
    host: SampleStateTensorsSpec
    # Per-request draft-token counts of the step this state samples, captured
    # in sample_async before dummy draft tokens are added (index-aligned with
    # `requests`; 0 for finished-context requests). update_requests pairs
    # them with py_num_accepted_draft_tokens: reading py_draft_tokens there
    # instead would see the NEXT step's buffer, which update_requests itself
    # installs.
    draft_lens: Optional[list[int]] = None
    dspark_confidence_execution_batch_size: Optional[int] = None
    dspark_confidence_route_epoch: Optional[int] = None
    dspark_confidence_verifier_token_budget: Optional[int] = None
    dspark_confidence_physical_draft_len: Optional[int] = None
    dspark_confidence_engine_generation: Optional[int] = None
    dspark_confidence_native_uniform: bool = False


class SpecSampler(Sampler[SampleStateSpec], AsyncWorkerMixin):
    """
    Sampler for all one-model speculative decoding modes.

    Provides:
    - Pre-allocated, slot-indexed GPU storage buffers
    - Async GPU->CPU copy in sample_async
    - Request state updates in update_requests

    This class carries no per-mode behavior. ``args.max_total_draft_tokens`` is
    ``spec_config.tokens_per_gen_step - 1`` (see ``create_torch_sampler_args``),
    i.e. the target's per-step input width minus one, which is exactly the
    draft length every mode used to compute for itself.
    """

    SampleState = SampleStateSpec

    def is_generation_model(self) -> bool:
        return True

    def validate_request(self, request: LlmRequest) -> None:
        """Reject sampling parameters the one-model speculative path cannot honor.

        The one-model sampling kernels take only temperature/top_k/top_p (see
        SpecMetadata.populate_sampling_params_for_one_model); min_p has no
        buffer there, so it would be silently dropped and the request would
        decode from a different distribution than the user asked for. Threading
        it through costs measurable throughput on the rejection path, so reject
        instead. Raised from validate_request (request admission), so only the
        offending request fails rather than the whole executor step.
        """
        sampling_config = request.sampling_config
        if sampling_config is None:
            return
        # min_p lives on the C++ SamplingConfig as an optional singleton list.
        min_p = sampling_config.min_p
        if min_p and min_p[0] > 0.0:
            raise ValueError(
                "min_p is not supported with one-model speculative decoding. "
                "Drop min_p from the request, or disable speculative decoding."
            )

    @dataclass(kw_only=True)
    class Store:
        """Storage for speculative decoding tensors."""

        new_tokens: torch.Tensor
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor
        next_draft_lens: torch.Tensor
        verified_draft_lens: torch.Tensor

    def __init__(self, args: TorchSampler.Args, *, accepted_path_len: Optional[int] = None):
        """
        Initialize the speculative sampler.

        Args:
            args: TorchSampler.Args with max_num_sequences, max_seq_len, etc.
            accepted_path_len: Upper bound on the number of tokens a single step
                can accept, used to size new_tokens. Defaults to
                ``args.max_draft_len + 1``; see the store comment below for the
                one mode that has to override it.
        """
        self._async_worker_init(args.enable_async_worker)
        self.mapping = None
        self._trace_dspark_budget = os.environ.get(
            "TLLM_DSPARK_BUDGET_TRACE", ""
        ).strip().lower() not in ("", "0", "false", "no", "off")
        self.max_seq_len = args.max_seq_len
        # Wire width minus one: the number of draft slots the target verifies
        # per step. Linear modes set max_total_draft_tokens == max_draft_len;
        # tree modes set it to the total node count; PARD sets it to 2K-1
        # because it also feeds mask tokens through the target.
        self.draft_len = args.max_total_draft_tokens

        seq_slots = args.max_num_sequences
        self.max_beam_width = args.max_beam_width
        assert self.max_beam_width == 1, "beam width must be 1 for speculative decoding"

        # new_tokens holds the accepted tokens only, so it is sized to how many
        # a step can accept rather than to the wire width. Normally that is
        # max_draft_len + 1: the drafter advances max_draft_len times, and the
        # golden token the target always accepts adds one. Verified against
        # Eagle3 dynamic tree (K=6, T=60), MTP dynamic tree, PARD (T=2K-1) and
        # the linear modes -- none exceed it.
        #
        # The exception is the deprecated eagle_choices static tree. There the
        # one-model drafter ignores the tree and runs _forward_draft_loop, a
        # linear loop over runtime_draft_len == max_total_draft_tokens, so a
        # step can accept up to max_total_draft_tokens + 1 tokens even though
        # max_draft_len only describes the depth of a tree that is never built.
        # (Tree-aware acceptance lives in TorchSampler, i.e. the two-model
        # path.) get_spec_decoder passes the wire width for that mode; both it
        # and this workaround go away with the feature in release 1.4.
        self.max_accepted_path_len = (
            accepted_path_len if accepted_path_len is not None else args.max_draft_len + 1
        )
        self.store = self.Store(
            new_tokens=int_tensor((self.max_accepted_path_len, seq_slots, self.max_beam_width)),
            next_new_tokens=int_tensor(
                (args.max_total_draft_tokens + 1, seq_slots, self.max_beam_width)
            ),
            next_draft_tokens=int_tensor((seq_slots, args.max_total_draft_tokens)),
            new_tokens_lens=int_tensor((seq_slots,)),
            next_draft_lens=int_tensor((seq_slots,)),
            verified_draft_lens=int_tensor((seq_slots,)),
        )

    def _trace_dspark_budget_transition(
        self,
        state: SampleStateSpec,
        next_draft_lens: list[int] | None,
    ) -> None:
        """Log the effective local V transition without another device sync.

        ``update_requests`` has already synchronized the sampler's existing
        asynchronous D2H copy before this helper runs. The opt-in trace only
        sums those host lists, so it does not add a transfer or perturb the
        CUDA-graph path when disabled. Attention-DP diagnostics can compare
        one line per rank to prove that every rank selected the same graph
        tier while retaining different request-local prefixes.
        """
        if not self._trace_dspark_budget or next_draft_lens is None:
            return

        active_indices = [
            index
            for index, request in enumerate(state.requests)
            if request.state != LlmRequestState.GENERATION_COMPLETE
        ]
        current_lens = state.draft_lens or [0] * len(state.requests)
        current_budget = len(active_indices) + sum(
            int(current_lens[index]) for index in active_indices
        )
        next_budget = len(active_indices) + sum(
            int(next_draft_lens[state.requests[index].py_seq_slot]) for index in active_indices
        )
        rank = getattr(self.mapping, "rank", "unknown")
        logger.info(
            "DSPARK_BUDGET_TRACE rank=%s requests=%d current_v=%d next_v=%d",
            rank,
            len(active_indices),
            current_budget,
            next_budget,
        )

    def _request_common_handling(
        self,
        request: LlmRequest,
        next_draft_tokens: list[list[int]],
        next_draft_len: int,
        physical_draft_len: int,
    ) -> None:
        """Common handling for both context and generation requests."""
        if request.py_return_context_logits:
            logger.warning(
                "return_context_logits not supported with speculative decoding, "
                "skipping for request %s",
                request.py_request_id,
            )
        if request.py_return_generation_logits:
            logger.warning(
                "return_generation_logits not supported with speculative decoding, "
                "skipping for request %s",
                request.py_request_id,
            )
        if request.py_return_log_probs:
            logger.warning(
                "return_log_probs not supported with speculative decoding, skipping for request %s",
                request.py_request_id,
            )
        # Keep the host-visible proposal buffer at its physical width. CUDA
        # graph selection keys on that stable K, while the separate effective
        # length describes the confidence-retained prefix for packing into V.
        # Legacy dynamic-width modes pass the same value for both lengths.
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot][:physical_draft_len]
        request.py_draft_tokens_effective_len = next_draft_len
        request.py_decoding_iter += 1

    def update_requests(
        self,
        state: SampleStateSpec,
        resource_manager: Optional[BaseResourceManager] = None,
    ) -> None:
        """
        CPU-side request updates after GPU->CPU sync.

        Waits for async copy to complete, then updates request state with:
        - Accepted tokens
        - Stop criteria checks
        - Next iteration draft tokens
        """
        assert isinstance(state, SampleStateSpec)

        state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens.tolist()
        new_tokens_lens_list = state.host.new_tokens_lens.tolist()
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        next_draft_lens_list = (
            state.host.next_draft_lens.tolist() if state.host.next_draft_lens is not None else None
        )
        planned_requests = [
            request for request in state.requests
            if request.state != LlmRequestState.GENERATION_COMPLETE
        ]
        verified_draft_lens_list = (
            state.host.verified_draft_lens.tolist()
            if state.host.verified_draft_lens is not None
            else None
        )
        self._trace_dspark_budget_transition(state, next_draft_lens_list)
        beam_idx = DEFAULT_BEAM_IDX
        runtime_draft_len = getattr(state, "runtime_draft_len", self.draft_len)
        native_uniform = bool(
            getattr(state, "dspark_confidence_native_uniform", False))
        native_uniform_draft_len = runtime_draft_len
        native_uniform_route_valid = False
        if native_uniform:
            native_uniform_draft_len_host = getattr(
                state.host,
                "dspark_confidence_native_uniform_draft_len_host",
                None,
            )
            if native_uniform_draft_len_host is not None:
                candidate = int(native_uniform_draft_len_host)
                native_uniform_route_valid = candidate in (
                    runtime_draft_len - 1,
                    runtime_draft_len,
                )
                if native_uniform_route_valid:
                    native_uniform_draft_len = candidate

        for req_idx, req in enumerate(state.requests):
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            num_new_tokens = new_tokens_lens_list[req.py_seq_slot]
            # new_tokens is sized to this bound, and add_token indexes a plain
            # host-side list, so a violation would otherwise surface as an
            # opaque IndexError.
            assert num_new_tokens <= self.max_accepted_path_len, (
                f"accepted {num_new_tokens} tokens in one step, but new_tokens is "
                f"sized for {self.max_accepted_path_len}"
            )
            for i in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=i)
                if TorchSampler._handle_stop_criteria(
                    req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                ):
                    break
            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            # Pair the acceptance count with the draft count of the SAME step,
            # emitted by the model beside new_tokens_lens in fixed-budget mode.
            # This remains iteration-aligned under the overlap scheduler; the
            # mutable request may already describe a different iteration when
            # sample_async runs. Other modes retain the request snapshot.
            completed_draft_len = (
                verified_draft_lens_list[req.py_seq_slot]
                if verified_draft_lens_list is not None
                else (state.draft_lens[req_idx] if state.draft_lens is not None else 0)
            )
            if req.py_num_accepted_draft_tokens > completed_draft_len:
                raise RuntimeError(
                    f"accepted {req.py_num_accepted_draft_tokens} draft tokens after "
                    f"verifying only {completed_draft_len} for request {req.py_request_id}"
                )
            req.py_num_draft_tokens_verified = completed_draft_len
            req.py_rewind_len = completed_draft_len - req.py_num_accepted_draft_tokens
            next_draft_len = (
                next_draft_lens_list[req.py_seq_slot]
                if next_draft_lens_list is not None
                else native_uniform_draft_len
            )
            self._request_common_handling(
                req,
                next_draft_tokens_list,
                next_draft_len,
                runtime_draft_len,
            )

        # Publish the route beside the host-visible retained lengths.  A
        # completion invalidates the exact V allocation, so surviving rows are
        # restored to the already-generated physical K block before the next
        # target input is packed.  ADP peers reconcile this ordinary fallback
        # at the next pre-pack collective.
        execution_g = state.dspark_confidence_execution_batch_size
        route_epoch = state.dspark_confidence_route_epoch
        verifier_budget = state.dspark_confidence_verifier_token_budget
        if verifier_budget is None:
            verifier_budget_host = getattr(
                state.host,
                "dspark_confidence_verifier_token_budget_host",
                None,
            )
            if verifier_budget_host is not None:
                verifier_budget = int(verifier_budget_host)
        semantic_route_valid = True
        semantics_host = getattr(
            state.host, "dspark_confidence_semantics_host", None)
        if semantics_host is not None:
            semantics = tuple(int(value) for value in semantics_host.tolist())
            if len(semantics) != 7:
                semantic_route_valid = False
            else:
                (valid, row_map_valid, retained_count, query_count,
                 cu_query_count, real_count, declared_budget) = semantics
                semantic_route_valid = bool(
                    valid == 1 and row_map_valid == 1
                    and execution_g is not None
                    and verifier_budget is not None
                    and int(verifier_budget) == declared_budget
                    and retained_count == declared_budget - int(execution_g)
                    and query_count == declared_budget
                    and cu_query_count == declared_budget
                    and real_count == len(planned_requests))
        surviving_requests = [
            request for request in planned_requests
            if request.state != LlmRequestState.GENERATION_COMPLETE
        ]
        route_complete = (
            next_draft_lens_list is not None and execution_g is not None
            and execution_g > 0 and route_epoch is not None
            and route_epoch > 0 and len(surviving_requests) == len(planned_requests)
            and semantic_route_valid
        )
        native_route_complete = (
            native_uniform and native_uniform_route_valid
            and execution_g is not None and execution_g > 0
            and route_epoch is not None and route_epoch > 0
            and len(surviving_requests) == len(planned_requests)
        )
        if route_complete or native_route_complete:
            computed_verifier_budget = (
                int(execution_g) * (1 + int(native_uniform_draft_len))
                if native_route_complete else int(execution_g) + sum(
                    int(next_draft_lens_list[request.py_seq_slot])
                    for request in surviving_requests
                ))
            verifier_budget = (verifier_budget if verifier_budget is not None
                               else computed_verifier_budget)
            if (not native_route_complete
                    and int(verifier_budget) != computed_verifier_budget):
                raise RuntimeError(
                    "DSpark carried verifier budget does not match retained "
                    "draft lengths")
            if native_route_complete:
                verifier_budget = computed_verifier_budget
            for request in surviving_requests:
                previous_epoch = getattr(
                    request, "py_dspark_confidence_route_epoch", None)
                if previous_epoch is not None and int(
                        previous_epoch) > int(route_epoch):
                    raise RuntimeError(
                        "DSpark confidence route epoch regressed during delayed "
                        "sampler publication")
                previous_route = (
                    int(getattr(
                        request,
                        "py_dspark_confidence_execution_batch_size", 0) or 0),
                    int(getattr(
                        request,
                        "py_dspark_confidence_verifier_token_budget", 0) or 0),
                )
                if (previous_epoch is not None
                        and int(previous_epoch) == int(route_epoch)
                        and previous_route !=
                        (int(execution_g), int(verifier_budget))):
                    raise RuntimeError(
                        "DSpark confidence route epoch was republished with "
                        "different G/V provenance")
                request.py_dspark_confidence_route_epoch = route_epoch
                request.py_dspark_confidence_execution_batch_size = execution_g
                request.py_dspark_confidence_verifier_token_budget = verifier_budget
        elif next_draft_lens_list is not None or native_uniform:
            for request in surviving_requests:
                request.py_draft_tokens_effective_len = len(
                    request.py_draft_tokens)
                request.py_dspark_confidence_route_epoch = None
                request.py_dspark_confidence_execution_batch_size = None
                request.py_dspark_confidence_verifier_token_budget = None

    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        outputs: dict[str, torch.Tensor],
        num_context_logits_prefix_sum: list[int],
    ) -> SampleStateSpec:
        """
        Async sampling - schedules GPU->CPU copy.
        Called after CUDA graph replay.

        Args:
            scheduled_requests: Batch of scheduled requests
            outputs: Dict from worker forward() containing:
                - new_tokens: [batch, max_draft_len + 1] accepted tokens
                - new_tokens_lens: [batch] number of accepted tokens
                - next_draft_tokens: [batch, max_draft_len] draft tokens for next iter
                - next_new_tokens: [batch, max_draft_len + 1] input for next iter
            num_context_logits_prefix_sum: Prefix sum of context logits (unused)

        Returns:
            SampleStateSpec with device and host tensors
        """
        num_skip = len(scheduled_requests.context_requests_chunking)
        finished_context_requests = scheduled_requests.context_requests_last_chunk
        sampling_requests = finished_context_requests + scheduled_requests.generation_requests
        num_sampling_requests = len(sampling_requests)

        # Snapshot each request's draft count for THIS step before
        # _add_dummy_draft_tokens below installs placeholder drafts on
        # finished-context requests; update_requests pairs these with the
        # acceptance counts (see SampleStateSpec.draft_lens). Drafter-fed
        # flows (NGram, SA) pad py_draft_tokens to the static max for CUDA
        # graphs before the forward, so prefer the pre-padding count the
        # drafter recorded; min() guards against a stale count when the
        # buffer was since cleared (e.g. speculation dynamically disabled).
        draft_lens = [
            min(r.py_draft_tokens_effective_len, get_draft_token_length(r))
            if r.py_draft_tokens_effective_len is not None
            else get_draft_token_length(r)
            for r in sampling_requests
        ]

        slots = torch.as_tensor([r.py_seq_slot for r in sampling_requests], dtype=torch.long)
        slots = slots.to(device="cuda", non_blocking=True)

        o_new_tokens = outputs["new_tokens"][num_skip : num_skip + num_sampling_requests]
        o_new_tokens_lens = outputs["new_tokens_lens"][num_skip : num_skip + num_sampling_requests]
        o_new_tokens_lens = o_new_tokens_lens.to(dtype=self.store.new_tokens_lens.dtype)
        o_next_draft_tokens = outputs["next_draft_tokens"][
            num_skip : num_skip + num_sampling_requests
        ]
        o_next_new_tokens = outputs["next_new_tokens"][num_skip : num_skip + num_sampling_requests]
        runtime_draft_len = o_next_draft_tokens.shape[1]
        o_next_draft_lens = outputs.get("next_draft_lens")
        o_confidence_layout = outputs.get("next_dspark_confidence_layout")
        o_verified_draft_lens = outputs.get("verified_draft_lens")
        if o_next_draft_lens is not None:
            o_next_draft_lens = o_next_draft_lens[num_skip : num_skip + num_sampling_requests]
            o_next_draft_lens = o_next_draft_lens.to(dtype=self.store.next_draft_lens.dtype)
        if o_verified_draft_lens is not None:
            o_verified_draft_lens = o_verified_draft_lens[
                num_skip : num_skip + num_sampling_requests
            ].to(dtype=self.store.verified_draft_lens.dtype)

        confidence_layout = None
        confidence_budget_host = None
        confidence_semantics_host = None
        confidence_query_lens_host = None
        confidence_budget_event = None
        native_uniform = bool(
            outputs.get("dspark_confidence_native_uniform", False))
        native_uniform_draft_len_host = None
        native_uniform_ready_event = None
        if native_uniform:
            native_uniform_draft_len = outputs.get(
                "dspark_confidence_native_uniform_draft_len")
            if (not isinstance(native_uniform_draft_len, torch.Tensor)
                    or native_uniform_draft_len.shape != ()):
                raise RuntimeError(
                    "DSpark native-uniform route requires one device scalar "
                    "draft length")
            native_uniform_draft_len_host = torch.empty(
                (), dtype=torch.int32, device="cpu", pin_memory=True)
            native_uniform_draft_len_host.copy_(
                native_uniform_draft_len, non_blocking=True)
            native_uniform_ready_event = torch.cuda.Event()
            native_uniform_ready_event.record()
        if o_confidence_layout is not None:
            # CUDA-graph output dictionaries and their tensors are persistent.
            # Clone every layout tensor into sampler-owned lifetime before the
            # next replay can overwrite it, and replace row indices with the
            # exact slot-indexed source used by the overlap buffers.
            confidence_layout = _clone_dspark_confidence_layout(
                o_confidence_layout)
            confidence_layout.source_batch_indices.zero_()
            confidence_layout.source_batch_indices[:num_sampling_requests].copy_(
                slots, non_blocking=True)
            confidence_layout.row_map_valid.copy_(
                validate_confidence_device_layout_row_map(
                    confidence_layout,
                    runtime_draft_len,
                    expected_source_slots=slots,
                ).to(torch.int32))
            confidence_semantics_host = torch.empty(
                7, dtype=torch.int32, device="cpu", pin_memory=True)
            confidence_semantics_host.copy_(torch.stack((
                confidence_layout.semantic_valid,
                confidence_layout.row_map_valid,
                confidence_layout.retained_token_count,
                confidence_layout.query_token_count,
                confidence_layout.cu_query_token_count,
                confidence_layout.real_request_count,
                confidence_layout.declared_verifier_token_budget,
            )), non_blocking=True)
            confidence_query_lens_host = torch.empty(
                confidence_layout.query_lens.shape[0],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True)
            confidence_query_lens_host.copy_(
                confidence_layout.query_lens, non_blocking=True)
            if outputs.get("dspark_confidence_verifier_token_budget") is None:
                confidence_budget_host = torch.empty(
                    (), dtype=torch.int32, device="cpu", pin_memory=True)
                confidence_budget_host.copy_(
                    confidence_layout.verifier_token_budget,
                    non_blocking=True)
            confidence_budget_event = torch.cuda.Event()
            confidence_budget_event.record()

        # Pad or truncate to match fixed-size store buffers for index_copy_.
        # The worker output width tracks runtime_draft_len, which dynamic draft
        # length shrinks below the statically allocated store width.
        new_tokens_width = self.store.new_tokens.shape[0]
        next_new_tokens_width = self.store.next_new_tokens.shape[0]
        draft_tokens_width = self.store.next_draft_tokens.shape[1]
        if o_new_tokens.shape[1] < new_tokens_width:
            o_new_tokens = torch.nn.functional.pad(
                o_new_tokens, (0, new_tokens_width - o_new_tokens.shape[1])
            )
        elif o_new_tokens.shape[1] > new_tokens_width:
            o_new_tokens = o_new_tokens[:, :new_tokens_width]
        if o_next_draft_tokens.shape[1] < draft_tokens_width:
            o_next_draft_tokens = torch.nn.functional.pad(
                o_next_draft_tokens, (0, draft_tokens_width - o_next_draft_tokens.shape[1])
            )
        elif o_next_draft_tokens.shape[1] > draft_tokens_width:
            o_next_draft_tokens = o_next_draft_tokens[:, :draft_tokens_width]
        if o_next_new_tokens.shape[1] < next_new_tokens_width:
            o_next_new_tokens = torch.nn.functional.pad(
                o_next_new_tokens, (0, next_new_tokens_width - o_next_new_tokens.shape[1])
            )
        elif o_next_new_tokens.shape[1] > next_new_tokens_width:
            o_next_new_tokens = o_next_new_tokens[:, :next_new_tokens_width]

        # Use index_copy_ for efficient copying (slots are unique)
        self.store.new_tokens.squeeze(-1).T.index_copy_(0, slots, o_new_tokens)
        self.store.next_new_tokens.squeeze(-1).T.index_copy_(0, slots, o_next_new_tokens)
        self.store.new_tokens_lens.index_copy_(0, slots, o_new_tokens_lens)
        self.store.next_draft_tokens.index_copy_(0, slots, o_next_draft_tokens)
        if o_next_draft_lens is not None:
            self.store.next_draft_lens.index_copy_(0, slots, o_next_draft_lens)
        if o_verified_draft_lens is not None:
            self.store.verified_draft_lens.index_copy_(0, slots, o_verified_draft_lens)

        # Create sample state with async D2H copy
        device_tensors = SampleStateTensorsSpec(
            new_tokens=self.store.next_new_tokens,
            new_tokens_lens=self.store.new_tokens_lens,
            next_draft_tokens=self.store.next_draft_tokens,
            next_draft_lens=(self.store.next_draft_lens if o_next_draft_lens is not None else None),
            verified_draft_lens=(
                self.store.verified_draft_lens if o_verified_draft_lens is not None else None
            ),
            dspark_confidence_layout=confidence_layout,
            dspark_confidence_execution_batch_size=outputs.get(
                "dspark_confidence_execution_batch_size"),
            dspark_confidence_verifier_token_budget=outputs.get(
                "dspark_confidence_verifier_token_budget"),
            dspark_confidence_verifier_token_budget_host=confidence_budget_host,
            dspark_confidence_semantics_host=confidence_semantics_host,
            dspark_confidence_query_lens_host=confidence_query_lens_host,
            dspark_confidence_budget_ready_event=confidence_budget_event,
            dspark_confidence_route_epoch=outputs.get(
                "dspark_confidence_route_epoch"),
            dspark_confidence_physical_draft_len=runtime_draft_len,
            dspark_confidence_engine_generation=outputs.get(
                "dspark_confidence_engine_generation"),
            dspark_confidence_request_ids=tuple(
                request.py_request_id for request in sampling_requests),
            dspark_confidence_seq_slots=tuple(
                request.py_seq_slot for request in sampling_requests),
            dspark_confidence_native_uniform=native_uniform,
            dspark_confidence_native_uniform_draft_len_host=(
                native_uniform_draft_len_host),
            dspark_confidence_native_uniform_ready_event=(
                native_uniform_ready_event),
        )

        host_tensors = SampleStateTensorsSpec(
            new_tokens=self._copy_to_host(self.store.new_tokens),
            new_tokens_lens=self._copy_to_host(self.store.new_tokens_lens),
            next_draft_tokens=self._copy_to_host(self.store.next_draft_tokens),
            next_draft_lens=(
                self._copy_to_host(self.store.next_draft_lens)
                if o_next_draft_lens is not None
                else None
            ),
            verified_draft_lens=(
                self._copy_to_host(self.store.verified_draft_lens)
                if o_verified_draft_lens is not None
                else None
            ),
            dspark_confidence_layout=confidence_layout,
            dspark_confidence_execution_batch_size=outputs.get(
                "dspark_confidence_execution_batch_size"),
            dspark_confidence_verifier_token_budget=outputs.get(
                "dspark_confidence_verifier_token_budget"),
            dspark_confidence_verifier_token_budget_host=confidence_budget_host,
            dspark_confidence_semantics_host=confidence_semantics_host,
            dspark_confidence_query_lens_host=confidence_query_lens_host,
            dspark_confidence_budget_ready_event=confidence_budget_event,
            dspark_confidence_route_epoch=outputs.get(
                "dspark_confidence_route_epoch"),
            dspark_confidence_physical_draft_len=runtime_draft_len,
            dspark_confidence_engine_generation=outputs.get(
                "dspark_confidence_engine_generation"),
            dspark_confidence_request_ids=tuple(
                request.py_request_id for request in sampling_requests),
            dspark_confidence_seq_slots=tuple(
                request.py_seq_slot for request in sampling_requests),
            dspark_confidence_native_uniform=native_uniform,
            dspark_confidence_native_uniform_draft_len_host=(
                native_uniform_draft_len_host),
            dspark_confidence_native_uniform_ready_event=(
                native_uniform_ready_event),
        )
        sampler_event = self._record_sampler_event()

        # Add dummy draft tokens to context requests for KV cache preparation
        for request in finished_context_requests:
            request.py_draft_tokens = [1] * self.draft_len

        return SampleStateSpec(
            requests=sampling_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
            runtime_draft_len=runtime_draft_len,
            draft_lens=draft_lens,
            dspark_confidence_execution_batch_size=outputs.get(
                "dspark_confidence_execution_batch_size"),
            dspark_confidence_route_epoch=outputs.get(
                "dspark_confidence_route_epoch"),
            dspark_confidence_verifier_token_budget=outputs.get(
                "dspark_confidence_verifier_token_budget"),
            dspark_confidence_physical_draft_len=runtime_draft_len,
            dspark_confidence_engine_generation=outputs.get(
                "dspark_confidence_engine_generation"),
            dspark_confidence_native_uniform=native_uniform,
        )
