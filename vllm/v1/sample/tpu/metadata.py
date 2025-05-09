# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional

import torch

from vllm.v1.worker.gpu_input_batch import InputBatch
import vllm.envs as envs
DEFAULT_SAMPLING_PARAMS = dict(
    temperature=-1.0,
    min_p=0.0,
    # strictly disabled for now
    top_k=0,
    top_p=1.0,
    # frequency_penalties=0.0,
    # presence_penalties=0.0,
    # repetition_penalties=0.0,
)
if envs.VLLM_TORCHAX_ENABLED:
    from jax.tree_util import register_pytree_node


@dataclass
class TPUSupportedSamplingMetadata:
    # This class exposes a more xla-friendly interface than SamplingMetadata
    # on TPU, in particular all arguments should be traceable and no optionals
    # are allowed, to avoid graph recompilation on Nones.
    temperature: torch.Tensor = None

    min_p: torch.Tensor = None
    top_k: torch.Tensor = None
    top_p: torch.Tensor = None

    all_greedy: bool = True

    # unsupported, you need to return an extra tensor of static size BxV
    max_num_logprobs = None

    # TODO No penalties for now
    no_penalties: bool = True
    prompt_token_ids = None
    frequency_penalties = None
    presence_penalties = None
    repetition_penalties = None
    # should use tensor
    output_token_ids: list[list[int]] = field(default_factory=lambda: list())

    min_tokens = None  # impl is not vectorized

    logit_bias: list[Optional[dict[int, float]]] = field(
        default_factory=lambda: list())

    allowed_token_ids_mask = None
    bad_words_token_ids = None

    # Generator not supported by xla
    _generators: dict[int,
                      torch.Generator] = field(default_factory=lambda: dict())

    @property
    def generators(self) -> dict[int, torch.Generator]:
        # Generator not supported by torch/xla. This field must be immutable.
        return self._generators

    @classmethod
    def from_input_batch(
        cls,
        input_batch: InputBatch,
        padded_num_reqs: int,
        xla_device: torch.device,
        generate_params_if_all_greedy: bool = False
    ) -> "TPUSupportedSamplingMetadata":
        """
        Copy sampling tensors slices from `input_batch` to on device tensors.

        `InputBatch._make_sampling_metadata` causes recompilation on XLA as it 
        slices dynamic shapes on device tensors. This impl moves the dynamic 
        ops to CPU and produces tensors of fixed `padded_num_reqs` size.

        Args:
            input_batch: The input batch containing sampling parameters.
            padded_num_reqs: The padded number of requests.
            xla_device: The XLA device.
            generate_params_if_all_greedy: If True, generate sampling parameters
                even if all requests are greedy. this is useful for cases where
                we want to pre-compile a graph with sampling parameters, even if
                they are not strictly needed for greedy decoding.
        """
        # Early return to avoid unnecessary cpu to tpu copy
        if (input_batch.all_greedy is True
                and generate_params_if_all_greedy is False):
            return cls(all_greedy=True)

        num_reqs = input_batch.num_reqs

        def fill_slice(cpu_tensor: torch.Tensor, fill_val) -> torch.Tensor:
            # Pad value is the default one.
            cpu_tensor[num_reqs:padded_num_reqs] = fill_val

        fill_slice(input_batch.temperature_cpu_tensor,
                   DEFAULT_SAMPLING_PARAMS["temperature"])
        fill_slice(input_batch.min_p_cpu_tensor,
                   DEFAULT_SAMPLING_PARAMS["min_p"])
        fill_slice(input_batch.top_k_cpu_tensor,
                   DEFAULT_SAMPLING_PARAMS["top_k"])
        fill_slice(input_batch.top_p_cpu_tensor,
                   DEFAULT_SAMPLING_PARAMS["top_p"])

        # Slice persistent device tensors to a fixed pre-compiled padded shape.
        return cls(
            temperature=input_batch.temperature_cpu_tensor[:padded_num_reqs].
            to(xla_device),
            all_greedy=input_batch.all_greedy,
            # TODO enable more and avoid returning None values
            top_p=input_batch.top_p_cpu_tensor[:padded_num_reqs].to(
                xla_device),
            top_k=input_batch.top_k_cpu_tensor[:padded_num_reqs].to(
                xla_device),
            min_p=input_batch.min_p_cpu_tensor[:padded_num_reqs].to(
                xla_device))

if envs.VLLM_TORCHAX_ENABLED:
    def flatten_tpu_metadata(metadata):
        children = (
            metadata.temperature,
            metadata.min_p,
            metadata.top_k,
            metadata.top_p,
            metadata.output_token_ids,
            metadata.logit_bias,
            )
        aux_data = {
            "all_greedy": metadata.all_greedy,
            "no_penalties": metadata.no_penalties,
            "prompt_token_ids": metadata.prompt_token_ids,
            "frequency_penalties": metadata.frequency_penalties,
            "presence_penalties": metadata.presence_penalties,
            "repetition_penalties": metadata.repetition_penalties,
            "min_tokens": metadata.min_tokens,
            "allowed_token_ids_mask": metadata.allowed_token_ids_mask,
            "bad_words_token_ids": metadata.bad_words_token_ids,
            "_generators": metadata._generators,
        }
        
        return children, aux_data

    def unflatten_tpu_metadata(aux_data, children):
        temperature, min_p, top_k, top_p, output_token_ids, logit_bias = children
        
        return TPUSupportedSamplingMetadata(
            temperature=temperature,
            min_p=min_p,
            top_k=top_k,
            top_p=top_p,
            all_greedy=aux_data["all_greedy"],
            no_penalties=aux_data["no_penalties"],
            output_token_ids=output_token_ids,
            logit_bias=logit_bias,
            _generators=aux_data["_generators"],
        )

    register_pytree_node(
        TPUSupportedSamplingMetadata,
        flatten_tpu_metadata,
        unflatten_tpu_metadata
    )
