import torch
from dataclasses import dataclass
from functools import partial
from transformers.utils import ModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    BaseModelOutputWithPast,
    logger,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import Optional, Tuple, Union, List

from ..samd_config import ForwardType



@dataclass
class SamdCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def llama_casuallm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, SamdCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = last_hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [torch.nn.functional.linear(hidden_states[:, slice_indices, :], lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    logits = logits.float()

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return SamdCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        last_hidden_states=last_hidden_states,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    return_dict: Optional[bool] = None,
    **flash_attn_kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    if input_ids is None and inputs_embeds is None:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        batch_size, seq_length = inputs_embeds.shape[:2]

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    if self.forward_state.forward_type == ForwardType.tree_decode and self.mask_state.mask is not None:
        if causal_mask is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_static_cache = isinstance(past_key_values, StaticCache)
            target_length = (
                past_key_values.get_max_cache_shape()
                if using_static_cache
                else past_seen_tokens + inputs_embeds.shape[1] + 1
            )
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=inputs_embeds.shape[1],
                target_length=target_length,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                cache_position=cache_position,
                batch_size=batch_size,
            )
        samd_attn_mask: torch.Tensor = self.mask_state.mask.to(causal_mask.device)
        tail_size = samd_attn_mask.shape[-1]
        if causal_mask.dtype == torch.bool:
            causal_mask[:, :, :, -tail_size:] = causal_mask[:, :, :, -tail_size:] & samd_attn_mask.bool()
        else:
            min_dtype = torch.finfo(causal_mask.dtype).min
            causal_mask[:, :, :, -tail_size:] = causal_mask[:, :, :, -tail_size:].masked_fill(
                samd_attn_mask == 0, min_dtype
            )

    # SAMD speculative decode can use custom cache semantics where HF's target_length heuristic
    # in `_update_causal_mask` does not always match the actual KV length consumed by attention.
    # Normalize the mask to [past_len + current_len] to keep SDPA tensor shapes aligned.
    if causal_mask is not None and causal_mask.dim() == 4:
        if isinstance(past_key_values, StaticCache):
            expected_kv_len = past_key_values.get_max_cache_shape()
        else:
            effective_past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            expected_kv_len = effective_past_len + inputs_embeds.shape[1]
        cur_kv_len = causal_mask.shape[-1]
        if cur_kv_len > expected_kv_len:
            causal_mask = causal_mask[..., :expected_kv_len]
        elif cur_kv_len < expected_kv_len:
            if causal_mask.dtype == torch.bool:
                pad = torch.zeros(
                    (*causal_mask.shape[:-1], expected_kv_len - cur_kv_len),
                    dtype=causal_mask.dtype,
                    device=causal_mask.device,
                )
            else:
                min_dtype = torch.finfo(causal_mask.dtype).min
                pad = torch.full(
                    (*causal_mask.shape[:-1], expected_kv_len - cur_kv_len),
                    fill_value=min_dtype,
                    dtype=causal_mask.dtype,
                    device=causal_mask.device,
                )
            causal_mask = torch.cat([causal_mask, pad], dim=-1)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                partial(decoder_layer.__call__, **flash_attn_kwargs),
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states, all_self_attns] if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


llama_patch_dict = {
    LlamaForCausalLM: [("forward", llama_casuallm_forward)]
}

llama_attn_patch_dict = {
    LlamaModel: [("forward", llama_model_forward)]
}
