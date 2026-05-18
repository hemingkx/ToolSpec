import argparse
from fastchat.utils import str_to_torch_dtype
from evaluation.eval_toolspec import run_eval, reorg_answer_file

from transformers import AutoConfig, AutoTokenizer
from transformers import StoppingCriteriaList, MaxLengthCriteria

from model.toolspec.kv_cache import initialize_past_key_values
from model.toolspec.modeling_qwen_kv import Qwen2ForCausalLM
from model.toolspec.modeling_llama_kv import LlamaForCausalLM
from model.toolspec.tree_template_ import choose_tree_template

import torch
import torch.nn.functional as F


def _normalize_token_ids(token_ids):
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple)):
        return [int(x) for x in token_ids]
    return [int(token_ids)]


def _first_token_index(tokens_1d: torch.Tensor, token_ids_1d: torch.Tensor):
    if tokens_1d.numel() == 0 or token_ids_1d.numel() == 0:
        return None
    pos = torch.nonzero(torch.isin(tokens_1d, token_ids_1d), as_tuple=False)
    if pos.numel() == 0:
        return None
    return int(pos[0].item())


def _truncate_generated_suffix(
    verify_input_ids: torch.Tensor,
    start_len: int,
    eos_token_ids: torch.Tensor,
    header_token_ids: torch.Tensor,
    accept_length_list,
):
    # Keep EOS if it appears first; drop assistant header token and anything after it.
    gen_tokens = verify_input_ids[:, start_len:]
    first_eos_idx = _first_token_index(gen_tokens[0], eos_token_ids)
    first_header_idx = _first_token_index(gen_tokens[0], header_token_ids)

    cut_generated_len = None
    if first_eos_idx is not None and (first_header_idx is None or first_eos_idx <= first_header_idx):
        cut_generated_len = first_eos_idx + 1
    elif first_header_idx is not None:
        cut_generated_len = first_header_idx

    if cut_generated_len is None:
        return verify_input_ids

    original_len = verify_input_ids.size(1)
    verify_input_ids = verify_input_ids[:, : start_len + cut_generated_len]
    truncated = original_len - verify_input_ids.size(1)
    if truncated > 0 and len(accept_length_list) > 0:
        accept_length_list[-1] = max(1, accept_length_list[-1] - truncated)
    return verify_input_ids


@torch.no_grad()
def prepare_data(template, input_ids):
    candi_attention_ids = torch.zeros([input_ids.size(0), len(template)+1, len(template)+1], dtype=torch.long)
    candi_positon_ids = torch.zeros([input_ids.size(0), len(template)+1], dtype=torch.long)

    candi_attention_ids[:, :, 0] = 1
    candi_positon_ids[:, 0] = 0
    search_path = []
    search_path_set = set()
    father_index = [-1]
    candi_index = [-1]
    deep_split = []
    
    # Optimization: Build a dict for O(1) parent lookup instead of O(n) template.index()
    template_to_idx = {tuple(path): idx + 1 for idx, path in enumerate(template)}
    
    for candi_i in range(len(template)):
        tree_deep = len(template[candi_i])
        tree_index = template[candi_i][-1]

        if tree_deep == 1:
            father_index.append(0)
        else:
            parent_path = tuple(template[candi_i][:-1])
            father_index.append(template_to_idx.get(parent_path, 0))
        candi_index.append(tree_index)

        candi_positon_ids[:, candi_i+1] = tree_deep

        if tree_deep != candi_positon_ids[:, candi_i]:
            deep_split.append(candi_i+1)

        cur_path = [0]
        candi_attention_ids[:, candi_i+1, candi_i+1] = 1

        for candi_j in range(candi_i):
            if template[candi_j] == template[candi_i][:len(template[candi_j])]:
                candi_attention_ids[:, candi_i+1, candi_j+1] = 1
                cur_path.append(candi_j+1)
        cur_path.append(candi_i+1)

        # Keep search_path_set incrementally to avoid rebuilding it every iteration.
        to_remove = []
        for sub_i in range(1, len(cur_path)):
            sub_path = tuple(cur_path[:sub_i])
            if sub_path in search_path_set:
                to_remove.append(sub_path)
        if to_remove:
            remove_set = set(to_remove)
            search_path = [p for p in search_path if tuple(p) not in remove_set]
            search_path_set.difference_update(remove_set)
        search_path.append(cur_path)
        search_path_set.add(tuple(cur_path))

    deep_split.append(len(template)+1)

    # Pad all paths in search_path to the same length.
    # Originally pad_len was tied to len(template[-1]) + 1 for a specific tree template.
    pad_len = max(len(path) for path in search_path)
    search_path = torch.tensor(
        [path + [-1] * (pad_len - len(path)) for path in search_path],
        dtype=torch.long,
        device=torch.device('cuda:0'),
    )
    father_index = torch.tensor(father_index, dtype=torch.long, device=torch.device('cuda:0'))
    candi_index = torch.tensor(candi_index, dtype=torch.long, device=torch.device('cuda:0'))
    
    return candi_attention_ids.cuda(), candi_positon_ids.cuda(), search_path, father_index, candi_index, deep_split

@torch.no_grad()
def create_template_from_candidates(candidate_pred_tokens):
    """
    Create a new tree template based on candidate_pred_tokens.
    For each root that has a valid candidate, creates paths from root to the candidate's depth.
    
    Args:
        candidate_pred_tokens: List of candidate sequences (each is a tensor, can be empty)
    
    Returns:
        new_template: List of paths created based on candidates
    """
    new_template = []
    
    # Process all candidates (number of roots equals number of candidates)
    for root_idx in range(len(candidate_pred_tokens)):
        seq = candidate_pred_tokens[root_idx]
        if seq.numel() == 0:
            # Skip empty candidates
            continue
        
        # Get the depth (length) of this candidate
        candidate_depth = seq.size(0)
        
        # Create paths for this root: [root], [root,0], [root,0,0], ..., up to candidate_depth
        for depth in range(1, candidate_depth + 1):
            if depth == 1:
                # Root node: [root_idx]
                path = [root_idx]
            else:
                # Deeper nodes: [root_idx, 0, 0, ..., 0] with (depth-1) zeros
                path = [root_idx] + [0] * (depth - 1)
            new_template.append(path)
    
    # If no valid candidates, return minimal template (just root path [0])
    if not new_template:
        return [[0]]
    
    return new_template


@torch.no_grad()
def prepare_data_input_ids(cur_input_ids, adj_matrix, father_index, candi_index, deep_split, output_id_topk):
    input_ids = torch.zeros_like(father_index, dtype=torch.long, device=cur_input_ids.device)
    input_ids[0] = cur_input_ids
    for layer in range(len(deep_split)-1):
        cur_father = input_ids[father_index[deep_split[layer]:deep_split[layer+1]]]
        cur_father_index = cur_father * output_id_topk + candi_index[deep_split[layer]:deep_split[layer+1]]
        input_ids[deep_split[layer]:deep_split[layer+1]] = adj_matrix.view(-1)[cur_father_index]
    return input_ids.unsqueeze(0)


@torch.no_grad()
def build_retrieval_tree_from_candidates(candidate_pred_tokens, input_ids):
    """
    Build a retrieval tree from candidate prediction tokens.
    
    Args:
        candidate_pred_tokens: List of candidate token sequences (each is a tensor)
        input_ids: Current input_ids tensor
        
    Returns:
        tuple: (input_ids, tree_template, attention_mask, position_ids, search_path, 
                father_index, candi_index, deep_split)
    """
    # Create a new template based on candidate_pred_tokens
    new_template = create_template_from_candidates(candidate_pred_tokens)
    template_key = tuple(tuple(path) for path in new_template)

    # Lightweight bounded cache for repeated template structures.
    cache = getattr(build_retrieval_tree_from_candidates, "_template_cache", None)
    if cache is None:
        cache = {}
        build_retrieval_tree_from_candidates._template_cache = cache

    cached = cache.get(template_key)
    if cached is not None and cached[0].device == input_ids.device:
        attention_mask, position_ids, search_path, father_index, candi_index, deep_split = cached
    else:
        # Compute prepare_data with the new template
        attention_mask, position_ids, search_path, father_index, candi_index, deep_split = prepare_data(
            new_template, input_ids
        )
        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[template_key] = (
            attention_mask,
            position_ids,
            search_path,
            father_index,
            candi_index,
            deep_split,
        )

    # Initialize a flat input_ids buffer for all tree nodes (index 0 is dummy root)
    retrieval_input_ids = torch.zeros_like(father_index, dtype=torch.long, device=input_ids.device)
    retrieval_input_ids[0] = input_ids[0, 0]

    # Map candidate sequences onto the new tree in a single pass over nodes.
    for node_idx, path in enumerate(new_template, start=1):
        if not path:
            continue
        root_idx = path[0]
        if root_idx < 0 or root_idx >= len(candidate_pred_tokens):
            continue
        seq = candidate_pred_tokens[root_idx]
        if seq.numel() == 0:
            continue
        depth = len(path)
        if depth <= seq.size(0):
            retrieval_input_ids[node_idx] = seq[depth - 1]

    input_ids = retrieval_input_ids.unsqueeze(0)
    tree_template = new_template  # Use new template for consistency
    
    return input_ids, tree_template, attention_mask, position_ids, search_path, father_index, candi_index, deep_split


@torch.no_grad()
def find_candidate_pred_tokens(
    input_ids,
    similar_outputs=None,
    retrieved_context=None,
    max_ngram_size=7,
    min_ngram_size=5,
    target_lengths=(32, 16, 8, 8), # 64 nodes
):
    """
    Return up to target_lengths candidate sequences by
    searching from the longest n-gram downwards. For each requested length,
    we take the first matching continuation we can find. Missing candidates
    are returned as empty tensors to keep a fixed-length list.
    
    Args:
        input_ids: The input ids of the current sequence
        similar_outputs: List of similar outputs from memory (each with "output_ids" key)
        max_ngram_size: Maximum n-gram size to search
        min_ngram_size: Minimum n-gram size to search
        target_lengths: Target lengths for candidate sequences
    """
    input_length = input_ids.size(1)
    
    # Merge retrieval memory into search context.
    if retrieved_context is not None and retrieved_context.numel() > 0:
        search_context = torch.cat([retrieved_context, input_ids], dim=1)
    else:
        search_context = input_ids
    if retrieved_context is None and similar_outputs:
        retrieved = []
        for entry in similar_outputs:
            output_ids = entry.get("output_ids") if isinstance(entry, dict) else entry
            if not output_ids:
                continue
            # Convert to tensor if it's a list
            if isinstance(output_ids, list):
                retrieved.append(torch.tensor(output_ids, device=input_ids.device).unsqueeze(0))
            else:
                retrieved.append(output_ids.unsqueeze(0) if output_ids.dim() == 1 else output_ids)
        if retrieved:
            search_context = torch.cat(retrieved + [search_context], dim=1)

    if max_ngram_size <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size")

    candidates = []
    seen_tensors = []  # store tensors for efficient prefix checking
    for ngram_size in range(max_ngram_size, min_ngram_size - 1, -1):
        if len(candidates) == len(target_lengths):
            break

        # Extract the last n tokens as our search ngram (keep as tensor to avoid conversion overhead)
        ngram_tensor = input_ids[0, -ngram_size:].unsqueeze(0)

        # Sliding windows of size ngram_size
        windows = search_context.unfold(dimension=1, size=ngram_size, step=1)

        matches = (windows == ngram_tensor).all(dim=2)
        match_indices = matches.nonzero(as_tuple=True)[1]

        for idx in match_indices:
            if len(candidates) == len(target_lengths):
                break
            start_idx = idx + ngram_size
            end_idx = start_idx + target_lengths[len(candidates)]
            # ensure continuation exists and is not self-match
            # Check against search_context length, but ensure we're not matching within original input_ids
            search_context_length = search_context.size(1)
            if end_idx <= search_context_length and start_idx < input_length - ngram_size:
                candidate_seq = search_context[0, start_idx:end_idx]
                # Use tensor for prefix checking (much faster than converting to tuple)
                seq_len = candidate_seq.size(0)
                is_duplicate = False
                # Check for exact duplicates and prefixes using tensor operations
                for existing_tensor in seen_tensors:
                    existing_len = existing_tensor.size(0)
                    if existing_len == seq_len:
                        if (candidate_seq == existing_tensor).all():
                            is_duplicate = True
                            break
                    elif existing_len > seq_len:
                        # Check if candidate is prefix of existing
                        if (candidate_seq == existing_tensor[:seq_len]).all():
                            is_duplicate = True
                            break
                    # Check if existing is prefix of candidate (candidate longer)
                    elif seq_len > existing_len:
                        if (existing_tensor == candidate_seq[:existing_len]).all():
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    continue
                
                # Store for future checks (clone to avoid view issues)
                seen_tensors.append(candidate_seq.clone())
                candidates.append(candidate_seq)

    # Pad with empty tensors if fewer than requested
    while len(candidates) < len(target_lengths):
        candidates.append(torch.tensor([], dtype=torch.long, device=input_ids.device))

    return candidates


@torch.no_grad()
def get_topk_similar_outputs(question_hidden_state, output_memory, top_k=3):
    """
    Compute cosine similarity between the current question representation and historical ones.
    Optimized to batch similarity computations.
    """
    if question_hidden_state is None or not output_memory:
        return []

    # Prepare query vector once
    q_vec = question_hidden_state.to(torch.float32).view(question_hidden_state.shape[0], -1)
    q_vec = F.normalize(q_vec, dim=-1)
    
    # Batch all memory states for efficient computation
    valid_memories = []
    memory_states_list = []
    for memory in output_memory:
        memory_state = memory.get("question_hidden_state")
        memory_output = memory.get("output_ids")
        memory_qid = memory.get("question_id")
        if memory_state is None or memory_output is None:
            continue
        if isinstance(memory_state, torch.Tensor):
            mem_vec = memory_state.to(torch.float32).view(memory_state.shape[0], -1)
        else:
            mem_vec = torch.tensor(memory_state, dtype=torch.float32).view(1, -1)
        valid_memories.append((memory_qid, memory_output))
        memory_states_list.append(mem_vec)
    
    if not memory_states_list:
        return []

    # Batch normalize all memory vectors at once (they should all have same feature dim)
    memory_states_tensor = torch.cat(memory_states_list, dim=0)
    memory_states_tensor = F.normalize(memory_states_tensor, dim=-1)
    
    # Compute all similarities in one batch operation
    # q_vec shape: [1, hidden_dim], memory_states_tensor shape: [num_memories, hidden_dim]
    similarities_tensor = torch.sum(q_vec * memory_states_tensor, dim=-1)
    similarities = [(sim.item(), valid_memories[i][0], valid_memories[i][1]) 
                    for i, sim in enumerate(similarities_tensor)]
    
    if not similarities:
        return []

    # Sort and return top-k
    similarities.sort(key=lambda x: x[0], reverse=True)
    topk = similarities[:top_k]
    return [
        {"similarity": sim, "question_id": qid, "output_ids": output_ids}
        for sim, qid, output_ids in topk
    ]


@torch.no_grad()
def toolspec_forward(
    inputs,
    output_memory,
    schema_fsm,
    model,
    tokenizer,
    max_new_tokens,
    output_id_topk=8,
    adj_matrix=None,
):
    input_ids = inputs.input_ids.cuda()
    accept_length_list = []
    step = 0
    start_len = input_ids.size(1)
    verify_input_ids = input_ids[:, :-1]
    input_ids = input_ids[:, -1:]
    
    # Extract question hidden state from the initial forward pass
    question_hidden_state = None
    question_length = verify_input_ids.size(1)
    general_template = choose_tree_template("general")
    max_cache_len = start_len + max_new_tokens + len(general_template) + 64
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model, max_cache_len=max_cache_len)
    stopping_criteria = StoppingCriteriaList(
        [MaxLengthCriteria(max_length=start_len + max_new_tokens)]
    )

    # Always extract hidden state to return it, and to use for similarity search if memory exists
    initial_outputs = model(
        input_ids=verify_input_ids,
        past_key_values=past_key_values,
        output_hidden_states=True,
    )
    if initial_outputs.hidden_states is not None:
        # Extract hidden state at the last token of the question
        question_hidden_state = initial_outputs.hidden_states[-1][:, question_length - 1:question_length, :].detach().cpu()
    
    eos_token_values = []
    eos_token_values.extend(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    eos_token_values.extend(_normalize_token_ids(getattr(model.generation_config, "eos_token_id", None)))
    eos_token_ids = torch.tensor(
        sorted(set(eos_token_values)),
        dtype=torch.long,
        device=input_ids.device,
    )
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    header_token_values = [int(start_header_id)] if start_header_id is not None and start_header_id >= 0 else []
    header_token_ids = torch.tensor(
        header_token_values,
        dtype=torch.long,
        device=input_ids.device,
    )
    stop_token_ids = torch.tensor(
        sorted(set(eos_token_values + header_token_values)),
        dtype=torch.long,
        device=input_ids.device,
    )

    def has_stop(token_tensor: torch.Tensor) -> bool:
        if stop_token_ids.numel() == 0:
            return False
        return torch.isin(token_tensor, stop_token_ids).any().item()

    # Get top-k similar outputs from memory if available
    similar_outputs = []
    if question_hidden_state is not None and output_memory:
        similar_outputs = get_topk_similar_outputs(question_hidden_state, output_memory, top_k=10)
    retrieved_context = None
    if similar_outputs:
        retrieved_chunks = []
        for entry in similar_outputs:
            output_ids = entry.get("output_ids") if isinstance(entry, dict) else entry
            if not output_ids:
                continue
            if isinstance(output_ids, list):
                retrieved_chunks.append(torch.tensor(output_ids, device=input_ids.device).unsqueeze(0))
            else:
                retrieved_chunks.append(output_ids.unsqueeze(0) if output_ids.dim() == 1 else output_ids)
        if retrieved_chunks:
            retrieved_context = torch.cat(retrieved_chunks, dim=1)

    verify_tool_name = None
    # Cache general-tree structure (independent of token values).
    general_tree_cache = None
    # Cache tool-name tensors on device to avoid repeated cpu().tolist() syncs.
    tool_name_tensors = []
    for tool_name_tuple in schema_fsm.tokenized_schema.keys():
        tool_name_tensors.append(
            (
                len(tool_name_tuple),
                torch.tensor(tool_name_tuple, dtype=torch.long, device=input_ids.device),
                tool_name_tuple,
            )
        )
    for step in range(max_new_tokens):
        full_input_ids = torch.cat([verify_input_ids, input_ids], dim=-1)
        param_span = full_input_ids[0, -4:].tolist()

        if schema_fsm.state == "init" or schema_fsm.state == "first_param":
            candidate_pred_tokens = schema_fsm.find_candidate_pred_tokens(tool_name=verify_tool_name)
            # Convert list of lists to list of tensors
            candidate_pred_tokens = [
                torch.tensor(tokens, dtype=torch.long, device=input_ids.device) 
                if tokens else torch.tensor([], dtype=torch.long, device=input_ids.device)
                for tokens in candidate_pred_tokens
            ]

            # retrieval tree: fill the tree with the found candidate prediction tokens
            input_ids, tree_template, attention_mask, position_ids, search_path, father_index, candi_index, deep_split = build_retrieval_tree_from_candidates(
                candidate_pred_tokens, input_ids
            )
            schema_fsm.state_transition()
        else:
            # Pass similar_outputs to find_candidate_pred_tokens to merge into search context
            schema_tokens = None
            if schema_fsm.seperator_id in param_span:
                schema_tokens = schema_fsm.find_candidate_pred_tokens(tool_name=verify_tool_name, param_span=param_span)

            candidate_pred_tokens = find_candidate_pred_tokens(
                full_input_ids,
                retrieved_context=retrieved_context,
            )
            non_empty_candidates = [c for c in candidate_pred_tokens if c.numel() > 0]
            #print("non empty candidates: \n", non_empty_candidates)

            if len(non_empty_candidates) == 0:
                if schema_tokens is not None:
                    candidate_pred_tokens = [
                        torch.tensor(tokens, dtype=torch.long, device=input_ids.device) 
                        if tokens else torch.tensor([], dtype=torch.long, device=input_ids.device)
                        for tokens in schema_tokens
                    ]
                    # retrieval tree: fill the tree with the found candidate prediction tokens
                    input_ids, tree_template, attention_mask, position_ids, search_path, father_index, candi_index, deep_split = build_retrieval_tree_from_candidates(
                        candidate_pred_tokens, input_ids
                    )
                else:
                    # general tree: use logits top-k candidates (original tokenrecycling behavior)
                    if general_tree_cache is None:
                        tree_template = general_template
                        general_tree_cache = (tree_template,) + prepare_data(tree_template, input_ids)
                    (
                        tree_template,
                        attention_mask,
                        position_ids,
                        search_path,
                        father_index,
                        candi_index,
                        deep_split,
                    ) = general_tree_cache
                    input_ids = prepare_data_input_ids(
                        input_ids, adj_matrix, father_index, candi_index, deep_split, output_id_topk
                    )
            else:
                # retrieval tree: fill the tree with the found candidate prediction tokens
                input_ids, tree_template, attention_mask, position_ids, search_path, father_index, candi_index, deep_split = build_retrieval_tree_from_candidates(
                    candidate_pred_tokens, input_ids
                )

        # Common verification logic for both general and retrieval trees
        # Optimization: Pre-allocate and reuse tensor if possible, avoid repeated device() calls
        verify_len = verify_input_ids.size(1)
        expected_prefix_shape = (input_ids.size(0), len(tree_template) + 1, verify_len)
        if (
            not hasattr(toolspec_forward, "_merge_attn_prefix")
            or tuple(toolspec_forward._merge_attn_prefix.shape) != expected_prefix_shape
            or toolspec_forward._merge_attn_prefix.device != input_ids.device
        ):
            toolspec_forward._merge_attn_prefix = torch.ones(
                expected_prefix_shape,
                dtype=torch.bool,
                device=input_ids.device,
            )
        merge_attention_mask = torch.cat([toolspec_forward._merge_attn_prefix, attention_mask.bool()], dim=-1)
        merge_positon_ids = position_ids + verify_len

        outputs = model(
            input_ids=input_ids,
            attention_mask=merge_attention_mask.unsqueeze(1),
            position_ids=merge_positon_ids,
            past_key_values=past_key_values,
        )
        
        model_res = torch.argmax(outputs.logits, dim=-1)
        
        all_input_path = input_ids[0][search_path]
        all_input_path[search_path==-1] = -100
        all_output_path = model_res[0][search_path]
        reward = torch.cumprod(all_input_path[:, 1:].eq(all_output_path[:, :-1]), dim=-1).sum(dim=-1)
        best_reward = reward.max()
        
        accept_len = 1 + best_reward
        accept_length_list.append(accept_len.item())
        best_path_index = torch.argmax(reward, dim=-1).to(torch.long)
        index_path = search_path[best_path_index][:accept_len]
        best_path_input = torch.index_select(input_ids, index=index_path, dim=1)

        tgt = past_key_values_data[..., verify_input_ids.size(1)+index_path, :]
        dst = past_key_values_data[..., verify_input_ids.size(1) : verify_input_ids.size(1) + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
        
        current_length_data.fill_(verify_input_ids.size(1) + tgt.shape[-2])
                
        verify_input_ids = torch.cat([verify_input_ids, best_path_input], dim=-1)

        # Verify tool name: check if the last tokens match any tool name in the schema
        verify_seq = verify_input_ids[0]  # Get the sequence tensor
        if not verify_tool_name:
            for tool_name_len, tool_name_tensor, tool_name_tuple in tool_name_tensors:
                if verify_seq.size(0) >= tool_name_len:
                    if torch.equal(verify_seq[-tool_name_len:], tool_name_tensor):
                        verify_tool_name = tool_name_tuple
                        break

        if has_stop(best_path_input) or stopping_criteria(verify_input_ids, None):
            break
        
        to_update = torch.topk(outputs.logits, k=8, dim=-1)[1][0]
        adj_matrix[input_ids.squeeze(0)] = to_update
        
        # Optimization: Avoid redundant .cuda() call - already on GPU
        input_ids = model_res[:, search_path[best_path_index][accept_len-1]].unsqueeze(-1)

        if has_stop(input_ids):
            break

        step += 1

    verify_input_ids = _truncate_generated_suffix(
        verify_input_ids=verify_input_ids,
        start_len=start_len,
        eos_token_ids=eos_token_ids,
        header_token_ids=header_token_ids,
        accept_length_list=accept_length_list,
    )

    return verify_input_ids, verify_input_ids.size(1) - start_len, step, accept_length_list, question_hidden_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--question-num", type=int, default=-1, help="The number of questions to evaluate.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--output-id-topk",
        type=int,
        default=8,
        help="how many tokens from output_ids to be used to renew the tree.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="toolalpaca",
        choices=["toolalpaca", "APIBank", "bfcl"],
        help="The benchmark to evaluate.",
    )
    args = parser.parse_args()

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"output/{args.benchmark}/{args.model_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model_config = AutoConfig.from_pretrained(args.model_path)
    if model_config.model_type == "llama":
        model_cls = LlamaForCausalLM
    elif model_config.model_type == "qwen2":
        model_cls = Qwen2ForCausalLM
    else:
        raise ValueError(f"Unsupported model_type for ToolSpec: {model_config.model_type}")

    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    adj_matrix = torch.zeros((model.vocab_size, args.output_id_topk), dtype=torch.long, device=model.device, requires_grad=False)
    adj_matrix_memory = adj_matrix.element_size() * adj_matrix.nelement()
    adj_matrix_memory_MB = adj_matrix_memory / (1024 ** 2)

    print(f'Adj_matrix size: ({model.vocab_size}, {args.output_id_topk}), memory: {adj_matrix_memory_MB:.2f} MB')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)


    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=toolspec_forward,
        model_id=args.model_id,
        answer_file=answer_file,
        question_num=args.question_num,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        output_id_topk=args.output_id_topk,
        benchmark=args.benchmark,
        adj_matrix=adj_matrix,
    )

    reorg_answer_file(answer_file)