import re
import ast
from typing import Dict, List


class SchemaFSM:
    def __init__(self, tools_metadata=None, tokenizer=None):
        """
        Initialize SchemaFSM with tool schema.
        
        Args:
            tools_metadata: Can be:
                - A dict mapping tool names to parameter lists: {"ToolName": ["param1", "param2"]}
                - A string (system prompt) to extract schema from
                - None (will be empty, can be populated later)
        """
        self.tokenizer = tokenizer
        if tools_metadata is None:
            self.schema = {}
        elif isinstance(tools_metadata, dict):
            self.schema = tools_metadata
        elif isinstance(tools_metadata, str):
            self.schema = self.extract_schema_from_system_prompt(tools_metadata)
        else:
            raise ValueError("tools_metadata must be dict, str, or None")
        # init_prefix suggestions: 
        # Qwen2.5, llama3.2-3B: self.init_prefix = ['{\"name\": \"', '<tool_call>\n{\"name\": \"', '<tool_call>\n  {\"name\": \"', '```<tool_call>\n{\"name\": \"']
        # llama3.1-8B ['<tool_call>\n{\"name\": \"', '</tool_call>\n{\"name\": \"']
        self.init_prefix =   ['{\"name\": \"', '<tool_call>\n{\"name\": \"', '<tool_call>\n  {\"name\": \"', '```<tool_call>\n{\"name\": \"']
        self.param_prefix = '\", \"parameters\": {\"'
        self.param_suffix = '\": \"'
        self.remaining_param_prefix = ' \"'

        self.tokenized_init_prefix = [self._encode(prefix) for prefix in self.init_prefix]
        self.tokenized_param_prefix = self._encode(self.param_prefix)
        self.tokenized_param_suffix = self._encode(self.param_suffix)
        self.tokenized_remaining_param_prefix = self._encode(self.remaining_param_prefix)
        self.seperator_id = self._encode("\",")[0]
        self.tokenized_schema = self.convert_schema_values_to_token_ids()

        self.state = "init" # init, first_param, remaining_param
    
    def state_transition(self):
        if self.state == "init":
            self.state = "first_param"
        elif self.state == "first_param":
            self.state = "remaining_param"
        else:
            raise ValueError("Invalid state")

    def find_candidate_pred_tokens(self, tool_name=None, param_span=None):
        if self.state == "init":
            candidates = []
            # Use tokenized tool names directly from tokenized_schema
            for init_prefix in self.tokenized_init_prefix:
                for tool_name_token_ids_tuple in self.tokenized_schema.keys():
                    # Convert tuple to list for concatenation
                    tool_name_token_ids = list(tool_name_token_ids_tuple)
                    candidate_token_ids = init_prefix + tool_name_token_ids
                    candidates.append(candidate_token_ids)
            return candidates
        elif self.state == "first_param":
            candidates = []
            if tool_name == None:
                # If tool name is None, only use the param prefix
                candidate_token_ids = self.tokenized_param_prefix[1:]
                candidates.append(candidate_token_ids)
            else:
                if tool_name not in self.tokenized_schema:
                    raise ValueError(f"Tool name {tool_name} not found in schema")
                param_list = self.tokenized_schema[tool_name]
                if not param_list:
                    # If tool has no parameters, only use the param prefix
                    candidate_token_ids = self.tokenized_param_prefix[1:]
                    candidates.append(candidate_token_ids)
                else:
                    for param_token_ids in param_list:
                        candidate_token_ids = self.tokenized_param_prefix[1:] + param_token_ids + self.tokenized_param_suffix
                        candidates.append(candidate_token_ids)
            return candidates
        elif self.state == "remaining_param":
            candidates = []
            if tool_name == None:
                return candidates
            if tool_name not in self.tokenized_schema:
                raise ValueError(f"Remaining param: tool name {tool_name} not found in schema")
            # Extract tokens after seperator_id in param_span
            separator_idx = param_span.index(self.seperator_id)
            # Get all tokens after the separator
            tokens_after_separator = param_span[separator_idx + 1:]
            
            param_list = self.tokenized_schema[tool_name]
            for param_token_ids in param_list:
                candidate_token_ids = self.tokenized_remaining_param_prefix + param_token_ids + self.tokenized_param_suffix
                # Only append candidates where the prefix matches tokens_after_separator
                if len(candidate_token_ids) >= len(tokens_after_separator):
                    candidate_prefix = candidate_token_ids[:len(tokens_after_separator)]
                    if candidate_prefix == tokens_after_separator:
                        candidates.append(candidate_token_ids[len(tokens_after_separator):])
            return candidates
        else:
            raise ValueError("Invalid state")

    def extract_schema_from_system_prompt(self, system_prompt: str) -> Dict[str, List[str]]:
        """
        Extract tool schema from a system prompt string.
        Supports two formats:
        1. "1. Name: ToolName\nDescription: ...\nParameters: {...}"
        2. "1. ToolName: Description.\nParameters: {...}" (level-3 format)
        
        Args:
            system_prompt: The system prompt containing tool definitions
            
        Returns:
            Dictionary mapping tool names to their parameter lists
        """
        schema = {}
        
        # Try pattern 1:
        # - "1. Name: ToolName\nDescription: ...\nParameters: {...}"
        # - "1. Name: ToolName\nDescription: Parameters: {...}" (inline Parameters)
        tool_entry_pattern1 = r'\d+\.\s*Name:\s*([^\n]+)\nDescription:[^\n]*?(?:\nParameters:\s*|Parameters:\s*)'
        matches1 = list(re.finditer(tool_entry_pattern1, system_prompt, re.MULTILINE))
        
        # Try pattern 2: "1. ToolName: Description.\nParameters: {...}" (level-3 format)
        # Tool name is before the colon, description is after colon until Parameters
        tool_entry_pattern2 = r'\d+\.\s*([^:]+):\s*[^\n]*\nParameters:\s*'
        matches2 = list(re.finditer(tool_entry_pattern2, system_prompt, re.MULTILINE))
        
        # Use the pattern that matches (prefer pattern 1 if both match)
        if matches1:
            matches = matches1
        elif matches2:
            matches = matches2
        else:
            return schema
        
        for i, match in enumerate(matches):
            tool_name = match.group(1).strip()
            start_pos = match.end()
            
            # Find the end of the Parameters dictionary by matching balanced braces
            if i < len(matches) - 1:
                # Next tool entry starts here
                end_pos = matches[i + 1].start()
                params_str = system_prompt[start_pos:end_pos]
            else:
                # Last tool entry, find the end of the dict or end of available tools section
                params_str = system_prompt[start_pos:]
            
            # Extract just the dictionary part (handle cases where there's text after)
            # Find the first { and match balanced braces
            brace_start = params_str.find('{')
            if brace_start != -1:
                brace_count = 0
                brace_end = brace_start
                for j, char in enumerate(params_str[brace_start:], start=brace_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = j + 1
                            break
                
                params_dict_str = params_str[brace_start:brace_end]
                
                # Extract parameter names from the Parameters dictionary
                param_names = self._extract_param_names(params_dict_str)
                # Always add the tool to schema, even if it has no parameters
                schema[tool_name] = param_names
        
        return schema

    def _extract_param_names(self, params_str: str) -> List[str]:
        """
        Extract parameter names from a Parameters dictionary string.
        
        Args:
            params_str: String representation of parameters dict, e.g., "{'param1': {...}, 'param2': {...}}"
            
        Returns:
            List of parameter names
        """
        param_names = []
        
        try:
            # Try to parse as Python dict literal (handles single quotes)
            # Replace single quotes with double quotes for JSON parsing if needed
            # But ast.literal_eval handles Python dict literals with single quotes
            params_dict = ast.literal_eval(params_str)
            
            if isinstance(params_dict, dict):
                param_names = list(params_dict.keys())
        except (ValueError, SyntaxError):
            # Fallback: use regex to extract keys
            # Pattern: 'key': or "key": followed by {...}
            key_pattern = r"['\"]([^'\"]+)['\"]\s*:"
            param_names = re.findall(key_pattern, params_str)
        
        return param_names

    def get_tool_names(self):
        """Return list of all tool names in the schema."""
        return list(self.schema.keys())

    def get_all_params(self, tool_name: str) -> List[str]:
        """Get all parameters for a given tool."""
        return self.schema.get(tool_name, [])

    def convert_schema_values_to_token_ids(self) -> Dict[tuple, List[List[int]]]:
        """
        Convert all tool names and parameter names in the schema to token IDs.
        
        Returns:
            Dictionary mapping tool name token ID tuples to lists of parameter token ID lists.
            Each tool name maps to a list where each element is a list of token IDs for a parameter.
            Returns empty dict if tokenizer is not available.
        """
        if self.tokenizer is None:
            return {}
        
        tokenized_schema = {}
        for tool_name, param_names in self.schema.items():
            # Convert tool name to all token IDs
            tool_token_ids = self._encode(tool_name)
            if not tool_token_ids:
                continue
            # Convert to tuple for use as dictionary key (lists are not hashable)
            tool_token_id_tuple = tuple(tool_token_ids)
            
            # Convert parameter names to token IDs (all tokens for each parameter)
            param_token_ids = []
            for param_name in param_names:
                # Encode parameter name to get all token IDs
                token_ids = self._encode(param_name)
                if token_ids:
                    param_token_ids.append(token_ids)
            tokenized_schema[tool_token_id_tuple] = param_token_ids
        
        return tokenized_schema

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def extract_schema_from_query(query_data: Dict) -> Dict[str, List[str]]:
        """
        Extract tool schema from a single query data dictionary.
        
        Args:
            query_data: Dictionary with 'system' key containing the system prompt
            
        Returns:
            Dictionary mapping tool names to parameter lists
        """
        if 'system' not in query_data:
            return {}
        
        schema_fsm = SchemaFSM(query_data['system'])
        return schema_fsm.schema

    def __repr__(self):
        return f"SchemaFSM(schema={self.schema})"






