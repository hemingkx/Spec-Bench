from .llama import llama_patch_dict, llama_attn_patch_dict

patch_dict = {}
attn_patch_dict = {}

patch_dict.update(llama_patch_dict)
attn_patch_dict.update(llama_attn_patch_dict)
