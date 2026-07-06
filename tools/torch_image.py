import sys, types, torch
# --- shim: eager replacement for flash_attn_varlen_func (CPU, no flash-attn) ---
def flash_attn_varlen_func(q,k,v,cu_seqlens_q=None,cu_seqlens_k=None,max_seqlen_q=None,max_seqlen_k=None,dropout_p=0.0,causal=False,**kw):
    # q,k,v: (T, H, D)
    out=torch.empty_like(q)
    cu=cu_seqlens_q.tolist()
    scale=1.0/(q.shape[-1]**0.5)
    for i in range(len(cu)-1):
        s,e=cu[i],cu[i+1]
        qi=q[s:e].transpose(0,1).float(); ki=k[s:e].transpose(0,1).float(); vi=v[s:e].transpose(0,1).float()
        sc=(qi@ki.transpose(-1,-2))*scale
        if causal:
            L=e-s; m=torch.triu(torch.full((L,L),float("-inf")),diagonal=1); sc=sc+m
        p=sc.softmax(-1)
        out[s:e]=(p@vi).transpose(0,1).to(out.dtype)
    return out
import importlib.machinery as _m
fa=types.ModuleType("flash_attn"); fa.flash_attn_varlen_func=flash_attn_varlen_func; fa.__spec__=_m.ModuleSpec("flash_attn",None); fa.__version__="2.0.0"
sys.modules["flash_attn"]=fa

from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
M=".\\Penguin-VL-2B"
proc=AutoProcessor.from_pretrained(M, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(M, torch_dtype="auto", trust_remote_code=True).float().eval()
for _n,_mod in list(sys.modules.items()):
    if _mod is not None and _n.endswith("modeling_penguinvl_encoder"):
        setattr(_mod, "flash_attn_varlen_func", flash_attn_varlen_func)
        print("[shim] injected into", _n, flush=True)
img=Image.open(".\\assets\\test.jpg").convert("RGB")
msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"请描述这张图片"}]}]
text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs=proc(text=text, images=[img], return_tensors="pt")
print("input_ids:", inputs["input_ids"].shape, "pixel_values:", inputs["pixel_values"].shape, flush=True)
with torch.no_grad():
    out=model.generate(**inputs, max_new_tokens=40, do_sample=False)
gen=out[0, inputs["input_ids"].shape[1]:]
cap=proc.tokenizer.decode(gen, skip_special_tokens=True)
open("torch_image_out.txt","w",encoding="utf-8").write(cap)
print("=== CAPTION WRITTEN ===", flush=True)


