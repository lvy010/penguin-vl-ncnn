import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M=".\\Penguin-VL-2B"
tok=AutoTokenizer.from_pretrained(M, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(M, torch_dtype="auto", trust_remote_code=True).float().eval()
# EXACT C++ template incl. the empty <think> block (add_think=False default)
ptxt="<|im_start|>user\n请用一句话介绍你自己<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids=tok(ptxt, add_special_tokens=False, return_tensors="pt").input_ids
print("prompt tokens:", ids.shape[1], flush=True)
eos=151645
gen=[]
with torch.no_grad():
    emb=model.model.embed_tokens(ids)
    for step in range(24):
        hid=model.model(inputs_embeds=emb).last_hidden_state
        nxt=int(model.lm_head(hid[:,-1]).argmax(-1))
        gen.append(nxt)
        if nxt==eos: break
        emb=torch.cat([emb,model.model.embed_tokens(torch.tensor([[nxt]]))],dim=1)
import json
print("torch greedy ids:", gen, flush=True)
open("torch_greedy_ids.json","w").write(json.dumps(gen))
open("torch_greedy_text.txt","w",encoding="utf-8").write(tok.decode(gen))
print("torch greedy text written to torch_greedy_text.txt", flush=True)
