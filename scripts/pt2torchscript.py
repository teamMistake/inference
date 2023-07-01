import torch
from jamo import JAMO

model = JAMO.from_pretrained("small", "/Volumes/T7/jamo/jamo_2b_finetuned_6_30/2023-06-30T13:23:54-iter-1950.tar", "cpu")
model.eval()

example = torch.randint(0, 20, (1,256))
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")