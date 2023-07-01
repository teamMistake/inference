import torch.onnx 
from dataclasses import dataclass
import onnxruntime
import numpy as np

from jamo import JAMO

model = JAMO.from_pretrained("small", "/Volumes/T7/jamo/jamo_2b_finetuned_7_1/2023-06-30T23:16:55-iter-2000.tar", "cpu")
model.eval()

dummy_input = torch.zeros(1, 256, requires_grad=True).type(torch.LongTensor)  
input_pos = torch.arange(0, 256)

# Export the model   
torch.onnx.export(model,   
        (dummy_input, 256, input_pos),       
        "./model.onnx",    
        input_names = ['modelInput'],  
        output_names = ['modelOutput'],
        dynamic_axes={"modelInput": {0: "batch", 1: "sequence"}},
        ) 
        
print('Model has been converted to ONNX') 
