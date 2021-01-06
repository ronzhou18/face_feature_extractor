#convert_jit.py
import torch

from modelInsightFace import Backbone

model_path="./checkpoint/face_step3_best.pth"
model_dict=torch.load(model_path,map_location="cpu")
model = Backbone(50, 0.4, "ir")
model.load_state_dict(model_dict)
torch.jit.trace(model,torch.randn(4,3,112,112)).save("jit_model")
