import sys, torch
sys.path.insert(0,'DM-Count')
import models
m=models.vgg19(); m.load_state_dict(torch.load('DM-Count/pretrained_models/model_qnrf.pth',map_location='cpu')); m.eval()
class W(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,x): return s.m(x)[0]
torch.onnx.export(W(m),torch.randn(1,3,384,512),'dmcount_qnrf.onnx',input_names=['input'],output_names=['density'],opset_version=13,dynamic_axes={'input':{2:'h',3:'w'},'density':{2:'h8',3:'w8'}},dynamo=False)
