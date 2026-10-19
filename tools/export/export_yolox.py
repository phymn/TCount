import sys, torch
sys.path.insert(0,'YOLOX')
from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
for name in ['yolox_nano','yolox_tiny']:
    exp = get_exp(f'YOLOX/exps/default/{name}.py', None)
    model = exp.get_model(); model.eval()
    ck = torch.load(f'{name}.pth', map_location='cpu')
    model.load_state_dict(ck['model'])
    model = replace_module(model, torch.nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    x = torch.randn(1,3,416,416)
    torch.onnx.export(model, x, f'{name}_dyn.onnx', input_names=['images'], output_names=['output'],
        opset_version=13, dynamic_axes={'images':{2:'h',3:'w'},'output':{1:'n'}}, dynamo=False)
    print('ok', name)
