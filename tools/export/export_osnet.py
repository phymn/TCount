import sys, torch, importlib.util
def load(path, mod):
    spec=importlib.util.spec_from_file_location(mod,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
osn=load('deep-person-reid/torchreid/models/osnet.py','osn'); ain=load('deep-person-reid/torchreid/models/osnet_ain.py','ain')
cfg=[('osnet_x0_25_msmt17',osn.osnet_x0_25),('osnet_x0_25_line76',osn.osnet_x0_25),('osnet_x0_5_msmt17',osn.osnet_x0_5),('osnet_x0_5_ca',osn.osnet_x0_5),('osnet_ain_x1_0_ca',ain.osnet_ain_x1_0)]
for name,fn in cfg:
    sd=torch.load(name+'.pth',map_location='cpu'); sd=sd.get('state_dict',sd)
    sd={k.replace('module.',''):v for k,v in sd.items()}
    nc=sd['classifier.weight'].shape[0]
    m=fn(num_classes=nc,pretrained=False,loss='softmax'); print(name, m.load_state_dict(sd,strict=False)); m.eval()
    torch.onnx.export(m,torch.randn(1,3,256,128),name+'.onnx',input_names=['input'],output_names=['feat'],opset_version=13,dynamic_axes={'input':{0:'b'},'feat':{0:'b'}},dynamo=False)
