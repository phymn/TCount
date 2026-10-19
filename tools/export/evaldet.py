import json, numpy as np, cv2, onnxruntime as ort, time, sys, contextlib, io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
def decode(out, H, W):
    grids=[];strides=[]
    for s in (8,16,32):
        h,w=H//s,W//s; yv,xv=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
        grids.append(np.stack([xv,yv],-1).reshape(-1,2)); strides.append(np.full((h*w,1),s))
    g=np.concatenate(grids); st=np.concatenate(strides)
    xy=(out[:,:2]+g)*st; wh=np.exp(out[:,2:4])*st
    return np.concatenate([xy,wh],1), out[:,4]*out[:,5]
def nms(b,s,th=0.45):
    x1=b[:,0]-b[:,2]/2;y1=b[:,1]-b[:,3]/2;x2=x1+b[:,2];y2=y1+b[:,3];o=s.argsort()[::-1];keep=[]
    while o.size:
        i=o[0];keep.append(i)
        xx1=np.maximum(x1[i],x1[o[1:]]);yy1=np.maximum(y1[i],y1[o[1:]]);xx2=np.minimum(x2[i],x2[o[1:]]);yy2=np.minimum(y2[i],y2[o[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1);iou=inter/(b[i,2]*b[i,3]+b[o[1:],2]*b[o[1:],3]-inter)
        o=o[1:][iou<th]
    return keep
coco=COCO('instances_val2017.json') if False else None
with contextlib.redirect_stdout(io.StringIO()): coco=COCO('instances_val2017.json')
sel=json.load(open('coco/sel.json'))['sel']
def run(model, S):
    so=ort.SessionOptions(); so.intra_op_num_threads=1
    sess=ort.InferenceSession(model,so); res=[]; t=0
    for iid in sel:
        info=coco.imgs[iid]; img=cv2.imread('coco/'+info['file_name'])
        h,w=img.shape[:2]; r=min(S/h,S/w); nh,nw=int(h*r),int(w*r)
        pad=np.full((S,S,3),114,np.uint8); pad[:nh,:nw]=cv2.resize(img,(nw,nh))
        x=pad.transpose(2,0,1)[None].astype(np.float32)
        t0=time.time(); out=sess.run(None,{'images':x})[0][0]; t+=time.time()-t0
        out=out[:,:6] if out.shape[1]==6 else np.concatenate([out[:,:5],out[:,5:6]],1)
        b,s=decode(out,S,S); m=s>0.01; b,s=b[m],s[m]
        for i in nms(b,s):
            cx,cy,bw,bh=b[i]/r; res.append({'image_id':iid,'category_id':1,'bbox':[float(cx-bw/2),float(cy-bh/2),float(bw),float(bh)],'score':float(s[i])})
    with contextlib.redirect_stdout(io.StringIO()):
        dt=coco.loadRes(res); E=COCOeval(coco,dt,'bbox'); E.params.imgIds=sel; E.params.catIds=[1]; E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0],E.stats[1],E.stats[3],E.stats[8], t/len(sel)*1000
for model in sys.argv[2:]:
    for S in map(int,sys.argv[1].split(',')):
        ap,ap50,aps,ar,ms=run(model,S); print(f'{model:32s} {S}: AP={ap:.3f} AP50={ap50:.3f} APsmall={aps:.3f} AR100={ar:.3f} {ms:.0f}ms/img(1thr)',flush=True)
