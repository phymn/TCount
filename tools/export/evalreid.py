import glob, os, random, numpy as np, cv2, onnxruntime as ort, sys, time
random.seed(0)
R='market/Market-1501-v15.09.15/' if os.path.isdir('market/Market-1501-v15.09.15') else 'market/'
def pid(f): b=os.path.basename(f); return int(b.split('_')[0]), int(b.split('_')[1][1])
Q=sorted(glob.glob(R+'query/*.jpg')); G=[g for g in sorted(glob.glob(R+'bounding_box_test/*.jpg')) if pid(g)[0]>0]
ids=sorted(set(pid(q)[0] for q in Q)); sel=set(random.sample(ids,250))
Q=[q for q in Q if pid(q)[0] in sel]; Q=[ [q for q in Q if pid(q)[0]==i][0] for i in sorted(sel)]
Gs=[g for g in G if pid(g)[0] in sel]; Gd=random.sample([g for g in G if pid(g)[0] not in sel],1500); G=Gs+Gd
mean=np.array([0.485,0.456,0.406],np.float32); std=np.array([0.229,0.224,0.225],np.float32)
def prep(f):
    im=cv2.imread(f)[:,:,::-1]; im=cv2.resize(im,(128,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
    return ((im-mean)/std).transpose(2,0,1)
X={f:prep(f) for f in Q+G}
for model in sys.argv[1:]:
    so=ort.SessionOptions(); so.intra_op_num_threads=1; s=ort.InferenceSession(model,so)
    def emb(fs):
        out=[]
        for i in range(0,len(fs),32): out.append(s.run(None,{'input':np.stack([X[f] for f in fs[i:i+32]])})[0])
        e=np.concatenate(out); return e/np.linalg.norm(e,axis=1,keepdims=True)
    s.run(None,{'input':X[Q[0]][None]}); t0=time.time(); [s.run(None,{'input':X[Q[0]][None]}) for _ in range(10)]; ms=(time.time()-t0)*100
    eq=emb(Q); eg=emb(G); S=eq@eg.T
    qp=[pid(q) for q in Q]; gp=[pid(g) for g in G]
    r1=0; pos=[]; neg=[]; maxneg=[]
    for i,(qi,qc) in enumerate(qp):
        valid=[j for j,(gi,gc) in enumerate(gp) if not (gi==qi and gc==qc)]
        best=max(valid,key=lambda j:S[i,j]); r1+= gp[best][0]==qi
        pos += [S[i,j] for j in valid if gp[j][0]==qi]; nn=[S[i,j] for j in valid if gp[j][0]!=qi]; neg+=nn; maxneg.append(max(nn))
    pos=np.array(pos); neg=np.array(neg)
    print(f'{model:30s} {ms:.1f}ms/crop rank1={r1/len(Q):.3f} pos med={np.median(pos):.3f} p10={np.percentile(pos,10):.3f} | neg p99={np.percentile(neg,99):.3f} p99.9={np.percentile(neg,99.9):.3f} | maxImpostor/query med={np.median(maxneg):.3f}', flush=True)
    for th in (0.5,0.6,0.65,0.7,0.75,0.8,0.85):
        print(f'   th={th}: TPR={np.mean(pos>th):.3f} FPR={np.mean(neg>th):.5f}', flush=True)
