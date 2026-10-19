# weight-only int8 (per-output-channel symmetric) via DequantizeLinear; compute stays fp32
import onnx, numpy as np, sys
from onnx import numpy_helper, helper, TensorProto
def wq(src, dst, person_only=False, min_elems=1024):
    m = onnx.load(src); g = m.graph
    inits = {i.name: i for i in g.initializer}
    conv_w = set(n.input[1] for n in g.node if n.op_type in ('Conv','Gemm','MatMul') and len(n.input)>1 and n.input[1] in inits)
    new_nodes = []; removed = []
    for name in conv_w:
        W = numpy_helper.to_array(inits[name]).astype(np.float32)
        if W.size < min_elems: continue
        if W.ndim >= 2:
            ax = 0
            mx = np.abs(W.reshape(W.shape[0], -1)).max(1); scale = np.maximum(mx, 1e-8)/127.0
            q = np.clip(np.round(W / scale.reshape([-1]+[1]*(W.ndim-1))), -127, 127).astype(np.int8)
        else: continue
        removed.append(inits[name])
        g.initializer.extend([numpy_helper.from_array(q, name+'_q'), numpy_helper.from_array(scale.astype(np.float32), name+'_s'),
                              numpy_helper.from_array(np.zeros(W.shape[0], np.int8), name+'_z')])
        new_nodes.append(helper.make_node('DequantizeLinear', [name+'_q', name+'_s', name+'_z'], [name], axis=ax, name=name+'_dq'))
    for r in removed: g.initializer.remove(r)
    nodes = list(g.node); del g.node[:]; g.node.extend(new_nodes + nodes)
    if person_only:
        out = g.output[0]; old = out.name; out.name = old+'_full'
        for n in g.node:
            for i,o in enumerate(n.output):
                if o == old: n.output[i] = old+'_full'
        g.initializer.extend([numpy_helper.from_array(np.array([0],np.int64),'ps_st'), numpy_helper.from_array(np.array([6],np.int64),'ps_en'), numpy_helper.from_array(np.array([2],np.int64),'ps_ax')])
        g.node.append(helper.make_node('Slice',[old+'_full','ps_st','ps_en','ps_ax'],[old]))
        g.output.remove(out)
        g.output.append(helper.make_tensor_value_info(old, TensorProto.FLOAT, [1,'n',6]))
    if m.opset_import[0].version < 13: m.opset_import[0].version = 13
    onnx.checker.check_model(m); onnx.save(m, dst); print(dst, len(removed), 'weights quantized')
if __name__ == '__main__':
    wq(sys.argv[1], sys.argv[2], person_only=len(sys.argv)>3 and sys.argv[3]=='person')
