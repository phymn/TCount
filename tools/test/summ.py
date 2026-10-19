import sys,json
for l in sys.stdin:
    try: r=json.loads(l)
    except: print(l.strip()); continue
    print(r['scene'],r['browser'],r['params'],'gt',r['gt'],'U',r['unique'],'pd',r['possible'],'uv',r['unverified'],r['det'],r['ep'],r['size'],'fps',r['fps'],'tiles',r['tiles_any'],'dark',r['dark_any'],'dense',r['dense_any'],'est',r['est_last'] and round(r['est_last'],1),r['est_src'],'peak',r['peak'] and round(r['peak'],1),'dms',r['denseMs'] and round(r['denseMs']),r['denseEp'],'lens',r['lens_final'],'sw',r['lensSw'],'maxIn',r['maxInFrame'],'final',r['final'][0]); print('   ',r['events'][-10:], r['errs'], r['logs'][:3], r['banners'])
