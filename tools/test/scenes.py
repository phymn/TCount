import json, random
ok = json.load(open('testassets/ok.json'))
S = [f'/testassets/sprites/p{i}.png' for i in ok['ok'] if i != 0]
PAIR = [f'/testassets/sprites/p{ok["pair"][0]}.png', f'/testassets/sprites/p{ok["pair"][1]}.png']
BG = [f'/testassets/bg/bg{i}.jpg' for i in ok['bg']]
others = [s for s in S if s not in PAIR]
random.seed(3)
L3 = [{'id': 'cam-uw', 'label': 'Back Ultra Wide Camera', 'zoom': 0.6}, {'id': 'cam-w', 'label': 'Back Camera', 'zoom': 1}, {'id': 'cam-t', 'label': 'Back Telephoto Camera', 'zoom': 2.5}]
def P(sprite, h, path, **k): return dict(sprite=sprite, h=h, path=[dict(t=a, x=b, y=c) for a, b, c in path], **k)
sc = {}
sc['walk3'] = dict(gt=3, dur=22, bg=BG[0], people=[
  P(others[0], 330, [(0, 150, 640), (20, 1100, 650)]),
  P(others[1], 310, [(0, 1150, 620), (20, 200, 630)]),
  P(others[2], 300, [(0, 600, 600), (10, 750, 600), (20, 550, 610)])])
sc['reenter'] = dict(gt=4, dur=27, bg=BG[1], people=[
  P(others[3], 320, [(0, 300, 650), (8, 1450, 650), (14, -150, 640), (22, 500, 640), (27, 520, 640)]),
  P(others[4], 300, [(0, 900, 620), (4, 900, 620), (9, -150, 620), (15, 1450, 630), (22, 950, 630), (27, 950, 630)]),
  P(others[5], 310, [(0, 620, 640), (27, 700, 640)]),
  P(others[6], 280, [(0, 1100, 600), (27, 1150, 600)])])
sc['similar'] = dict(gt=2, dur=31, bg=BG[2], people=[
  P(PAIR[0], 330, [(0, -100, 650), (9, 1400, 650), (20, 1400, 650), (29, -150, 650)]),
  P(PAIR[1], 330, [(10, -100, 650), (19, 1400, 650)])])
sc['similar_together'] = dict(gt=2, dur=20, bg=BG[2], people=[
  P(PAIR[0], 330, [(0, 300, 650), (18, 900, 650)]), P(PAIR[1], 330, [(0, 1000, 650), (18, 450, 650)])])
pan_people = [P(others[7 + i], 300 + (i % 3) * 20, [(0, 350 + i * 500, 640 + (i % 2) * 20), (30, 400 + i * 500, 640 + (i % 2) * 20)]) for i in range(6)]
sc['pan'] = dict(gt=6, dur=31, bg=BG[3], worldW=3200, worldH=720, camera=[dict(t=0, x=640, y=360), dict(t=14, x=2560, y=360), dict(t=28, x=640, y=360), dict(t=31, x=640, y=360)], people=pan_people)
# dense synthetic: rows with perspective
dp = []; k = 0
for row in range(7):
    y = 260 + row * 75; h = 70 + row * 32; n = 6 + (6 - row)
    for c in range(n):
        x = 60 + c * (1160 / max(1, n - 1)) + random.uniform(-25, 25)
        sp = S[(k * 7) % len(S)]; k += 1
        dp.append(P(sp, h, [(0, x, y + h * 0.9), (20, x + random.uniform(-8, 8), y + h * 0.9)]))
sc['dense'] = dict(gt=len(dp), dur=22, bg=BG[4], people=dp)
sc['dense_photo'] = dict(gt=None, dur=14, bg='/testassets/crowd2.png', people=[])
sc['dark'] = dict(sc['walk3'], brightness=0.13, noise=0.06)
sc['empty'] = dict(gt=0, dur=20, bg=BG[5], people=[], noise=0.03, lenses=L3)
sc['far'] = dict(gt=2, dur=34, bg=BG[6], lenses=L3, people=[
  P(others[13], 55, [(0, 500, 420), (34, 700, 420)]), P(others[14], 50, [(0, 760, 410), (34, 650, 410)])])
for k, v in sc.items():
    v.setdefault('W', 1280); v.setdefault('H', 720)
    v['preload'] = [v['bg']] + [p['sprite'] for p in v['people']]
json.dump(sc, open('scenes.json', 'w'))
print({k: (v['gt'], len(v['people'])) for k, v in sc.items()})
