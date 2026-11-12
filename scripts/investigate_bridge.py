"""One bounded, read-only bridge experiment. No production changes or ML calls.

uv run --no-sync python scripts/investigate_bridge.py

Fixed hypothesis: low-chroma metal + transverse opening to remove strings.
20/25/30 are a stability sweep, NOT a search for the best-looking threshold.
All fixture windows below are inspection aids, never production coordinates.
"""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from validate_knobs import ROOT, geometry, load_image, knobs, no_ml, render, save, tr
from validate_knob_export import preview, text

DEST = ROOT/'out/bridge-recovery'
NAMES = ('lp_ref_1', 'lp_ref_2', 'lp_ref_3', 'ref_sg_1', 'ibanez_rgms8')
RES = .5
ORIGIN = np.array([0., -100.])
LIMITS = (20, 25, 30)


def load_fixture(name):
    with patch('gbcv.segment.probability', no_ml), patch('gbcv.sam.Session', no_ml), \
            patch.object(knobs, 'recover', return_value=[]):
        if name == 'ref_sg_1':
            source = ROOT/'samples/ref_sg_1.jpg'
            cached = cv2.imread(str(ROOT/'out/e2e-lp-sg/ref_sg_1/seg-silhouette.png'), 0)
            with patch.object(tr, 'silhouette', return_value=cached), patch.object(tr, '_sam_parts', return_value=[]):
                t = tr.trace(load_image(source, mode='RGBA'))
            svg_path = ROOT/'out/e2e-lp-sg/ref_sg_1/final.svg'
        else:
            source = next((ROOT/'samples/dl').glob(name+'.*'))
            record = next(r for r in json.loads((ROOT/'out/evaluation/manifest.json').read_text())['records'] if r['id'] == name)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == record['source_sha256']
            t = geometry(load_image(source, mode='RGBA'), cv2.imread(str(ROOT/'out'/name/'mask.png'), 0))
            svg_path = ROOT/'out/evaluation'/name/'current.svg'
            if name == 'lp_ref_1':
                svg_path = ROOT/'out/knob-export/lp_ref_1/production-after.svg'
            elif name == 'ibanez_rgms8':
                svg_path = ROOT/'out/knob-export/ibanez_rgms8/after.svg'
    return t, svg_path.read_text(encoding='utf-8')


def grid(t):
    return tr._rect(t.photo, t.mask, t.ruler, 0, float(t.board[1:3,0].max()), -100, 100, RES)


def metal_contours(photo, mask, limit):
    """Actual segmented boundaries, never rectangles or family templates."""
    lab = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[...,1:]-128, axis=2)
    raw = ((chroma < limit) & (mask > 0)).astype(np.uint8)
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((8,1), np.uint8))
    opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shapes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        # Broad transverse proposals only. Tailpieces and pickup strips may
        # propose here; bilateral string crossings must disambiguate them.
        if w < 60 and 80 < h < 220 and h/w > 3 and y < 200 < y+h:
            shapes.append(c[:,0].astype(float)*RES+ORIGIN)
    return shapes, opened


def stability95(c, groups):
    errors = []
    for group in groups:
        if not group:
            errors.append(None)
            continue
        q = min(group, key=lambda q: np.linalg.norm(q.mean(0)-c.mean(0)))
        errors.append(float(max(np.quantile(cKDTree(q).query(c)[0], .95),
                                np.quantile(cKDTree(c).query(q)[0], .95))))
    return errors


def string_crossings(photo, c):
    """Bright thin ridges that continue on BOTH sides, not merely near metal."""
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    ridges = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9,1), np.uint8))
    lo,hi = c.min(0),c.max(0)
    profiles = []
    for start,end in ((lo[0]-14,lo[0]-2), (hi[0]+2,hi[0]+14)):
        band = ridges[:,max(0,round(start/RES)):min(photo.shape[1],round(end/RES))]
        if band.shape[1] == 0:
            profiles.append(np.array([]))
            continue
        peaks,_ = find_peaks(np.median(band,axis=1), height=12, distance=round(3/RES))
        ys = peaks*RES+ORIGIN[1]
        profiles.append(ys[(ys > lo[1]+5) & (ys < hi[1]-5)])
    left,right = profiles
    matched = [float(y) for y in left if len(right) and min(abs(right-y)) < 1.5]
    return dict(left=left.tolist(), right=right.tolist(), matched=matched,
                span_share=float(np.ptp(matched)/(hi[1]-lo[1])) if len(matched)>1 else 0.)


def boundary_evidence(photo, c):
    lab = cv2.GaussianBlur(cv2.cvtColor(photo,cv2.COLOR_RGB2LAB).astype(np.float32), (0,0), .8)
    gx = cv2.Sobel(lab,cv2.CV_32F,1,0)/8
    gy = cv2.Sobel(lab,cv2.CV_32F,0,1)/8
    gradient = np.sqrt(np.sum(gx*gx+gy*gy,axis=2))
    p = np.round((c-ORIGIN)/RES).astype(int)
    strength = gradient[p[:,1],p[:,0]]
    support = strength >= 5
    lo,hi = c.min(0),c.max(0)
    # Fit long sides only; retain curved endcaps in the actual hypothesis.
    heights = np.arange(round((lo[1]-ORIGIN[1])/RES)+1,round((hi[1]-ORIGIN[1])/RES))
    sides = []
    for y in heights:
        row = p[p[:,1] == y,0]
        if row.size and lo[1]+.15*(hi[1]-lo[1]) < y*RES+ORIGIN[1] < hi[1]-.15*(hi[1]-lo[1]):
            sides.append((y, row.min(), row.max()))
    deviations = []
    if len(sides) >= 8:
        side = np.asarray(sides,float)*RES
        for j in (1,2):
            fit = np.polyfit(side[:,0],side[:,j],1)
            deviations.append(float(np.quantile(abs(side[:,j]-np.polyval(fit,side[:,0])),.9)))
    return dict(edge_support=float(support.mean()),
                weakest_boundary_eighth=float(min(q.mean() for q in np.array_split(support,8))),
                side_line_residual90=deviations)


def rejection_reasons(stability, crossings, boundary):
    reasons = []
    # Fixed conservative evidence checks: no repeated threshold tuning.
    if any(e is None or e > 1.5 for e in stability):
        reasons.append('unstable_colour_boundary')
    if len(crossings['matched']) < 4 or crossings['span_share'] < .45:
        reasons.append('insufficient_bilateral_string_crossings')
    if boundary['edge_support'] < .8 or boundary['weakest_boundary_eighth'] < .5:
        reasons.append('incomplete_photo_edge_support')
    if len(boundary['side_line_residual90']) != 2 or max(boundary['side_line_residual90'],default=999) > 2:
        reasons.append('irregular_long_sides')
    return reasons


def sam_history(t):
    cache = np.load(ROOT/'out/debug-sam/lp_ref_1/before/sam-cache.npz', allow_pickle=False)
    records = json.loads((ROOT/'out/debug-sam/lp_ref_1/before/candidates.json').read_text())
    points = t.ruler.to_mm(cache['points']+cache['crop'][:2])
    indices = np.flatnonzero((points[:,0]>175) & (points[:,0]<210) & (abs(points[:,1])<45))
    rows = [{**r, 'prompt_mm':points[r['prompt']].tolist()} for r in records if r['prompt'] in indices]
    (DEST/'cached-sam-review.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    photo = tr.masked_photo(t)
    cards = []
    for r in sorted((r for r in rows if r.get('reason') != 'area_max'),key=lambda r:-r['score'])[:3]:
        m = cv2.resize(cache['masks'][r['id']],(int(cache['crop'][2]-cache['crop'][0]),int(cache['crop'][3]-cache['crop'][1])),interpolation=cv2.INTER_NEAREST)
        image = photo.copy()
        x0,y0,x1,y1 = cache['crop']
        region = image[y0:y1,x0:x1]
        region[m>0] = (.6*region[m>0]+.4*np.array([30,160,250])).astype(np.uint8)
        left,top = np.floor(t.ruler.to_px([[165,-55]])[0]).astype(int)
        right,bottom = np.ceil(t.ruler.to_px([[225,55]])[0]).astype(int)
        path = f'cached-sam-{r["id"]}.png'
        save(DEST/path,image[top:bottom,left:right])
        cards.append(f'<figure><figcaption>C{r["id"]}: {r["score"]:.3f}, {r["reason"]}</figcaption><img src="{path}"></figure>')
    return ''.join(cards)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    results,cards = {},[]
    history = ''
    for name in NAMES:
        t,baseline = load_fixture(name)
        folder = DEST/name
        folder.mkdir(exist_ok=True)
        photo,mask = grid(t)
        shown = np.where(mask[...,None]>0,photo,255).astype(np.uint8)
        groups = []
        for limit in LIMITS:
            shapes,opened = metal_contours(photo,mask,limit)
            groups.append(shapes)
            overlay = shown.copy()
            for c in shapes:
                cv2.polylines(overlay,[np.round((c-ORIGIN)/RES).astype(np.int32)],True,(255,0,0),1,cv2.LINE_AA)
            save(folder/f'chroma-{limit}.png',cv2.resize(overlay,None,fx=2,fy=2))
            cv2.imwrite(str(folder/f'mask-{limit}.png'),opened*255)
        rows = []
        for c in groups[1]:
            stable = stability95(c,(groups[0],groups[2]))
            crossings = string_crossings(photo,c)
            boundary = boundary_evidence(photo,c)
            reasons = rejection_reasons(stable,crossings,boundary)
            rows.append(dict(bounds=[c.min(0).tolist(),c.max(0).tolist()],width=float(np.ptp(c[:,0])),
                             contour=c.tolist(),stability95=stable,crossings=crossings,boundary=boundary,reasons=reasons))
        # These numerical checks are NOT a bridge classifier. In LP 3, wood
        # texture beside the tailpiece even imitates bilateral string ridges.
        # Passing those checks must never silently add a contour to the SVG.
        root = ET.fromstring(baseline)
        hypothetical = copy.deepcopy(root)
        if name == 'lp_ref_1':
            history = sam_history(t)
            candidate = max(zip(groups[1],rows),key=lambda pair:len(pair[1]['crossings']['matched']))[0]
            ET.SubElement(hypothetical,'{http://www.w3.org/2000/svg}path',
                          d=tr.curve_path(tr.smoothed(candidate,tr.DEFAULT_SMOOTH_MM),tr.PART_CURVE_TOL_MM),
                          fill='none',stroke='currentColor',attrib={'stroke-width':str(tr.STYLES['png']['det'])})
        # Honest before/after: no rejected contour in the final SVG.
        for label in ('before','after'):
            (folder/f'{label}.svg').write_text(baseline,encoding='utf-8')
        assert (folder/'before.svg').read_bytes() == (folder/'after.svg').read_bytes()
        (folder/'rejected-hypothesis.svg').write_text(text(hypothetical),encoding='utf-8')
        # Inspect the bridge neighbourhood, not a fitted/exported rectangle.
        lo,hi = ([155,-60],[230,60]) if name!='ibanez_rgms8' else ([45,-65],[180,65])
        for label,svgroot in (('before',root),('hypothesis',hypothetical),('after',root)):
            zoom = copy.deepcopy(svgroot)
            zoom.set('viewBox',' '.join(map(str,[*lo,*(np.asarray(hi)-lo)])))
            (folder/f'{label}-bridge.svg').write_text(text(zoom),encoding='utf-8')
            save(folder/f'{label}-bridge.png',preview(zoom,800))
            for height in (160,800):
                save(folder/f'{label}-{height}.png',preview(svgroot,height))
        masked = tr.masked_photo(t)
        comparisons = [masked,render(root,t,masked),render(hypothetical,t,masked),render(root,t,masked)]
        left,top = np.maximum(0,np.floor(t.ruler.to_px([lo])[0])).astype(int)
        right,bottom = np.minimum([masked.shape[1],masked.shape[0]],np.ceil(t.ruler.to_px([hi])[0])).astype(int)
        save(folder/'bridge-photo-comparison.png',np.concatenate([cv2.resize(im[top:bottom,left:right],None,fx=2,fy=2) for im in comparisons],axis=1))
        # Directional lines are diagnostics, NOT instructions to draw a box.
        edges = cv2.Canny(cv2.cvtColor(shown,cv2.COLOR_RGB2GRAY),40,100)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength=20,maxLineGap=5)
        directional = shown.copy()
        counts = dict(longitudinal=0,transverse=0)
        if lines is not None:
            for x0,y0,x1,y1 in lines.reshape(-1,4):
                dx,dy = abs(x1-x0),abs(y1-y0)
                kind = 'longitudinal' if dy <= .27*dx else 'transverse' if dx <= .27*dy else None
                if kind:
                    counts[kind] += 1
                    cv2.line(directional,(x0,y0),(x1,y1),(30,100,250) if kind=='longitudinal' else (250,30,30),1)
        save(folder/'directional-edges.png',directional)
        results[name] = dict(final_svg_byte_identical=True,added_parts=0,proposals=rows,directional_lines=counts,
                             numeric_checks_passed=sum(not row['reasons'] for row in rows),
                             inference=False,decision='fallback; no sufficiently supported bridge')
        cards.append(f'<section><h2>{name}</h2><p>Фото / до / отвергнутая гипотеза (только LP 1) / итоговый fallback.</p><img src="{name}/bridge-photo-comparison.png"><p>Красные линии: поперёк гитары; синие: вдоль.</p><img src="{name}/directional-edges.png"><h3>Устойчивость: пороги 20 / 25 / 30</h3>'+''.join(f'<img width="32%" src="{name}/chroma-{limit}.png">' for limit in LIMITS)+f'<h3>Увеличение SVG</h3><div class="pair"><img src="{name}/before-bridge.svg"><img src="{name}/hypothesis-bridge.svg"><img src="{name}/after-bridge.svg"></div>'+''.join(f'<h3>{h} px по высоте: до / гипотеза / итог</h3><div class="pair"><img src="{name}/before-{h}.png"><img src="{name}/hypothesis-{h}.png"><img src="{name}/after-{h}.png"></div>' for h in (160,800))+'</section>')
        print(name,[(row['bounds'],row['stability95'],row['reasons']) for row in rows])
    local = DEST/'local-sam/results.json'
    sam_cards = ''
    if local.exists():
        records = json.loads(local.read_text())
        assert max(r['score'] for r in records) < tr.SAM_MIN_SCORE
        sam_cards = '<h2>Единственный локальный SAM-прогон (кэш)</h2><div class="pair">'+''.join(f'<figure><figcaption>{r["score"]:.3f}, компонент: {r["components"]}; отклонён</figcaption><img src="local-sam/candidate-{r["id"]}.png"></figure>' for r in records)+'</div>'
    (DEST/'verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Бридж: ограниченная итерация</title><style>body{font:16px sans-serif;margin:24px;line-height:1.5}img{max-width:100%}section{border-top:1px solid #aaa;margin-top:30px}.pair{display:flex;gap:10px;overflow:auto}.pair>img,.pair>figure{width:32%;object-fit:contain;align-self:start}figure{margin:0}figcaption{margin:8px}</style><h1>Бридж: гипотеза отклонена, production не изменён</h1><p>Одна фиксированная CV-гипотеза, один локальный SAM-прогон. Прямоугольник и шаблоны не подставлялись. LP 1: узнаваемая гипотеза, но изменение границы до 4 единиц при ширине 16. LP 2: цветовая граница устойчивее, но не подтверждены достаточно широко распределённые струны с обеих сторон. LP 3: маска бриджа сливается с соседями; текстура дерева рядом со струнодержателем создаёт ложные совпадения со струнами, и сам струнодержатель проходит численные проверки — их недостаточно для классификации бриджа. SG: нет устойчивой отдельной границы бриджа. Ibanez: отдельного подходящего поперечного контура нет. Никакие кандидаты не включены в конечные SVG; они побайтово равны исходным.</p><p>PNG — диагностическая растеризация с исходными толщинами. До/после не выдаются за улучшение; средняя колонка явно обозначает отвергнутую гипотезу. Фото — только внутри маски гитары.</p><p><a href="verification.json">Геометрия, метрики и причины</a> · <a href="cached-sam-review.json">Сохранённые SAM-кандидаты</a></p><h2>Сохранённые глобальные SAM-кандидаты</h2><p>Подсказки x=178 и 206 прошли мимо узкого бриджа x≈184–200. Кандидаты с приемлемым score захватывают корпус; остальные ниже порога.</p><div class="pair">'+history+'</div>'+sam_cards+''.join(cards),encoding='utf-8')


if __name__ == '__main__':
    main()
