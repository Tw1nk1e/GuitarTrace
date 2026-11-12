"""Cached body validation: no SAM/BiRefNet inference; exact old SVG paths kept.

uv run --no-sync python scripts/validate_knobs.py
Inspection windows and fixture expectations here are NOT production rules.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from debug_fretboard import ROOT, geometry, load_image, no_ml, save, tr
from gbcv import knobs
from validate_headstock import sample_path

DEST = ROOT/'out/knob-recovery'
NAMES = ('lp_ref_1','fender_1','fender_2','ibanez_rgms8')


def render(root, t, background, factor=1):
    image = cv2.resize(background,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
    for e in root:
        if e.tag.endswith('path'):
            points=sample_path(e.attrib['d'],step=.3)
            closed=e.attrib['d'].endswith('Z')
        elif e.tag.endswith('circle'):
            theta=np.linspace(0,2*np.pi,80)
            points=np.c_[float(e.attrib['cx'])+float(e.attrib['r'])*np.cos(theta),float(e.attrib['cy'])+float(e.attrib['r'])*np.sin(theta)]
            closed=True
        else:
            continue
        p=np.round(t.ruler.to_px(points)*factor*256).astype(np.int32)
        cv2.polylines(image,[p],closed,(230,30,30),max(1,factor),cv2.LINE_AA,shift=8)
    return image


def main():
    manifest=json.loads((ROOT/'out/evaluation/manifest.json').read_text())
    all_metrics={}
    cards=[]
    with patch('gbcv.segment.probability',no_ml), patch('gbcv.sam.Session',no_ml):
        for name in NAMES:
            folder=DEST/name
            folder.mkdir(parents=True,exist_ok=True)
            record=next(r for r in manifest['records'] if r['id']==name)
            source=ROOT/record['source']
            assert hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256']
            with patch.object(knobs,'recover',return_value=[]):
                t=geometry(load_image(source,mode='RGBA'),cv2.imread(str(ROOT/'out'/name/'mask.png'),0))
            before_text=(ROOT/'out/evaluation'/name/'current.svg').read_text()
            before=ET.fromstring(before_text)
            count=record['cached_body_paths']
            parts=list(before)[-count:]
            t.parts=[sample_path(e.attrib['d'],step=.4) for e in parts]
            assert all(c[:,0].max()<t.nut_x-50 for c in t.parts)
            debug={}
            added=knobs.recover(t,float(t.board[1:3,0].max()),diagnostics=debug)
            after=ET.fromstring(before_text)
            for c in added:
                path=tr.curve_path(tr.smoothed(c,tr.DEFAULT_SMOOTH_MM),tr.PART_CURVE_TOL_MM)
                e=ET.Element('{http://www.w3.org/2000/svg}path',d=path,fill='none',stroke='currentColor',attrib={'stroke-width':str(tr.STYLES['png']['det']),'stroke-linejoin':'round','stroke-linecap':'round'})
                after.append(e)
            assert len(after)==len(before)+len(added)
            assert all(a.tag==b.tag and a.attrib==b.attrib for a,b in zip(before,after))
            (folder/'before.svg').write_text(before_text,encoding='utf-8')
            after_text=ET.tostring(after,encoding='unicode').replace('ns0:','').replace(':ns0','') if added else before_text
            (folder/'after.svg').write_text(after_text,encoding='utf-8')
            photo=tr.masked_photo(t)
            old_overlay,new_overlay=render(before,t,photo),render(after,t,photo)
            save(folder/'before-overlay.png',old_overlay)
            save(folder/'after-overlay.png',new_overlay)
            candidates=debug['candidates']
            rows=[]
            arrays={}
            normalized=np.where(debug['mask'][...,None]>0,debug['photo'],255).astype(np.uint8)
            for i,q in enumerate(candidates):
                row=dict(id=i,centre=list(q.centre),axes=list(q.axes),angle=q.angle,source=q.source,reason=q.reason,**q.metrics)
                rows.append(row)
                if q.contour is not None:
                    arrays[f'candidate_{i}']=q.contour*debug['res']+debug['origin']
                if q.reason in ('added','existing_contour'):
                    color=(0,170,0) if q.reason=='added' else (50,100,240)
                    cv2.polylines(normalized,[np.round(q.contour).astype(np.int32)],True,color,1)
                    cv2.putText(normalized,str(i),tuple(np.round(q.centre).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            (folder/'geometry.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            np.savez_compressed(folder/'geometry.npz',**arrays)
            save(folder/'candidates.png',normalized)
            # Display whole body: same audit crop, then 2x local evidence for every
            # accepted new contour. No cropped-away people/background pixels.
            x0,y0,x1,y1=record['crops']['body']
            save(folder/'body-comparison.png',np.concatenate([photo[y0:y1,x0:x1],old_overlay[y0:y1,x0:x1],new_overlay[y0:y1,x0:x1]],axis=1))
            closeups=[]
            for i,c in enumerate(added):
                px=t.ruler.to_px(c)
                margin=round(4*t.ruler.d)
                left,top=np.maximum(0,np.floor(px.min(0)-margin)).astype(int)
                right,bottom=np.minimum([photo.shape[1],photo.shape[0]],np.ceil(px.max(0)+margin)).astype(int)
                images=[cv2.resize(im[top:bottom,left:right],None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC) for im in (photo,old_overlay,new_overlay)]
                save(folder/f'added-{i}-closeup.png',np.concatenate(images,axis=1))
                closeups.append(f'<img src="{name}/added-{i}-closeup.png">')
            accepted=[r for r in rows if r['reason']=='added']
            all_metrics[name]=dict(added=len(added),accepted=accepted,old_elements_unchanged=True,
                                   diagnostics_candidates=len(rows),ml_inference=False,
                                   svg_byte_identical=before_text==after_text)
            cards.append(f'<section><h2>{name}: +{len(added)}</h2><p>Фото / до / после. Красные линии — реальные кривые SVG, не оценка эталонной точности.</p><img src="{name}/body-comparison.png"><div>{"".join(closeups)}</div><div class="vectors"><img src="{name}/before.svg"><img src="{name}/after.svg"></div><p><a href="{name}/before.svg">SVG до</a> <a href="{name}/after.svg">SVG после</a> <a href="{name}/geometry.json">Все кандидаты и причины</a> <a href="{name}/candidates.png">Принятые ID</a></p></section>')
            print(name,all_metrics[name]['added'],[(r['id'],r['centre']) for r in accepted])
    (DEST/'verification.json').write_text(json.dumps(all_metrics,indent=2),encoding='utf-8')
    experiment=DEST/'local-sam/results.json'
    local=''
    if experiment.exists():
        results=json.loads(experiment.read_text())
        local='<section><h2>Один локальный SAM-эксперимент: верхняя левая ручка</h2><p>Crop 100 условных единиц, положительная подсказка из геометрического кандидата, четыре отрицательные за его краем. Один encoder и один decoder, без повторной настройки.</p><p>Полная маска стала чище, но её оценка 0,342 ниже штатного порога 0,7; варианты около 0,707 выделяют только центр. В SVG не добавлены. Этот эксперимент не используется в production: восстановление выполняется только OpenCV.</p><div class="sam">'+''.join(f'<figure><figcaption>SAM {r["id"]}: {r["score"]:.4f}</figcaption><img src="local-sam/candidate-{r["id"]}.png"></figure>' for r in results)+'</div><a href="local-sam/results.json">Маски, оценки и контуры</a></section>'
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Ручки: локальная геометрия</title><style>body{font:16px sans-serif;margin:24px;line-height:1.5}img{max-width:100%}section{border-top:1px solid #bbb;margin-top:25px}.vectors{display:flex}.vectors img{width:49%}.sam{display:flex}.sam figure{width:31%;margin:8px}</style><h1>Ручки: одна локальная итерация</h1><p><b>LP 1: восстановлена нижняя правая ручка, стало 3 из 4.</b> Цель 4 из 4 не достигнута. У верхней левой ручки худший сектор внешней границы подтверждён лишь на 42%; консервативный детектор её не добавляет. Дальнейшей настройки порогов в этой итерации нет.</p><p>Кандидаты окружностей и эллипсов проверены по внешнему замкнутому Canny-контуру, независимым кольцам в радиальном профиле и контрасту. Hough без подтверждения замкнутым эллипсом отклоняется. Внутренняя граница одного диска не становится отдельной ручкой. Число/раскладка ручек не заданы. Геометрия — измеренные фотограницы, не нарисованная окружность. Все существующие элементы SVG сохранены без изменения; Fender 1, Fender 2 и RGMS8 побайтово неизменны.</p><p>Сравнения построены без глобального ML: актуальная система координат + точные сохранённые SAM-кривые из аудита. Соответствие сохранённых исходников проверено SHA-256. Автоматические контуры не считаются эталоном качества; совпадение восстановленного контура с видимым ободом проверено на увеличении. Проверка ограничена этими четырьмя каталожными фото.</p><p>Остаются: верхняя левая ручка и бридж LP 1, пропуски ручек на других фото, шумные SAM-контуры. Восстановление сглаженных/размытых металлических ручек намеренно не форсируется.</p><p><a href="verification.json">Машинные проверки сохранности SVG</a> · <a href="tests.txt">Полный результат тестов</a></p>'+''.join(cards)+local,encoding='utf-8')


if __name__=='__main__':
    main()
