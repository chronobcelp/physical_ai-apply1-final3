"""Create the *measured* 120-second narrated Japanese comparison video.

Both baseline and AI are recorded on Episode 1 (same layout and seed).
ALL robot images come from real MuJoCo evaluation frames. No simulated
metrics or outcomes are substituted. The three-finger claw close-up is
captured by a second real MuJoCo camera.

Speech: Google gTTS (requires Colab internet); cached on retry. Alternatively
put a user-recorded WAV at results_bento3d/narration_override.wav.
"""
from __future__ import annotations
from pathlib import Path
import json, shutil, subprocess, hashlib
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import soundfile as sf

FPS = 12
DURATION = 120
W, H = 960, 540
SR = 24000
LABELS = ['おにぎり','卵焼き','ブロッコリー','唐揚げ','トマト']
ORDER_SLOTS=(0,1,2,3,4,5)
RECIPE=(0,1,2,3,0,1)
TIMELINE = [
    (0, 10, '画像で位置とサイズを判断する、六マスのお弁当ロボットです。'),
    (10, 38, '左は固定順、右はカメラから中心位置と種類と大きさを認識します。特大品は回収し、普通サイズを盛り付けます。'),
    (48, 40, '三本爪を開いて画像で見つけた食品の上へ移動します。下降してつかみ、指定区画へ運びます。サイズが合わない食品は回収し、適切なものを盛り付けます。'),
    (88, 13, '六つの区画と、欠品した食品を確認します。'),
    (101, 12, '実測した作業成功率と、サイズ判定結果を比較します。'),
    (113, 7, '食品の把持には簡易モデルを使用しています。'),
]
# 各セクションで収まりきらない場合は、等速のまま短い原稿へ段階的に変更。
# 採用された原稿をTIMELINEに書き戻して字幕と一致させる。
VOICE_ALTERNATIVES = {
    0: ['六マスのAIお弁当ロボットです。', 'AIお弁当ロボットです。'],
    1: ['左は固定順、右は種類とサイズで六マスに配置します。', '六マスの従来方式とAI方式を比較します。'],
    2: ['三本爪でつかみ、指定区画へ。規格外品は回収します。', '三本爪で食品を運び、規格外品は回収します。'],
    3: ['六区画の結果を比較します。', '欠品を確認します。'],
    4: ['実測の作業成功率を比較します。', '実測結果です。'],
    5: ['把持は簡易モデルです。', '簡易把持モデルです。'],
}

FONT_PATHS = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
    '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]

@lru_cache(maxsize=32)
def font(size, bold=False):
    paths=([p.replace('Regular','Bold') for p in FONT_PATHS[:2]]+FONT_PATHS) if bold else FONT_PATHS
    for f in paths:
        if Path(f).exists():return ImageFont.truetype(f,size)
    raise RuntimeError('日本語フォントがありません。Colabで !apt-get -qq install -y fonts-noto-cjk を実行してください。')

def label_name(cls):
    if cls is None or int(cls)<0:return '不明'
    return LABELS[int(cls)]

def destination(event):
    if not event:return '---'
    slot=int(event['slot'])
    return '回収トレー' if slot<0 else f'第{slot+1}区画({label_name(RECIPE[slot])})'

def get_event(events, pos):
    if not events:return None
    for e in events:
        if int(e['from_frame'])<=pos<int(e['to_frame']):return e
    return events[-1] if pos>=int(events[-1]['from_frame']) else events[0]

def load_frame(paths, i, size):
    if not paths:raise RuntimeError('実際のMuJoCo録画フレームがありません。評価・録画を先に完了してください。')
    src=cv2.imread(str(paths[min(len(paths)-1, max(0, i))]))
    if src is None:raise RuntimeError(f'読めない録画フレーム: {paths[i]}')
    src=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    # This is a pure rescaling and small display-only contrast adjustment.
    # No black strokes or object errors are painted over or removed.
    image=Image.fromarray(src).resize(size,Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(image).enhance(1.035)

def frame_index(t, start, span, length):
    return max(0,min(length-1,int((t-start)/span*length)))

def box(draw, xy, color=(15,28,41), outline=None, r=12):
    draw.rounded_rectangle(xy,radius=r,fill=color,outline=outline,width=2 if outline else 1)

def write(draw, xy, text, size, fill=(246,250,251), bold=False):
    draw.text(xy,text,font=font(size,bold),fill=fill,stroke_width=0)

def event_copy(e, mode):
    if e is None:return ['判断待ち','---','---']
    who='従来方式' if mode=='baseline' else 'AI方式'
    t=label_name(e['true_class'])
    if mode=='baseline':
        decision=f'食品認識なし → {destination(e)}'
    else:
        decision=f'CNN: {label_name(e["prediction"])} → {destination(e)}'
    verdict = '対象外' if int(e['true_class'])==4 else ('判定一致' if int(e.get('prediction',-1))==int(e['true_class']) else '認識誤り')
    if mode=='baseline':verdict='画像を見ずに固定順で配置'
    else:
        conf=float(e.get('confidence',0))
        ss=e.get('size_est_x'); size_label='測定不可' if ss is None else f'{float(ss):.2f}倍'
        verdict=f'RGB-Dサイズ {size_label} / CNN {conf:.0%}' + (' / 要確認' if conf < .60 else '')
        if int(e.get('prediction',-1)) != int(e['true_class']):verdict+=' / 誤認識'
    outcome=e.get('correct',None)
    if outcome is not None:
        verdict+=(' | 作業OK' if int(outcome) else ' | 作業NG')
    img_xy=e.get('image_xy')
    if mode=='ai' and img_xy is not None:
        verdict=f'画像位置({img_xy[0]:.2f},{img_xy[1]:.2f})m / ' + verdict
    actual_size=e.get('size_true_x')
    actual_info=f' | 実サイズ {actual_size:.2f}倍' if actual_size is not None else ''
    return [f'{who} | 実物: {t}'+actual_info,decision,verdict]

def plain_canvas():return Image.new('RGB',(W,H),(12,22,36))

PHASE_JA={'OPEN':'爪を開く','APPROACH':'食品の真上へ移動','LOWER':'下降','CLOSE':'爪を閉じる',
          'LIFT':'持ち上げる','TRANSPORT':'運搬','PLACE_LOWER':'箱へ下降','RELEASE':'爪を開いて置く',
          'RETRACT':'上昇','READY':'次の食品を判断','IDLE':'待機'}

def phase_label(phases, idx):
    return PHASE_JA.get(phases[min(max(0,idx),len(phases)-1)],'移動中') if phases else '動作情報なし'

def draw_compare(t, baseline, ai, be, ae, pb=None, pa=None):
    res=plain_canvas();dr=ImageDraw.Draw(res)
    write(dr,(27,10),'6マスで比較：欠品＋サイズ違い＋順番入替',27,bold=True)
    write(dr,(27,51),'左：固定順  /  右：画像中心座標＋CNN種類＋RGB-Dサイズ測定',17,fill=(160,192,207))
    parts=[(baseline,be,pb,'従来方式｜固定順','baseline',(240,174,109),22),
           (ai,ae,pa,'AI方式｜CNN＋注文照合','ai',(92,217,174),490)]
    for frames, events, phases, title,mode,accent,x in parts:
        box(dr,(x,82,x+448,404),(24,36,51),accent,12)
        write(dr,(x+10,88),title,20,accent,bold=True)
        idx=frame_index(t,10,38,len(frames))
        pic=load_frame(frames,idx,(424,276))
        res.paste(pic,(x+12,121))
        ev=get_event(events,idx)
        ls=event_copy(ev,mode)
        box(dr,(x,412,x+448,526),(25,40,54),None,12)
        write(dr,(x+12,421),ls[0][:32],16,bold=True)
        write(dr,(x+12,448),ls[1][:29],16,fill=accent)
        write(dr,(x+12,474),ls[2][:43],13,fill=(229,170,141) if mode=='baseline' else (165,205,210))
        write(dr,(x+12,501),'● '+phase_label(phases,idx),14,fill=(252,229,166),bold=True)
    # Clear visual attribution: baseline and AI share the same robot and same starting scene.
    return res

def draw_close(t, frames, zoom, events, phases=None):
    res=plain_canvas();dr=ImageDraw.Draw(res)
    write(dr,(24,11),'AI方式｜3本爪アームの拡大映像',29,bold=True)
    write(dr,(24,53),'画像中心座標 → CNN食品分類とRGB-Dサイズ → 爪で把持',17,fill=(164,204,220))
    idx=frame_index(t,48,40,len(frames))
    # Larger clear camera view with a thin neutral border.
    pic=load_frame(frames,idx,(634,420))
    res.paste(pic,(16,92))
    dr.rectangle((15,91,651,513), outline=(109,165,183), width=2)
    box(dr,(666,90,949,513),(29,45,60),(82,208,175))
    if zoom:
        cidx=frame_index(t,48,40,len(zoom))
        close=load_frame(zoom,cidx,(267,213))
        res.paste(close,(674,131))
        dr.rectangle((673,130,942,345),outline=(99,225,188),width=2)
        write(dr,(681,101),'実際の3D近接カメラ',18,(112,235,202),bold=True)
    else:
        write(dr,(681,101),'メインカメラ映像',18,(112,235,202))
    e=get_event(events,idx)
    lines=event_copy(e,'ai')
    write(dr,(676,365),lines[0][:20],15,bold=True)
    write(dr,(676,395),lines[1][:15],16,(131,231,202))
    write(dr,(676,428),lines[2][:22],14,(174,214,215))
    write(dr,(676,462),'● '+phase_label(phases,idx),18,(255,219,151),bold=True)
    write(dr,(676,489),'OPEN → CLOSE → LIFT',12,(175,208,221))
    return res

def draw_missing(base, ai, summary, audits=None):
    res=plain_canvas();dr=ImageDraw.Draw(res)
    write(dr,(28,14),'6マス弁当：ブロッコリー欠品をどう扱う？',30,bold=True)
    for img,x,title,accent in [(base,20,'従来方式：固定区画へ順番に入れる',(239,176,118)),
                               (ai,489,'AI方式：種類に応じて配置する',(92,219,178))]:
        box(dr,(x,74,x+451,407),(29,45,59),accent)
        write(dr,(x+10,82),title,18,accent,bold=True)
        res.paste(load_frame(img, len(img)-1,(433,283)),(x+9,115))
    box(dr,(20,422,940,525),(34,52,70))
    if audits and all(k in audits for k in ('baseline','ai')):
        # Two 2-row x 3-column status panels; slot IDs remain independent of classes.
        short={j:label_name(RECIPE[j]) for j in ORDER_SLOTS}
        result={'CORRECT':'正解','EMPTY':'空き','WRONG_OR_OVERFULL':'誤配置','OVERSIZE':'サイズ超過'}
        write(dr,(36,429),'完成品の区画監査（MuJoCo真値）',17,(226,239,243),bold=True)
        for mode,xx,color in (('baseline',38,(243,185,133)),('ai',502,(104,235,190))):
            audit={int(s['compartment']):s['state'] for s in audits[mode]}
            for j in ORDER_SLOTS:
                x=xx+(j%3)*146
                y=455+(j//3)*25
                write(dr,(x,y),f'{j+1} {short[j]}:{result.get(audit.get(j),"未確認")}',12,color)
        write(dr,(36,502),'※ この監査は画像AIによる再認識ではありません。',14,(196,213,224))
    else:
        write(dr,(35,433),'注文：おにぎり2・卵焼き2・ブロッコリー1・唐揚げ1',21,bold=True)
        write(dr,(35,465),'ブロッコリーは元から存在しません。CNN予測と作業結果を表示。',18,(244,207,145))
    return res

def draw_metrics(summary):
    b=summary['baseline'];a=summary['ai'];res=plain_canvas();d=ImageDraw.Draw(res)
    write(d,(45,25),'実測結果：同じ初期条件で比較',34,bold=True)
    write(d,(48,89),'項目別の成功率（評価後の summary.json を表示）',22,(172,206,222))
    rows=[('6区画の最終配置 正解率',b['compartment_success_rate'],a['compartment_success_rate']),
          ('食品の扱い 成功率',b['item_success_rate'],a['item_success_rate'])]
    for i,(label,br,ar) in enumerate(rows):
        yy=151+i*105
        write(d,(42,yy),label,23)
        write(d,(430,yy),f'従来 {br*100:.1f}%',24,(244,182,123),bold=True)
        write(d,(690,yy),f'AI {ar*100:.1f}%',24,(115,236,187),bold=True)
    ab=summary.get('ai_no_size',{})
    if ab:
        write(d,(43,351),f'サイズ判定なしCNN：作業成功率 {ab["item_success_rate"]*100:.1f}%',21,(232,187,123))
    acc=a.get('classification_accuracy')
    accuracy='未評価' if acc is None else f'{acc*100:.1f}%'
    oracle=summary.get('oracle',{})
    oracle_text=f"参考：完全識別上限 {oracle['item_success_rate']*100:.1f}%" if oracle else '参考：理想識別上限は未測定'
    write(d,(43,407),f'食品CNN分類正解率：{accuracy}  ｜  評価シーン：{summary["scenes"]}',19,(170,216,221))
    write(d,(43,436),oracle_text+'（AIではない）',16,(174,218,206))
    sz=a.get('size_fit_decision_accuracy')
    size_txt='未測定' if sz is None else f'{sz*100:.1f}%'
    recall=a.get('oversize_reject_recall')
    recall_txt='未測定' if recall is None else f'{recall*100:.1f}%'
    write(d,(43,465),'RGB-Dサイズ規格一致：'+size_txt+'　大きすぎる食品の除外率：'+recall_txt,17,(142,231,198))
    write(d,(43,499),'※合成環境・簡易把持。中心位置は画像検出、ID対応は既知セル。',15,(247,202,150))
    return res

def draw_intro(src):
    res=load_frame(src,len(src)//4,(W,H)).filter(ImageFilter.GaussianBlur(3))
    shade=Image.new('RGBA',(W,H),(5,15,28,166))
    res=Image.alpha_composite(res.convert('RGBA'),shade).convert('RGB');d=ImageDraw.Draw(res)
    write(d,(50,76),'Physical AI 最終課題',31,(154,224,225))
    write(d,(47,155),'AIお弁当盛り付け',48,bold=True)
    write(d,(47,222),'3Dクレーン型・3本爪ロボット',33,bold=True)
    write(d,(51,280),'カメラで中心位置・種類・サイズを認識、6マスの規格と照合',24,(137,239,203))
    write(d,(50,322),'同一シーンの従来方式とAI方式を、左右で比較',24,(242,223,171))
    write(d,(51,421),'MuJoCo 3D / CNN / 逆運動学 / 音声解説',23,(193,218,225))
    # Static recipe diagram, clearly labelled as a plan rather than sensor output.
    box(d,(664,133,946,384),(22,43,55),(104,185,186),15)
    write(d,(677,148),'注文：2行×3列',21,(164,245,219),bold=True)
    for slot,cls in enumerate(RECIPE):
        x=679+(slot%3)*86
        y=196+(slot//3)*86
        box(d,(x,y,x+79,y+72),(41,66,78),(88,143,157),8)
        write(d,(x+7,y+7),f'第{slot+1}区画',12,(181,213,230))
        write(d,(x+9,y+35),label_name(cls),14,(251,247,218),bold=True)

    return res

def draw_disclaimer():
    res=plain_canvas();d=ImageDraw.Draw(res)
    write(d,(60,105),'実装範囲と限界',43,bold=True)
    write(d,(60,204),'3Dロボットの関節・爪の開閉はMuJoCoで描画',25,(134,222,198))
    write(d,(60,264),'食品の保持は理想化したキネマティック拘束',25)
    write(d,(60,318),'サイズ推定は深度差分による輪郭面積と幅の校正',23)
    write(d,(60,360),'中心XYは画像検出、把持の保持・ID対応は簡易化',22)
    write(d,(60,424),'映像は評価フレームを編集（実時間ではありません）',21,(213,201,159))
    return res

def narration_audio(root,override=None):
    """Produce 120s Japanese voice WAV. No silent substitute is allowed."""
    root=Path(root); path=root/'narration_ja.wav'
    override=Path(override) if override else root/'narration_override.wav'
    if override.exists():
        # Input can be WAV/MP3/M4A; ffmpeg converts to fixed sample rate.
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(override),
             '-ar',str(SR),'-ac','1','-f','s16le','-']
        p=subprocess.run(cmd,capture_output=True)
        if p.returncode:raise RuntimeError('ユーザー音声の変換失敗: '+p.stderr.decode(errors='replace'))
        raw=np.frombuffer(p.stdout,dtype='<i2').astype(np.float32)/32768
        if len(raw)<SR*30:raise RuntimeError('提供音声が30秒未満です。120秒に近い日本語ナレーションを用意してください。')
        audio=np.zeros(SR*DURATION,np.float32)
        audio[:min(len(audio),len(raw))]=raw[:len(audio)]
        sf.write(path,audio,SR,subtype='PCM_16')
        return path
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise RuntimeError('日本語音声には gTTS が必要です: %pip install gTTS') from exc
    cache=root/'voice_cache';cache.mkdir(exist_ok=True)
    audio=np.zeros(SR*DURATION,dtype=np.float32)
    for i,(start,seconds,text) in enumerate(TIMELINE):
        # 録音の話速は全編一定。長い場合は原稿を短縮し、速度変更はしない。
        selected = None
        clip = None
        candidates = [text] + VOICE_ALTERNATIVES.get(i, [])
        for candidate in candidates:
            digest=hashlib.sha256(candidate.encode('utf-8')).hexdigest()[:12]
            mp3=cache/f'part_{i:02d}_{digest}.mp3'
            if not mp3.exists() or mp3.stat().st_size<1000:
                try:
                    gTTS(candidate,lang='ja',slow=False).save(str(mp3))
                except Exception as e:
                    raise RuntimeError('日本語音声合成に失敗しました。Colabの接続を確認するか、'
                                       f'{root}/narration_override.wav に自分の声を置いてください。') from e
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-i',str(mp3),
                 '-ar',str(SR),'-ac','1','-f','s16le','-']
            proc=subprocess.run(cmd,capture_output=True)
            if proc.returncode:
                raise RuntimeError('音声変換失敗: '+proc.stderr.decode(errors='replace')[:500])
            candidate_clip=np.frombuffer(proc.stdout,dtype='<i2').astype(np.float32)/32768
            available=seconds-.60
            if len(candidate_clip)<=int(available*SR):
                clip=candidate_clip.copy()
                selected=candidate
                break
            print(f'[VOICE] 区間{i}: {len(candidate_clip)/SR:.1f}秒 > {available:.1f}秒。等速で短い原稿へ変更します。',flush=True)
        if clip is None:
            raise RuntimeError(f'音声区間{i}の候補がすべて長すぎます。VOICE_ALTERNATIVES[{i}]の短縮が必要です。')
        # 字幕原稿も採用された台詞に統一する。
        TIMELINE[i]=(start,seconds,selected)
        fade=min(len(clip)//8,int(.12*SR))
        if fade:
            clip[-fade:]*=np.linspace(1,0,fade,dtype=np.float32)
        st=int((start+.20)*SR)
        audio[st:st+len(clip)]+=clip*.92
    peak=np.max(np.abs(audio))
    if peak>0.95:audio*=.95/peak
    if peak<.005:raise RuntimeError('音声が無音です。ナレーションを確認してください。')
    sf.write(path,audio,SR,subtype='PCM_16')
    return path

def srt(root):
    def stamp(sec):
        sec=int(sec)
        return f'{sec//3600:02d}:{(sec//60)%60:02d}:{sec%60:02d},000'
    txt=''
    for k,(start,sec,line) in enumerate(TIMELINE,1):
        txt+=f'{k}\n{stamp(start)} --> {stamp(start+sec)}\n{line}\n\n'
    p=Path(root)/'narration_ja.srt';p.write_text(txt,encoding='utf-8');return p

def make_video(root,summary,output,voice_override=None):
    root=Path(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    b=sorted((root/'frames_baseline').glob('frame_*.jpg'))
    a=sorted((root/'frames_ai').glob('frame_*.jpg'))
    z=sorted((root/'frames_ai_claw').glob('frame_*.jpg'))
    befile=root/'events_baseline.json';aefile=root/'events_ai.json'
    if min(len(a),len(b))<12 or not befile.exists() or not aefile.exists():
        raise RuntimeError('同一Scene 1の従来方式・AI方式の実録画がありません。先に --scenes 2 以上で本番評価を実行してください。')
    be=json.loads(befile.read_text());ae=json.loads(aefile.read_text())
    if not be or not ae or any(x['scene']!=1 for x in be+ae):
        raise RuntimeError('比較映像のScene IDが一致しません。評価を再実行してください。')
    # First create narration. A video without voice is NOT marked completed.
    speech=narration_audio(root,voice_override)
    srt(root)
    intro=draw_intro(a);final_b=load_frame(b,len(b)-1,(720,480));final_a=load_frame(a,len(a)-1,(720,480))
    audits={}
    for mode in ('baseline','ai'):
        af=root/f'inspection_scene1_{mode}.json'
        if af.exists():audits[mode]=json.loads(af.read_text(encoding='utf-8'))
    phases={}
    for mode in ('baseline','ai'):
        pf=root/f'phases_{mode}.json'
        phases[mode]=json.loads(pf.read_text(encoding='utf-8')) if pf.exists() else []
    missing=draw_missing(b,a,summary,audits)
    missing.save(root/'comparison_scene.png')
    score=draw_metrics(summary);lim=draw_disclaimer()
    raw=root/'video_raw_noaudio.mp4'
    w=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
    if not w.isOpened():raise RuntimeError('MP4中間ファイルの作成に失敗しました')
    try:
        for f in range(FPS*DURATION):
            t=f/FPS
            if t<10: im=intro
            elif t<48: im=draw_compare(t,b,a,be,ae,phases['baseline'],phases['ai'])
            elif t<88: im=draw_close(t,a,z,ae,phases['ai'])
            elif t<101: im=missing
            elif t<113: im=score
            else: im=lim
            w.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR))
    finally:w.release()
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(raw),'-i',str(speech),
         '-map','0:v:0','-map','1:a:0','-c:v','libx264','-crf','22','-preset','veryfast',
         '-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-t',str(DURATION),'-movflags','+faststart',str(output)]
    proc=subprocess.run(cmd,capture_output=True,text=True)
    if proc.returncode or not output.exists():
        raise RuntimeError('日本語音声つきMP4の作成失敗: '+proc.stderr[-1500:])
    p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                       '-of','default=nw=1:nk=1',str(output)],capture_output=True,text=True)
    duration=float(p.stdout.strip())
    if abs(duration-DURATION)>.5:raise RuntimeError(f'動画が120秒になっていません: {duration}')
    has_audio=subprocess.run(['ffprobe','-v','error','-select_streams','a:0',
                               '-show_entries','stream=codec_name','-of','csv=p=0',str(output)],
                              capture_output=True,text=True).stdout.strip()
    if not has_audio:raise RuntimeError('音声ストリームがありません。')
    raw.unlink(missing_ok=True)
    return duration

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',default='results_bento6')
    ap.add_argument('--output',default='')
    ap.add_argument('--voice-override',default='')
    x=ap.parse_args()
    root=Path(x.results)
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    out=Path(x.output) if x.output else root/'PAI_Bento3D_120s.mp4'
    print('Narrated video seconds:',make_video(root,summary,out,x.voice_override or None))
