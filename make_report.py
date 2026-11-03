"""Two-page Japanese PDF generated ONLY after the actual evaluation finishes."""
from pathlib import Path
import json
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image,PageBreak,KeepTogether
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
FONT='HeiseiKakuGo-W5'
ST={
 't':ParagraphStyle('t',fontName=FONT,fontSize=17,leading=23,spaceAfter=9,textColor=colors.HexColor('#22515C')),
 'h':ParagraphStyle('h',fontName=FONT,fontSize=11,leading=16,spaceBefore=7,spaceAfter=4,textColor=colors.HexColor('#36717C'),keepWithNext=True),
 'b':ParagraphStyle('b',fontName=FONT,fontSize=8.5,leading=13,spaceAfter=5),
 's':ParagraphStyle('s',fontName=FONT,fontSize=7.5,leading=11,spaceAfter=5,textColor=colors.HexColor('#5B6570')),
}
def make_report(summary_path,output,account,github='',youtube=''):
    data=json.loads(Path(summary_path).read_text(encoding='utf8'))
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    d=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=36,bottomMargin=37,title='Physical AI 最終課題 AIお弁当盛り付けロボット')
    items=[]
    def p(s,t='b'):
        items.append(Paragraph(escape(str(s)).replace('\n','<br/>'),ST[t]))
    def h(s):p(s,'h')
    p('PAI最終課題｜AIお弁当盛り付けロボット（3D）','t')
    p('普通サイズと特大サイズを識別し、画像で中心位置を測って6区画に盛り付ける','s')
    h('1. 課題とねらい')
    p('混在する食品から、おにぎり・卵焼きを各2個、ブロッコリー・唐揚げを各1個選び、指定の弁当箱6区画へ順番に盛り付ける。余剰食品や注文外のトマトは別の回収トレーへ運搬する。欠品がある場合には未充足の区画を記録する。サイズは普通1.00倍と特大（おにぎり1.55倍・卵焼き1.50倍）だけを用意し、盛り付けサイズ規格を超えるものは回収する。')
    h('2. AI・ロボットの構成')
    p('MuJoCoでSCARA型4自由度ロボット、弁当箱6区画、作業台、食品の3D形状を作成。MuJoCoカメラ画像から学習用RGBを生成して5クラスCNNを学習する。推論時は上方カメラの全画面深度差分から食品の輪郭と中心点を検出し、カメラの既知の投影パラメータでXY座標に変換する。その座標をCNN切り出しとアームのXY目標に利用する。種類CNNに加え、食品と空背景の深度差から輪郭を抽出し、面積と幅を校正して相対サイズを推定する。各区画の盛り付け規格に収まる候補を選び、各区画1個の制約を守って解析的逆運動学でピック＆プレースする。サイズ規格は料理の品質基準であり、物理衝突による容量制約ではない。')
    h('3. 評価の設定')
    p(f"CNN学習画像 {data['train_n']} 枚、検証画像 {data['val_n']} 枚、{data['epochs']} エポック。評価は {data['scenes']} シーンで、従来方式と提案AI方式に共通の初期条件を使用する。食品の入れ替わり、大小の食材、規格外サイズ、余剰品、異物、欠品を含む。")
    h('4. 実測結果（この実行のsummary.jsonから自動入力）')
    b,a=data['baseline'],data['ai']
    o=data.get('oracle',{})
    lines=[['評価項目','固定順','提案AI','理想識別※'],
           ['商品の扱いの正解率',f"{b['item_success_rate']*100:.1f}%",f"{a['item_success_rate']*100:.1f}%",f"{o.get('item_success_rate',0)*100:.1f}%" if o else '-'],
           ['6区画の最終配置正解率',f"{b['compartment_success_rate']*100:.1f}%",f"{a['compartment_success_rate']*100:.1f}%",f"{o.get('compartment_success_rate',0)*100:.1f}%" if o else '-'],
           ['区画別欠品判定精度',f"{b['missing_detection_accuracy']*100:.1f}%",f"{a['missing_detection_accuracy']*100:.1f}%",f"{o.get('missing_detection_accuracy',0)*100:.1f}%" if o else '-'],
           ['CNN分類正解率','未使用',f"{a['classification_accuracy']*100:.1f}%",'真値入力'],
           ['サイズ規格判定一致率','未使用',f"{(a.get('size_fit_decision_accuracy') or 0)*100:.1f}%",'真値入力'],
           ['評価件数',str(b['n_items']),str(a['n_items']),str(o.get('n_items','-'))]]
    table=Table(lines,colWidths=[174,103,103,103],rowHeights=25)
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),FONT),('FONTSIZE',(0,0),(-1,-1),8),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#235263')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3F0EF')),('GRID',(0,0),(-1,-1),.45,colors.HexColor('#BCCDD0')),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    items.append(table)
    ab=data.get('ai_no_size')
    if ab:p(f"アブレーション：CNNで種類のみ判定しサイズ規格を使用しない場合の作業成功率は {ab['item_success_rate']*100:.1f}%（同じ初期条件）。提案AIとの差でサイズ規格判定の寄与を調べる。",'s')
    if a.get('oversize_reject_recall') is not None:
        p(f"規格外食材の回収率 {a['oversize_reject_recall']*100:.1f}%、規格内食品の正規配置率 {a.get('valid_food_placed_rate',0)*100:.1f}%（評価シーン）。",'s')
    p(f"画像中心のXY座標推定MAE: {data.get('localization_mae_m',float('nan'))*1000:.1f} mm。検出後にのみ評価用真値と比較。",'s')
    p('※理想識別は食品種類の真値を計画器に与える参考上限（AIではない）。固定順・AI・理想識別は同じ初期配置で評価。成功の定義は正しい区画／回収、位置、方向を満たすこと。CNN認識精度とロボット作業成功率は別指標。','s')
    h('5. 比較方法と応用')
    p('動画では6区画の両方式を同じScene 1（ブロッコリー欠品、順番の入れ替わり）で比較する。従来方式は食品の種類・サイズを認識せず固定順で配置し、提案方式は画像からの中心位置、CNN分類、深度画像からのサイズ推定と注文規格に基づき区画を選択する。結果には誤認識も含め、そのまま実測評価する。給食・惣菜製造での取り違え防止を目指す概念実証である。')
    items.append(PageBreak())
    h('6. 同じ欠品シーンの従来方式とAI方式を比較')
    img=out.parent/'comparison_scene.png'
    if not img.exists():img=out.parent/'robot_action.png'
    if img.exists():
        from PIL import Image as PILImage
        with PILImage.open(img) as im:w,hg=im.size
        tw=480;items.append(Image(str(img),width=tw,height=tw*hg/w))
        p('同一Scene 1で録画した従来方式とAI方式の実際のMuJoCoフレーム。映像は日本語ナレーション付き120秒MP4に編集。CNNの予測は真値へ置換せずに表示する。','s')
    else:p('動画フレーム未生成（短縮テストでは動画を作成しない）。','s')
    h('7. 配置後の確認と失敗分析')
    audits=[]
    for name in ('baseline','ai'):
        f=out.parent/f'inspection_scene1_{name}.json'
        if f.exists():
            a=json.loads(f.read_text(encoding='utf-8'))
            status=' / '.join(f"{['おにぎり','卵焼き','ブロッコリー','唐揚げ','おにぎり','卵焼き'][int(row['compartment'])]}:{row['state']}" for row in a)
            audits.append(f'{name}: {status}')
    p(' / '.join(audits) if audits else '配置後の照合データは未生成。')
    p('配置後検査はMuJoCoの物体座標、食品真値、規格内サイズを照合するもので、画像AIによる再認識ではない。CNN分類の誤りはevents_ai.jsonに残し、評価値を改変しない。','s')
    h('8. 実装範囲と重要な限界')
    p('3Dロボットの関節角・上下動は逆運動学で決め、MuJoCoの関節状態を直接更新する。3本の爪はMuJoCoのヒンジ関節で開閉するが、食品を持ち上げる保持処理はキネマティックな理想化拘束であり、接触把持力、滑り、衝突回避や食品変形は検証していない。CNNの切り出し中心とアーム接近XYは、カメラ画像の輪郭重心を投影変換した座標を使用する。ただし供給トレーの既知セルとMuJoCo食品IDの対応、把持角・高さ、空背景生成はシミュレーターに依存する。食品サイズは校正済みのRGB-D深度差分の輪郭面積・幅から推定しており、未知の遮蔽やカメラ条件では未検証。規格外判定は盛り付け規格であり物理把持の限界は表さない。学習・検証は合成環境のみで、実機への一般化は未検証。')
    h('9. 課題の拡張案と参考')
    p('将来は画像からの3D寸法復元、食品把持の力学、複数個配置、重量・アレルゲン判定、注文変更へのリアルタイム適応、遮蔽下や未知の配置でも動く一般物体検出を追加する。参考：MuJoCo公式ドキュメント、PyTorch公式ドキュメント。製作には生成AIをコード作成支援として利用。')
    h('10. 提出先情報')
    p('omnicampus: '+account)
    p('GitHub: '+(github or '公開後にURLを追記'))
    p('YouTube: '+(youtube or '公開後にURLを追記'))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont(FONT,8);canvas.setFillColor(colors.grey)
        canvas.drawString(44,22,'PAI最終課題｜実測結果に基づく報告資料')
        canvas.drawRightString(A4[0]-44,22,str(doc.page));canvas.restoreState()
    d.build(items,onFirstPage=footer,onLaterPages=footer)
    return str(out)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--summary',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--account',default='ACCOUNT');ap.add_argument('--github',default='');ap.add_argument('--youtube',default='')
    a=ap.parse_args();print(make_report(a.summary,a.output,a.account,a.github,a.youtube))
