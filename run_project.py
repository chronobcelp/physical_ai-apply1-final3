"""Train MuJoCo-camera CNN, evaluate bento task, record ACTUAL robot frames, make MP4/PDF.
No metrics or video frames are fabricated. 3D kinematic robot with articulated 3-finger crane claw; idealized payload attachment.
"""
import argparse,csv,json,math,os,sys,random,shutil,zipfile
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch
from scene_3d import BentoSim,ITEM_CLASSES,ITEM_SCALES,SIZE_LIMITS,size_fits,destination_height,NAMES,SLOTS,REJECT,ORDER,RECIPE,SLOT_CLASSES,EPISODES,SOURCE
from vision_model import (FoodCNN,train,predict,image_crop,calibrate_sizes,
                          estimate_size,validate_size_camera,SAFE_MARGIN,
                          detect_food_centers,image_crop_pixel,draw_detection_overlay,
                          depth_mask,calibrate_color_prototypes,
                          clean_neighbor_pixels,predict_robust)
from make_video import make_video
from make_report import make_report


def reject_pos(index):
    return (0.033+0.054*(index%2),-.45 +.032*(index//2),.081)

def angle_error(yaw):
    # Food items need their visual direction aligned to the compartment.
    return abs(((math.degrees(yaw)+90)%180)-90)

def inspect_compartments(sim):
    """Ground-truth geometric audit after the episode; distinct from CNN perception."""
    slots=[]
    for slot in ORDER:
        cls=SLOT_CLASSES[slot]
        x,y,z=SLOTS[slot]
        present=[]
        for item in sim.active:
            pos,_=sim.food_pose(item)
            if np.linalg.norm(pos[:2]-np.array((x,y))) < .050:
                present.append({'item_id':int(item),'true_class':int(ITEM_CLASSES[item])})
        state='EMPTY' if not present else ('CORRECT' if len(present)==1 and present[0]['true_class']==cls and size_fits(cls,ITEM_SCALES[present[0]['item_id']]) else 'OVERSIZE' if len(present)==1 and present[0]['true_class']==cls else 'WRONG_OR_OVERFULL')
        slots.append({'compartment':int(slot),'recipe_class':int(cls),'state':state,'items':present})
    return slots



def plan_six_slots(mode, active, predictions, estimated_sizes):
    """Deterministic 6-slot plan; food classes may repeat in different slots.

    Accepts predicted class and RGB-D estimated scale, never the object's
    ground-truth identity or size for AI decision-making.
    """
    if mode=='baseline':
        return [(item,ORDER[k],'PLACE') if k<len(ORDER) else (item,None,'REJECT')
                for k,item in enumerate(active)]
    remaining=set(active)
    result=[]
    for slot in ORDER:
        cls=SLOT_CLASSES[slot]
        candidates=[item for item in remaining if predictions[item][0]==cls
                    and (mode=='ai_no_size' or
                         (np.isfinite(estimated_sizes[item]) and
                          estimated_sizes[item]<=SIZE_LIMITS[cls]-SAFE_MARGIN))]
        if candidates:
            item=min(candidates,key=lambda i:(
                abs(estimated_sizes[i]-1.0) if mode!='ai_no_size' else 0,
                -predictions[i][1],i))
            result.append((item,slot,'PLACE'))
            remaining.remove(item)
    return result+[(item,None,'REJECT') for item in active if item in remaining]


def evaluate(sim,model,device,mode,seed,nscenes,out,reference,record=False,color_references=None):
    out=Path(out)
    records=[];scene_records=[];localization_rows=[]
    for scene_idx in range(nscenes):
        sim.reset(scene_idx,seed)
        overhead=sim.render(top=True)
        depth=sim.render_depth()
        empty_depth=sim.render_empty_depth()
        if scene_idx==0 and mode=='ai':
            cv2.imwrite(str(out/'camera_overhead.png'),cv2.cvtColor(overhead,cv2.COLOR_RGB2BGR))
        predictions={}; estimated_sizes={}; image_positions={}; detected_pixels={}; classification_rows=[]
        if mode in ('ai','ai_no_size'):
            # Segment ALL foods from one overhead RGB-D frame. No GT-centered
            # crop is provided to the CNN or the size estimator.
            try:
                detections, full_mask = detect_food_centers(overhead,depth,empty_depth,SOURCE)
            except RuntimeError as exc:
                # Save evidence even if localization stops: no hidden GT fallback.
                prefix=out/f'localization_failed_{mode}_scene{scene_idx}'
                cv2.imwrite(str(prefix.with_name(prefix.name+'_rgb.png')),
                            cv2.cvtColor(overhead,cv2.COLOR_RGB2BGR))
                delta=np.uint8(np.clip((empty_depth-depth)*2000,0,255))
                cv2.imwrite(str(prefix.with_name(prefix.name+'_depth_difference.png')),
                            delta)
                cv2.imwrite(str(prefix.with_name(prefix.name+'_mask.png')),
                            np.uint8((np.isfinite(depth)&np.isfinite(empty_depth)&
                              (empty_depth-depth)>.008))*255)
                print('IMAGE DETECTION FAILED; saved diagnostic PNGs:',prefix,flush=True)
                raise
            if mode=='ai':
                if scene_idx in (0,1):
                    cv2.imwrite(str(out/f'detection_overlay_scene{scene_idx}.png'),
                                draw_detection_overlay(overhead,detections))
                    cv2.imwrite(str(out/f'detection_mask_scene{scene_idx}.png'),full_mask)
            for d in detections:
                # Nominal tray cell -> MuJoCo payload ID is executor-only metadata.
                # The image centroid controls the XY approach of the robot.
                obj=sim.active[d['cell']]
                x,y=d['xy'];u,v=d['pixel']
                image_positions[obj]=(x,y)
                detected_pixels[obj]=(u,v)
                roi=image_crop_pixel(overhead,u,v)
                # The original CNN sees neighbouring items in its 96px crop;
                # near the rear cells this caused rice/egg misclassification.
                # Suppress other measured RGB-D foreground, not the actual item.
                target_mask=depth_mask(depth,empty_depth,(x,y))
                clean_roi=clean_neighbor_pixels(roi,full_mask,target_mask,u,v)
                if mode=='ai' and scene_idx in (0,1):
                    cv2.imwrite(str(out/f'roi_scene{scene_idx}_item_{obj}.png'),
                                cv2.cvtColor(clean_roi,cv2.COLOR_RGB2BGR))
                cls,conf,debug=predict_robust(model,clean_roi,target_mask,device,
                                             color_references,return_debug=True)
                predictions[obj]=(cls,conf)
                classification_rows.append({'scene':scene_idx,'mode':mode,
                    'source_cell':d['cell'],'detected_x':x,'detected_y':y,
                    **debug})
                estimated_sizes[obj]=estimate_size(depth,empty_depth,(x,y),
                                                    predictions[obj][0],reference)
                # Ground truth is used ONLY AFTER image detection, for accuracy audit.
                pos,_=sim.food_pose(obj)
                error=float(np.hypot(x-pos[0],y-pos[1]))
                localization_rows.append({'mode':mode,'scene':scene_idx,'item_id':obj,
                    'cell':d['cell'],'pixel_u':round(u,2),'pixel_v':round(v,2),
                    'detected_x_m':x,'detected_y_m':y,
                    'true_x_m':round(float(pos[0]),5),
                    'true_y_m':round(float(pos[1]),5),
                    'error_m':round(error,5)})
        else:
            for obj in sim.active:
                if mode=='oracle':
                    predictions[obj]=(int(ITEM_CLASSES[obj]),1.0)
                    estimated_sizes[obj]=float(ITEM_SCALES[obj])
                else:
                    predictions[obj]=(None,0.)
                    estimated_sizes[obj]=float('nan')
        if mode=='ai' and scene_idx in (0,1) and classification_rows:
            with (out/f'classification_scene{scene_idx}.csv').open('w',newline='',encoding='utf8') as f:
                writer=csv.DictWriter(f,classification_rows[0].keys())
                writer.writeheader();writer.writerows(classification_rows)
        actions=plan_six_slots(mode,sim.active,predictions,estimated_sizes)
        placed=set();rejected=0;frame_index=0;frame_written=0
        # Crucial for a FAIR video comparison: BOTH policies record episode 1,
        # with exactly the same initial layout and random seed.
        # AI episode 0 is also recorded for a short demonstration.
        capture_this=record and mode in ('baseline','ai') and scene_idx in (0,1) and (mode=='ai' or scene_idx==1)
        if mode=='baseline':
            frame_dir=out/'frames_baseline'
            event_name='events_baseline.json'
        elif scene_idx==1:
            frame_dir=out/'frames_ai'
            event_name='events_ai.json'
        else:
            frame_dir=out/'frames_ai_full'
            event_name='events_ai_full.json'
        zoom_dir=out/'frames_ai_claw' if mode=='ai' and scene_idx==1 else None
        events=[]
        frame_phases=[]
        if capture_this:
            if frame_dir.exists():shutil.rmtree(frame_dir)
            frame_dir.mkdir(parents=True)
            if zoom_dir is not None:
                if zoom_dir.exists():shutil.rmtree(zoom_dir)
                zoom_dir.mkdir(parents=True)
            print(f'RECORD matched episode={scene_idx}, mode={mode}, frames={frame_dir}',flush=True)
        for step,(obj,slot,action) in enumerate(actions):
            pred_cls,conf=predictions[obj]
            true_cls=ITEM_CLASSES[obj]
            gt_fits=size_fits(true_cls,ITEM_SCALES[obj])
            if action=='PLACE':
                target=(*SLOTS[slot][:2],destination_height(obj))
                placed.add(slot)
            else:
                target=(*reject_pos(rejected)[:2],destination_height(obj, reject=True));rejected+=1
            action_start=frame_written
            def capture(_):
                nonlocal frame_index,frame_written
                if not capture_this:return
                frame_index+=1
                if frame_index%2:return
                image=cv2.cvtColor(sim.render(top=False),cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_dir/f'frame_{frame_written:05d}.jpg'),image,
                            [cv2.IMWRITE_JPEG_QUALITY,90])
                if zoom_dir is not None:
                    detail=cv2.cvtColor(sim.render_claw(),cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(zoom_dir/f'frame_{frame_written:05d}.jpg'),detail,
                                [cv2.IMWRITE_JPEG_QUALITY,87])
                frame_phases.append(str(getattr(sim,'phase','UNKNOWN')))
                frame_written+=1
            callback=capture if capture_this else None
            # Image-derived XY controls approach in both AI policies.
            # Baseline/oracle remain their original non-vision controls.
            sim.pick(obj,callback=callback,
                     xy_override=image_positions.get(obj))
            sim.place(obj,target,callback=callback)
            if capture_this:
                # Actual model prediction is retained, not replaced by ground truth.
                events.append({'from_frame':action_start,'to_frame':frame_written,
                               'scene':scene_idx,'mode':mode,'object_id':int(obj),
                               'true_class':int(true_cls),
                               'prediction':int(pred_cls) if pred_cls is not None else -1,
                               'confidence':round(float(conf),3),
                               'image_xy':list(image_positions[obj]) if obj in image_positions else None,
                               'image_uv':list(detected_pixels[obj]) if obj in detected_pixels else None,
                               'size_true_x':round(float(ITEM_SCALES[obj]),2),
                               'size_est_x':round(float(estimated_sizes[obj]),2) if np.isfinite(estimated_sizes[obj]) else None,
                               'size_limit_x':float(SIZE_LIMITS[SLOT_CLASSES[slot]]) if slot is not None else None,
                               'size_status':'IN_SPEC' if action=='PLACE' else ('OVERSIZE' if not gt_fits and true_cls<4 else 'EXTRA_OR_FOREIGN'),
                               'rejection_reason': ('IMG_TOO_LARGE' if action=='REJECT' and pred_cls in SIZE_LIMITS and np.isfinite(estimated_sizes[obj]) and estimated_sizes[obj]>SIZE_LIMITS[pred_cls]-SAFE_MARGIN else 'UNCERTAIN_MEASUREMENT' if action=='REJECT' and pred_cls in SIZE_LIMITS and not np.isfinite(estimated_sizes[obj]) else 'EXTRA_OR_FOREIGN' if action=='REJECT' else ''),
                               'action':action,
                               'slot':int(slot) if slot is not None else -1,
                               'slot_recipe_class':int(SLOT_CLASSES[slot]) if slot is not None else -1,
                               'confidence_warning':bool(mode=='ai' and conf<0.60)})
            actual,yaw=sim.food_pose(obj)
            target_reached=bool(np.linalg.norm(actual[:2]-np.array(target[:2]))<.020)
            # Each placed item must occupy the intended food-class slot. A class can
            # legitimately appear twice in the six-compartment recipe.
            correct_place=bool(action=='PLACE' and gt_fits and
                SLOT_CLASSES[slot]==true_cls and target_reached and
                angle_error(yaw)<12)
            # Rejecting edible, in-spec ingredients is correct only if the recipe
            # has already been fulfilled for that food class with correct placements.
            properly_placed=sum(int(r['action']=='PLACE' and r['correct']==1 and
                r['true_class']==true_cls) for r in records if
                r['scene']==scene_idx and r['mode']==mode)
            required=RECIPE.count(true_cls)
            correct_reject=bool(action=='REJECT' and target_reached and (
                true_cls==4 or not gt_fits or properly_placed>=required))
            correct=bool(correct_place or correct_reject)
            if capture_this:
                events[-1]['correct']=int(correct)
                events[-1]['final_yaw_error_deg']=round(angle_error(yaw),2)
            records.append({'mode':mode,'scene':scene_idx,'item_id':obj,'true_class':true_cls,
                'pred_class':pred_cls if pred_cls is not None else -1,'confidence':round(conf,4),
                'action':action,'requested_compartment':slot if slot is not None else -1,
                'position_x':round(float(actual[0]),4),'position_y':round(float(actual[1]),4),
                'final_yaw_error_deg':round(angle_error(yaw),2),
                'detected_x_m':round(image_positions[obj][0],4) if obj in image_positions else '',
                'detected_y_m':round(image_positions[obj][1],4) if obj in image_positions else '',
                'size_true_x':round(float(ITEM_SCALES[obj]),3),
                'size_est_x':round(float(estimated_sizes[obj]),3) if np.isfinite(estimated_sizes[obj]) else '',
                'size_eligible':int(gt_fits),'correct':int(correct)})
        # Store all six post-action compartments for every mode and every scene.
        audited=inspect_compartments(sim)
        if mode=='ai' and scene_idx==1:
            rear=[v for v in audited if v['compartment'] in (4,5)]
            (out/'scene1_rear_slots_audit.json').write_text(
                json.dumps(rear,ensure_ascii=False,indent=2),encoding='utf8')
            for status in rear:
                print(f"REAR SLOT {status['compartment']+1}: "
                      f"{NAMES[status['recipe_class']]} / "
                      f"observed={status['state']} items={status['items']}",flush=True)

        available_now=Counter(ITEM_CLASSES[i] for i in sim.active
                              if size_fits(ITEM_CLASSES[i],ITEM_SCALES[i]))
        expected_seen=Counter()
        for slot_status in audited:
            slot_id=int(slot_status['compartment'])
            cls=SLOT_CLASSES[slot_id]
            expected_seen[cls]+=1
            should_be_filled=available_now[cls]>=expected_seen[cls]
            slot_status.update({'mode':mode,'scene':scene_idx,
                                'expected':'CORRECT' if should_be_filled else 'EMPTY',
                                'slot_correct':int(slot_status['state']==
                                    ('CORRECT' if should_be_filled else 'EMPTY'))})
        scene_records_slot_status=audited
        if capture_this:
            (out/event_name).write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf8')
            (out/('phases_baseline.json' if mode=='baseline' else
                  'phases_ai.json' if scene_idx==1 else 'phases_ai_full.json')).write_text(
                  json.dumps(frame_phases,ensure_ascii=False),encoding='utf8')
            if scene_idx==1:
                (out/f'inspection_scene1_{mode}.json').write_text(
                    json.dumps(inspect_compartments(sim),ensure_ascii=False,indent=2),encoding='utf8')
        # Ground-truth missing slots: for repeated classes, the first N slots are
        # fulfillable if N eligible items exist, and the rest are genuinely missing.
        available=Counter(ITEM_CLASSES[i] for i in sim.active
                          if size_fits(ITEM_CLASSES[i],ITEM_SCALES[i]))
        remaining_true=Counter()
        missing_gt=[]
        for slot in ORDER:
            cls=SLOT_CLASSES[slot]
            remaining_true[cls]+=1
            if available[cls] < remaining_true[cls]:
                missing_gt.append(slot)
        missing_pred=[slot for slot in ORDER if slot not in placed]
        scene_records.append({'mode':mode,'scene':scene_idx,'slot_audit':scene_records_slot_status,
                              'correct_slots':sum(r['slot_correct'] for r in scene_records_slot_status),
                              'missing_true':missing_gt,
                              'missing_pred':missing_pred,
                              'missing_classes_correct':sum((slot in missing_gt)==(slot in missing_pred) for slot in ORDER)})
        print(f'{mode}: scene {scene_idx+1}/{nscenes} item success={sum(r["correct"] for r in records if r["scene"]==scene_idx)}/{len(sim.active)}; missing GT={missing_gt}, predicted={missing_pred}',flush=True)
    return records,scene_records,localization_rows

def metrics(items,scenes):
    n=len(items)
    ai_items=[r for r in items if r['pred_class']>=0]
    valid_size=[r for r in items if r['mode']!='baseline' and r['pred_class']>=0]
    oversize=[r for r in items if r['true_class']<4 and not r['size_eligible']]
    valid=[r for r in items if r['true_class']<4 and r['size_eligible']]
    oversize_recall=sum(r['action']=='REJECT' for r in oversize)/len(oversize) if oversize else None
    valid_placed=sum(r['action']=='PLACE' and r['correct'] for r in valid)/len(valid) if valid else None
    return {'n_items':n,'oversize_reject_recall':oversize_recall,'valid_food_placed_rate':valid_placed,'item_success_rate':sum(r['correct'] for r in items)/n,
            'size_fit_decision_accuracy':sum(int((r['pred_class'] in SIZE_LIMITS and float(r['size_est_x'])<=SIZE_LIMITS[r['pred_class']]-SAFE_MARGIN) == bool(r['size_eligible'])) for r in valid_size if r['size_est_x']!='')/max(1,sum(r['size_est_x']!='' for r in valid_size)) if valid_size and items[0]['mode']!='ai_no_size' else None,
            'size_estimation_mae':sum(abs(float(r['size_est_x'])-r['size_true_x']) for r in valid_size if r['size_est_x']!='')/max(1,sum(r['size_est_x']!='' for r in valid_size)) if valid_size else None,
            'classification_accuracy':sum(r['true_class']==r['pred_class'] for r in ai_items)/len(ai_items) if ai_items else None,
            'missing_detection_accuracy':sum(r['missing_classes_correct'] for r in scenes)/(len(ORDER)*len(scenes)),
            'compartment_success_rate':sum(r['correct_slots'] for r in scenes)/(len(ORDER)*len(scenes)),
            'n_scenes':len(scenes)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results_bento6');ap.add_argument('--train-n',type=int,default=500)
    ap.add_argument('--val-n',type=int,default=125);ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--scenes',type=int,default=4);ap.add_argument('--seed',type=int,default=31031)
    ap.add_argument('--reuse',action='store_true');ap.add_argument('--quick',action='store_true')
    ap.add_argument('--account',default='YOUR_OMNICAMPUS_ACCOUNT');ap.add_argument('--github',default='');ap.add_argument('--youtube',default='')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        if old.get('n_compartments')!=len(ORDER):
            print('WARNING: output directory contains a different recipe. Use results_bento6 to avoid mixing recordings.',flush=True)
    if a.quick:a.train_n=50;a.val_n=25;a.epochs=2;a.scenes=2
    if not a.quick and a.scenes < 2:raise ValueError('Full video comparison requires --scenes >= 2 (Scene 1)')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'PyTorch {torch.__version__}; training device={device}',flush=True)
    print('Starting 3D MuJoCo SCARA / SIX-compartment (2x3) workbench',flush=True)
    sim=BentoSim(size=(720,480),seed=a.seed)
    try:
        # Early render and camera check, before training.
        img=sim.render(top=False)
        cv2.imwrite(str(out/'scene_preview.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        # Store a second close-up preview for judging camera framing before training.
        close_preview=sim.render_claw()
        cv2.imwrite(str(out/'claw_preview.png'),cv2.cvtColor(close_preview,cv2.COLOR_RGB2BGR))
        print('STUDIO PREVIEW: clean scene and 3-finger claw preview saved.', flush=True)
        if a.reuse:
            cp=out/'food_cnn.pt'
            if not cp.exists():raise FileNotFoundError(f'Model not saved: {cp}. Disable reuse.')
            model=FoodCNN().to(device)
            model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['weights'])
            model.eval();print('Reusing saved CNN checkpoint',flush=True)
        else:
            model=train(sim,out,device,a.train_n,a.val_n,a.epochs,a.seed)
        print('Training or checkpoint loading complete. Starting fresh 3D evaluation...',flush=True)
        reference=calibrate_sizes(sim)
        color_references=calibrate_color_prototypes(sim)
        (out/'color_calibration.json').write_text(
            json.dumps(color_references,ensure_ascii=False,indent=2),encoding='utf8')
        print('Calibrated RGB-D colour references for food disambiguation',flush=True)
        (out/'size_calibration.json').write_text(json.dumps(reference,indent=2),encoding='utf8')
        print('RGB-D size calibration:',reference,flush=True)
        size_check=validate_size_camera(sim,reference,out/'size_validation.csv')
        all_items=[];all_scenes=[];all_localizations=[]
        for mode in ['baseline','ai_no_size','ai','oracle']:
            rows,scenes,locs=evaluate(sim,model,device,mode,a.seed+500,a.scenes,out,reference,record=not a.quick,color_references=color_references)
            all_items+=rows;all_scenes+=scenes;all_localizations+=locs
        def export_csv(path,rows):
            with open(path,'w',newline='',encoding='utf8') as f:
                fieldnames=list(rows[0].keys());w=csv.DictWriter(f,fieldnames=fieldnames)
                w.writeheader();w.writerows(rows)
        export_csv(out/'evaluation_detail.csv',all_items)
        if all_localizations:
            export_csv(out/'localization_detail.csv',all_localizations)
        localization_mae=sum(r['error_m'] for r in all_localizations if r['mode']=='ai') / max(1,sum(r['mode']=='ai' for r in all_localizations))
        export_csv(out/'compartment_audit.csv',[
            {'mode':row['mode'],'scene':row['scene'],
             'slot':slot['compartment'],'recipe_class':slot['recipe_class'],
             'actual':slot['state'],'expected':slot['expected'],
             'correct':slot['slot_correct'],
             'items':str(slot['items'])}
            for row in all_scenes for slot in row['slot_audit']])
        export_csv(out/'missing_detail.csv',[{**{k:v for k,v in r.items() if k!='slot_audit'},'missing_true':str(r['missing_true']),'missing_pred':str(r['missing_pred'])} for r in all_scenes])
        summary={'project':'AI Bento Plating Robot 3D / 6 compartments','seed':a.seed,'scenes':a.scenes,
                 'train_n':a.train_n,'val_n':a.val_n,'epochs':a.epochs,
                 'baseline':metrics([r for r in all_items if r['mode']=='baseline'],[r for r in all_scenes if r['mode']=='baseline']),
                 'ai':metrics([r for r in all_items if r['mode']=='ai'],[r for r in all_scenes if r['mode']=='ai']),
                 'ai_no_size':metrics([r for r in all_items if r['mode']=='ai_no_size'],[r for r in all_scenes if r['mode']=='ai_no_size']),
                 'oracle':metrics([r for r in all_items if r['mode']=='oracle'],[r for r in all_scenes if r['mode']=='oracle']),
                 'localization_mae_m':round(localization_mae,5),'localization_method':'Full overhead depth background subtraction, tray-cell pixel partition, pinhole pixel-to-XY backprojection, tray cell association; no GT crop/XY for CNN and robot XY control',
                 'food_classes':NAMES,'n_compartments':len(ORDER),'recipe_classes':list(RECIPE),'compartment_centers':{str(k):list(v) for k,v in SLOTS.items()},'food_size_factors':list(ITEM_SCALES),'rgbd_size_validation':size_check,
                 'allowed_size_max':SIZE_LIMITS,
                 'size_estimation':'RGB-D background-subtracted depth silhouette; 1.0x/oversize classes only; crop and grasp XY derived from camera centroids; empty-background rendering and tray-cell-to-ID mapping are simulator-assisted.','scene_cases':'2x3 six-compartment recipe; two rice slots and two omelet slots, oversize rice/egg, missing broccoli, duplicates, foreign tomato',
                 'vision':'CNN five-crop prediction averaging and training-calibrated RGB-D food-colour features; neighbour-food pixels suppressed by foreground silhouette; XY inferred by image camera pinhole backprojection, not GT',
                 'planning':'Six independent recipe slots (2x3); CNN class + calibrated RGB-D size + safety margin; greedy closest-to-target eligible item without reusing object; reject oversize/uncertain/surplus/foreign.',
                 'robot':'3D SCARA 4-DOF with three animated claw hinge joints; analytical IK and kinematic grasp attachment',
                 'video_comparison':'Baseline and AI recorded from identical missing-broccoli Scene 1; AI-no-size ablation and oracle are measurement-only',
                 'audit_method':'Simulator ground-truth geometric slot occupancy checked AFTER actions; not AI vision inspection',
                 'oracle_definition':'True class known; same rule planner and kinematic manipulator; ideal recognition upper bound, never presented as learned AI',
                 'limitation':'Three articulated fingers close visually; payload follows a kinematic grasp constraint, NOT contact-force gripping. Known tray-cell layout maps detected foods to simulator payload IDs, yaw and pick height remain simulator-assisted, empty background is synthetic. No slip/force/collision validation or real camera validation.'}
        if summary['ai']['item_success_rate']<=summary['ai_no_size']['item_success_rate']:
            print('CAUTION: The size-aware controller did not outperform no-size CNN. Show the measured result; do not claim improvement.',flush=True)
        if (summary['ai']['oversize_reject_recall'] is not None and summary['ai']['oversize_reject_recall']<1.0):
            print('CAUTION: Some oversize objects were NOT rejected. Review evaluation_detail.csv and depth masks.',flush=True)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        if not a.quick:
            frames=sorted((out/'frames_ai').glob('*.jpg'))
            if frames:
                frame=cv2.imread(str(frames[len(frames)//2]));
                if frame is not None:cv2.imwrite(str(out/'robot_action.png'),frame)
            print('Creating 120sec MP4 from genuine 3D robot frames...',flush=True)
            duration=make_video(out,summary,out/'PAI_Bento3D_120s.mp4')
            print('Video duration:',duration,flush=True)
            print('Creating Japanese PDF based on measured results...',flush=True)
            make_report(out/'summary.json',out/f'説明資料_PAI最終課題_{a.account}.pdf',a.account,a.github,a.youtube)
        else:print('QUICK mode: no 120sec video and no PDF',flush=True)
        archive=out/'PAI_Bento3D_Results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for file in out.iterdir():
                if file.is_file() and file!=archive:z.write(file,file.name)
        print('ALL DONE:',out.resolve(),flush=True)
    finally:sim.close()

if __name__=='__main__':main()
