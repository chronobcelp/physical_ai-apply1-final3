"""Food CNN + image-based 2D centroid localization from calibrated RGB-D.
The camera/background/source-cell mapping are known, but item world XY used for
AI classification, size selection and SCARA XY motion comes from measured pixels.
Not a trained bounding-box detector. Physical attachment remains idealized.
"""
from pathlib import Path
import csv,math,random
import cv2,numpy as np,torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from scene_3d import BentoSim, ITEM_CLASSES, ITEM_SCALES, NAMES, SIZE_LIMITS

CROP=96

def project_xy(x,y,shape,z=.081):
    h,w=shape[:2]
    span=2*(1.36-z)*math.tan(math.radians(43/2))
    u=(x-.30)/(span*w/h)*w+w*.5
    v=h*.5-y/span*h
    return round(u),round(v)

def image_crop(rgb,xy):
    u,v=project_xy(*xy,rgb.shape)
    h,w=rgb.shape[:2]
    radius=round(h*.095) # ~ 17cm square around object at 480px high
    image=cv2.copyMakeBorder(rgb,radius,radius,radius,radius,cv2.BORDER_CONSTANT,value=[36,48,57])
    roi=image[int(v):int(v+2*radius),int(u):int(u+2*radius)]
    if roi.size==0:raise ValueError(f'Cannot crop at xy={xy}, uv={(u,v)}')
    return cv2.resize(roi,(CROP,CROP),interpolation=cv2.INTER_AREA)

# Fixed calibrated overhead camera: MuJoCo's camera at (0.30,0,1.36) looks
# along -Z with 43-degree vertical field of view. Pixel centers follow the
# standard pinhole calibration. This uses *image pixels*, never food body poses.
CAMERA_X, CAMERA_Y, CAMERA_Z = .30, 0.0, 1.36
CAMERA_FOVY_DEG = 43.0


def pixel_to_world_xy(u,v,depth_m,shape):
    """Backproject an RGB-D pixel into XY using the known overhead camera."""
    h,w=shape[:2]
    f=h/(2.0*math.tan(math.radians(CAMERA_FOVY_DEG/2)))
    return (float(CAMERA_X+(u-(w-1)/2)*float(depth_m)/f),
            float(CAMERA_Y+((h-1)/2-v)*float(depth_m)/f))


def world_to_pixel_xy(x,y,z,shape):
    """Projection used ONLY for calibration/tests and localization error audit."""
    h,w=shape[:2]
    f=h/(2.0*math.tan(math.radians(CAMERA_FOVY_DEG/2)))
    d=CAMERA_Z-z
    return ((w-1)/2+(x-CAMERA_X)*f/d,
            (h-1)/2-(y-CAMERA_Y)*f/d)


def image_crop_pixel(rgb,u,v):
    """The same 96x96 training-style crop, centered on a DETECTED pixel."""
    h,w=rgb.shape[:2]
    radius=round(h*.095)
    pad=cv2.copyMakeBorder(rgb,radius,radius,radius,radius,
                           cv2.BORDER_CONSTANT,value=[36,48,57])
    ui,vi=int(round(u)),int(round(v))
    roi=pad[vi:vi+radius*2,ui:ui+radius*2]
    if roi.size==0 or roi.shape[:2]!=(2*radius,2*radius):
        raise RuntimeError('Detected image centroid fell outside the valid crop area')
    return cv2.resize(roi,(CROP,CROP),interpolation=cv2.INTER_AREA)


def detect_food_centers(rgb,depth,empty_depth,source_positions):
    """Detect food center XY from an actual RGB-D image, robust to merged blobs.

    The old whole-image connected-component approach counted two touching foods
    as ONE region, intermittently reporting 8/9. This replacement uses known
    *nominal tray CELL locations* solely to partition foreground PIXELS into
    nine regions. Object centers are then measured from camera foreground,
    not recovered from simulator food body poses or per-item ground truth.

    The known 3x3 tray layout is a restriction of this demonstration; this is
    not a free-form object detector, and a missing cell remains a hard error.
    """
    if rgb.shape[:2] != depth.shape or depth.shape != empty_depth.shape:
        raise ValueError('RGB and two depth images need the same resolution')
    h,w=depth.shape
    good=(np.isfinite(depth)&np.isfinite(empty_depth)&(depth>0)&
          (empty_depth>0))
    foreground=np.uint8(good & ((empty_depth-depth)>DEPTH_DELTA_M))*255
    # Do NOT close the full mask: closing merges nearby foods into one blob.
    foreground=cv2.morphologyEx(foreground,cv2.MORPH_OPEN,
                                np.ones((3,3),np.uint8))
    ys,xs=np.nonzero(foreground)
    if len(xs)==0:
        raise RuntimeError('RGB-D has zero foreground food pixels; inspect depth camera.')

    f=h/(2.0*math.tan(math.radians(CAMERA_FOVY_DEG/2)))
    measured_depth=depth[ys,xs].astype(np.float64)
    # Backproject every foreground pixel using MEASURED depth, not food GT XY.
    wx=CAMERA_X+(xs-(w-1)/2.0)*measured_depth/f
    wy=CAMERA_Y+((h-1)/2.0-ys)*measured_depth/f
    xy=np.stack([wx,wy],axis=-1)
    nominal=np.asarray(source_positions,dtype=np.float64)
    if nominal.ndim!=2 or nominal.shape[1]!=2:
        raise ValueError('source_positions must contain (x,y) tray cell centers')
    dist2=((xy[:,None,:]-nominal[None,:,:])**2).sum(axis=2)
    nearest=dist2.argmin(axis=1)
    # Radius covers a 1.55x rice ball plus randomized tray offsets.
    keep=np.min(dist2,axis=1)<(.077**2)
    # No dependence on food IDs; Voronoi partitions split touching food masks.
    found=[];counts=[]
    for cell in range(len(source_positions)):
        choose=(nearest==cell)&keep
        cell_xs=xs[choose];cell_ys=ys[choose]
        counts.append(len(cell_xs))
        if len(cell_xs)<max(MIN_AREA_PX*2,45):
            found.append(None)
            continue
        # Take the largest local connected component, then include nearby
        # foreground pieces (broccoli florets may be disjoint in depth).
        local=np.zeros((h,w),dtype=np.uint8)
        local[cell_ys,cell_xs]=255
        local=cv2.morphologyEx(local,cv2.MORPH_CLOSE,
                              np.ones((3,3),np.uint8))
        n,lab,stats,cent=cv2.connectedComponentsWithStats(local,8)
        candidates=[k for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=MIN_AREA_PX]
        if not candidates:
            found.append(None)
            continue
        central=max(candidates,key=lambda k:stats[k,cv2.CC_STAT_AREA])
        central_xy=np.asarray(cent[central],dtype=np.float64)
        # Only include parts close to the main silhouette; do not use the
        # nominal cell CENTER as a fabricated estimate.
        med_depth=float(np.median(depth[lab==central]))
        radius_px=.047*f/max(.5,med_depth)
        near=(np.hypot(cell_xs-central_xy[0],cell_ys-central_xy[1])<=radius_px)
        if int(near.sum())>=45:
            cell_xs=cell_xs[near];cell_ys=cell_ys[near]
        u=float(cell_xs.mean());v=float(cell_ys.mean())
        ds=depth[cell_ys,cell_xs]
        d=float(np.median(ds))
        x,y=pixel_to_world_xy(u,v,d,depth.shape)
        if not (math.isfinite(x) and math.isfinite(y)):
            found.append(None)
            continue
        bb=[int(cell_xs.min()),int(cell_ys.min()),
            int(cell_xs.max()-cell_xs.min()+1),
            int(cell_ys.max()-cell_ys.min()+1)]
        found.append({'cell':cell,'pixel':[round(u,2),round(v,2)],
                      'xy':[round(x,5),round(y,5)],
                      'depth_m':round(d,5),'mask_area_px':len(cell_xs),
                      'bbox_px':bb})
    missed=[i for i,obj in enumerate(found) if obj is None]
    if missed:
        raise RuntimeError('Image localization missing food in tray cells '
                           f'{missed}; foreground counts per cell={counts}. '
                           'No ground-truth XY fallback was used. '
                           'Review localization_failed_* diagnostics.')
    return found,foreground

def draw_detection_overlay(rgb,detections):
    """Save the evidence of image-based center localization, not simulated dots."""
    canvas=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR).copy()
    for d in detections:
        u,v=map(int,map(round,d['pixel']))
        cv2.circle(canvas,(u,v),8,(15,225,255),2)
        cv2.circle(canvas,(u,v),2,(0,0,250),-1)
        label=f"C{d['cell']} XY({d['xy'][0]:.2f},{d['xy'][1]:.2f})"
        cv2.putText(canvas,label,(max(0,u-24),max(14,v-12)),
                    cv2.FONT_HERSHEY_SIMPLEX,.43,(10,25,40),3,cv2.LINE_AA)
        cv2.putText(canvas,label,(max(0,u-24),max(14,v-12)),
                    cv2.FONT_HERSHEY_SIMPLEX,.43,(255,245,155),1,cv2.LINE_AA)
    return canvas

class FoodCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(96,96,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(96,64),nn.ReLU(),nn.Linear(64,len(NAMES)))
    def forward(self,x):return self.net(x)

# Depth foreground measurement: avoid class-colour masks, which previously
# underestimated white large rice balls and overestimated yellow omelettes.
# Image centers still come from MuJoCo truth -- this is a controlled RGB-D
# segmentation experiment, NOT a learned object detector.
DEPTH_DELTA_M = 0.008
MIN_AREA_PX = 18
SAFE_MARGIN = 0.035

def crop_depth(depth,xy):
    """Identical fixed ROI transform to the RGB classifier, without quantizing depth."""
    h,w=depth.shape[:2]
    u,v=project_xy(*xy,depth.shape)
    radius=round(h*.095)
    padded=cv2.copyMakeBorder(np.asarray(depth,dtype=np.float32),radius,radius,
                               radius,radius,cv2.BORDER_CONSTANT,value=0)
    roi=padded[int(v):int(v+2*radius),int(u):int(u+2*radius)]
    if roi.size==0:raise ValueError(f'Depth crop failed for {xy}, pixel {(u,v)}')
    return cv2.resize(roi,(CROP,CROP),interpolation=cv2.INTER_NEAREST)

def depth_mask(depth,empty_depth,xy):
    """Connected RGB-D foreground against an empty calibrated workbench.

    No object index/mesh radius/true size is used in this measurement.
    The depth difference isolates food even when the colour is close to the
    background. The geometric position of the crop is simulator-provided.
    """
    foreground=crop_depth(depth,xy)
    empty=crop_depth(empty_depth,xy)
    valid=np.isfinite(foreground)&np.isfinite(empty)&(foreground>0)&(empty>0)
    mask=np.uint8(valid & ((empty-foreground)>DEPTH_DELTA_M))*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # Reject remote components from neighboring food or the robot.
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for k in range(1,n):
        area=stats[k,cv2.CC_STAT_AREA]
        cx,cy=cent[k]
        distance=float(np.hypot(cx-CROP/2,cy-CROP/2))
        if area>=MIN_AREA_PX and distance<CROP*.30:
            candidates.append((distance, -int(area),k))
    if not candidates:return np.zeros((CROP,CROP),dtype=np.uint8)
    # A component that actually covers the centre is preferred; otherwise
    # choose the nearest substantial component.
    center_label=int(lab[CROP//2,CROP//2])
    chosen=center_label if center_label and any(k==center_label for _,_,k in candidates) else min(candidates)[2]
    return (lab==chosen).astype(np.uint8)*255

def depth_measurement(depth,empty_depth,xy):
    m=depth_mask(depth,empty_depth,xy)
    contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return 0.,0.
    contour=max(contours,key=cv2.contourArea)
    area=float(cv2.contourArea(contour))
    rect=cv2.minAreaRect(contour)
    width=max(float(rect[1][0]),float(rect[1][1]))
    return area,width

def calibrate_sizes(sim):
    """Known-size reference item for each class; no evaluation item scale used.

    Each reference is physically rendered in isolation, then normalised by
    its documented calibration scale.  The assessed item is measured from
    depth foreground pixels only.
    """
    reference={}
    for cls in range(len(NAMES)):
        idx=next(i for i,c in enumerate(ITEM_CLASSES) if c==cls)
        xy=(.28,-.25)
        sim.train_view(idx,xy,0.0)
        dep=sim.render_depth();empty=sim.render_empty_depth()
        area,width=depth_measurement(dep,empty,xy)
        if area<MIN_AREA_PX or width<3:
            raise RuntimeError(f'Depth calibration failed for class {cls}: area={area}, width={width}; inspect 3D camera')
        scale=float(ITEM_SCALES[idx]) # CALIBRATION exemplar only
        reference[cls]={'area_at_1x':area/(scale**2),'width_at_1x':width/scale,
                        'reference_item':int(idx),'reference_scale':scale}
    return reference

def estimate_size(depth,empty_depth,xy,cls,reference):
    """Pixel-derived linear size; none of the assessed object's truth is used."""
    r=reference.get(int(cls))
    if r is None:return float('nan')
    area,width=depth_measurement(depth,empty_depth,xy)
    if area<MIN_AREA_PX or width<3:return float('nan')
    area_scale=float(np.sqrt(area/r['area_at_1x']))
    width_scale=float(width/r['width_at_1x'])
    # If silhouette width and area disagree sharply, do not silently accept.
    if not (np.isfinite(area_scale) and np.isfinite(width_scale)):
        return float('nan')
    if max(area_scale,width_scale)/max(min(area_scale,width_scale),1e-6)>1.65:
        return float('nan')
    return float(.7*area_scale+.3*width_scale)

def validate_size_camera(sim,reference,output):
    """Check all available food examples without passing their truth to prediction."""
    import csv
    rows=[]
    xy=(.28,-.25)
    for idx,cls in enumerate(ITEM_CLASSES):
        for yaw in (0.0,.65):
            sim.train_view(idx,xy,yaw)
            dep=sim.render_depth();empty=sim.render_empty_depth()
            est=estimate_size(dep,empty,xy,cls,reference)
            if yaw==0.0:
                cv2.imwrite(str(Path(output).parent/f'depth_mask_item_{idx}.png'),depth_mask(dep,empty,xy))
            true=float(ITEM_SCALES[idx])
            rows.append({'food_id':idx,'class':cls,'yaw_rad':yaw,
                         'true_scale':true,'estimated_scale':round(est,4) if np.isfinite(est) else '',
                         'absolute_error':round(abs(est-true),4) if np.isfinite(est) else '',
                         'true_in_spec':bool(cls in (0,1,2,3) and true<=SIZE_LIMITS[cls]),
                         'estimated_in_spec':bool(cls in (0,1,2,3) and np.isfinite(est) and est<=SIZE_LIMITS[cls]-SAFE_MARGIN)})
    with open(output,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    valid=[r for r in rows if r['estimated_scale']!='']
    mae=sum(r['absolute_error'] for r in valid)/max(1,len(valid))
    count=sum(r['true_in_spec']==r['estimated_in_spec'] for r in rows)
    report={'evaluated':len(rows),'valid_measurements':len(valid),
            'mae_scale':round(mae,4),'size_spec_accuracy':round(count/len(rows),4)}
    print('RGB-D size self-check (isolated views):',report,flush=True)
    return report

def render_dataset(sim,n,seed,augment):
    rng=np.random.default_rng(seed)
    xs=[];ys=[]
    for k in range(n):
        cls=k%len(NAMES)
        options=[i for i,c in enumerate(ITEM_CLASSES) if c==cls]
        i=int(rng.choice(options))
        x=float(rng.uniform(.11,.415));y=float(rng.uniform(-.386,-.164))
        yaw=float(rng.uniform(-2.8,2.8))
        image=sim.train_view(i,(x,y),yaw)
        roi=image_crop(image,(x,y))
        if augment:
            # Training scale jitter makes the 5-class CNN robust to nonuniform food size.
            zoom=float(rng.uniform(.82,1.22))
            M=cv2.getRotationMatrix2D((CROP/2,CROP/2),0,zoom)
            roi=cv2.warpAffine(roi,M,(CROP,CROP),borderMode=cv2.BORDER_REFLECT)
            brightness=float(rng.uniform(.75,1.24))
            roi=np.uint8(np.clip(roi.astype(np.float32)*brightness+rng.normal(0,2.5,roi.shape),0,255))
            if rng.random()<.6:
                dx,dy=rng.integers(-4,5,size=2)
                M=np.float32([[1,0,dx],[0,1,dy]])
                roi=cv2.warpAffine(roi,M,(CROP,CROP),borderMode=cv2.BORDER_REFLECT)
        xs.append(roi);ys.append(cls)
        if (k+1)%100==0:print(f'3D camera dataset: {k+1}/{n}',flush=True)
    return np.stack(xs),np.array(ys,np.int64)

def train(sim,out,device,train_n=700,val_n=150,epochs=10,seed=31031):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.random.seed(seed);torch.manual_seed(seed);random.seed(seed)
    x,y=render_dataset(sim,train_n,seed,True)
    xv,yv=render_dataset(sim,val_n,seed+9001,False)
    def loader(X,Y,shuffle):
        tensor=torch.from_numpy(X.copy()).permute(0,3,1,2).float()/255
        return DataLoader(TensorDataset(tensor,torch.from_numpy(Y)),batch_size=48,
                          shuffle=shuffle,num_workers=0)
    tr,vl=loader(x,y,True),loader(xv,yv,False)
    model=FoodCNN().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.0001)
    history=[];best=-1
    for ep in range(1,epochs+1):
        model.train();ls=[]
        for xx,yy in tr:
            xx,yy=xx.to(device),yy.to(device)
            pred=model(xx);loss=nn.functional.cross_entropy(pred,yy)
            opt.zero_grad();loss.backward();opt.step();ls.append(float(loss.item()))
        model.eval();hits=0;total=0
        with torch.inference_mode():
            for xx,yy in vl:
                p=model(xx.to(device)).argmax(1).cpu()
                hits+=int((p==yy).sum());total+=len(yy)
        acc=hits/max(1,total)
        history.append({'epoch':ep,'train_loss':round(float(np.mean(ls)),6),'validation_accuracy':round(acc,5)})
        print(f'epoch {ep}/{epochs} loss={np.mean(ls):.4f}, val food class accuracy={acc:.1%}',flush=True)
        if acc>best:
            best=acc;torch.save({'weights':model.state_dict(),'classes':NAMES,'seed':seed},out/'food_cnn.pt')
    with (out/'training_history.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,history[0].keys());w.writeheader();w.writerows(history)
    model.load_state_dict(torch.load(out/'food_cnn.pt',map_location='cpu',weights_only=True)['weights'])
    model.eval()
    # Held-out validation confusion matrix for transparent per-food error analysis.
    cm=np.zeros((len(NAMES),len(NAMES)),dtype=int)
    with torch.inference_mode():
        for xx,yy in vl:
            guesses=model(xx.to(device)).argmax(1).cpu().numpy()
            for true,pred in zip(yy.numpy(),guesses):
                cm[int(true),int(pred)]+=1
    with (out/'confusion_matrix.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['true_class / predicted_class']+NAMES)
        for label,row in zip(NAMES,cm):w.writerow([label]+row.tolist())
    # The image shows numeric counts and does not alter the CNN predictions.
    cell=70;left=142;top=100;canvas=np.full((top+cell*len(NAMES)+45,left+cell*len(NAMES)+15,3),245,np.uint8)
    maxv=max(1,int(cm.max()))
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            val=int(cm[i,j]);shade=int(238-175*val/maxv)
            cv2.rectangle(canvas,(left+j*cell,top+i*cell),(left+(j+1)*cell-2,top+(i+1)*cell-2),(shade,235,225),-1)
            cv2.putText(canvas,str(val),(left+j*cell+13,top+i*cell+40),cv2.FONT_HERSHEY_SIMPLEX,.85,(25,45,50),2)
        cv2.putText(canvas,str(i),(96,top+i*cell+40),cv2.FONT_HERSHEY_SIMPLEX,.75,(25,45,50),2)
        cv2.putText(canvas,str(i),(left+i*cell+27,81),cv2.FONT_HERSHEY_SIMPLEX,.75,(25,45,50),2)
    cv2.putText(canvas,'CNN VAL CONFUSION / class ids 0-4',(8,33),cv2.FONT_HERSHEY_SIMPLEX,.63,(25,45,50),2)
    cv2.imwrite(str(out/'confusion_matrix.png'),canvas)
    return model

def predict(model,roi,device):
    x=torch.from_numpy(roi.copy()).permute(2,0,1).float()[None].to(device)/255
    model.eval()
    with torch.inference_mode():
        p=model(x)[0].softmax(0)
    c=int(p.argmax());return c,float(p[c])


# -----------------------------------------------------------------------------
# 6-slot placement quality: RGB-D isolated-item classification with TTA and
# calibrated colour evidence. All prototypes are obtained from labelled
# SIMULATED TRAINING views; never from test-item ground truth or evaluation IDs.
# -----------------------------------------------------------------------------
def color_signature(roi, mask):
    """Lighting-tolerant chromaticity+sat. from measured food foreground."""
    a=np.asarray(roi,dtype=np.uint8)
    m=np.asarray(mask)>0
    if a.ndim!=3 or a.shape[-1]!=3 or a.shape[:2]!=m.shape:
        return None
    pixels=a[m].astype(np.float32)/255.0
    if len(pixels)<30:
        return None
    # Avoid dark decorative pieces (nori, omelet shadow) dominating the result.
    bright=np.max(pixels,axis=1)
    pixels=pixels[bright>=np.quantile(bright,.25)]
    if len(pixels)<20:return None
    chrom=pixels/(pixels.sum(axis=1,keepdims=True)+1e-6)
    median_chrom=np.median(chrom,axis=0)
    hsv=cv2.cvtColor(np.asarray(roi,dtype=np.uint8),cv2.COLOR_RGB2HSV)
    saturation=np.median(hsv[m,1].astype(np.float32))/255.0
    return np.asarray([*median_chrom, saturation],dtype=np.float64)


def calibrate_color_prototypes(sim):
    """Create 5 labelled colour references from isolated TRAINING renderings.

    The test scene's true class/size/position is NEVER used by the classifier.
    This calibration is compatible with a previously trained CNN checkpoint.
    """
    prototypes={}
    xy=(.28,-.25)
    for cls in range(len(NAMES)):
        idx=next(i for i,c in enumerate(ITEM_CLASSES) if c==cls)
        feats=[]
        for yaw in (-.7,0.,.85):
            im=sim.train_view(idx,xy,yaw)
            d=sim.render_depth()
            empty=sim.render_empty_depth()
            roi=image_crop(im,xy)
            mask=depth_mask(d,empty,xy)
            feat=color_signature(roi,mask)
            if feat is not None:feats.append(feat)
        if not feats:
            raise RuntimeError(f'Colour calibration failed for class {cls}')
        prototypes[cls]=np.mean(feats,axis=0).tolist()
    return prototypes


def clean_neighbor_pixels(roi, full_mask, target_mask, u, v):
    """Remove only FOREIGN food foreground from an RGB ROI.

    Keeps target food, source tray and background; does not use food IDs/labels.
    A 7x7 guard around the target depth silhouette prevents erasing its edges.
    """
    all_crop=image_crop_pixel(np.repeat(full_mask[...,None],3,axis=2),u,v)[:,:,0]
    target=np.asarray(target_mask,dtype=np.uint8)
    keep=cv2.dilate(target,np.ones((7,7),np.uint8),iterations=1)>0
    foreign=(all_crop>60)&(~keep)
    cleaned=roi.copy()
    cleaned[foreign]=[64,81,87]  # neutral supply-tray RGB, not a class cue
    return cleaned


def predict_robust(model,roi,mask,device,prototypes,return_debug=False):
    """Softmax ensemble + calibrated colour, for CNN + RGB-D inference.

    Same model weights as before. Does not use truth labels of the assessed item.
    Only an image ROI and its measured depth foreground are inputs.
    """
    shift_xy=[(0,0),(3,0),(-3,0),(0,3),(0,-3)]
    imgs=[]
    for dx,dy in shift_xy:
        M=np.float32([[1,0,dx],[0,1,dy]])
        imgs.append(cv2.warpAffine(roi,M,(roi.shape[1],roi.shape[0]),
                                    borderMode=cv2.BORDER_REFLECT_101))
    batch=torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).float().to(device)/255.
    model.eval()
    with torch.inference_mode():
        probabilities=model(batch).softmax(dim=1).mean(dim=0).cpu().numpy()
    cnn_probs=np.asarray(probabilities,dtype=float)
    cnn_probs/=cnn_probs.sum()
    signature=color_signature(roi,mask)
    color_probs=np.ones(len(NAMES),dtype=float)/len(NAMES)
    if signature is not None:
        ref=np.asarray([prototypes[i] for i in range(len(NAMES))],dtype=float)
        # Colour differences normalised independently per channel.
        std=np.array([.09,.09,.09,.18],dtype=float)
        dist2=np.square((ref-signature[None,:])/std[None,:]).sum(axis=1)
        color_probs=np.exp(-.5*(dist2-dist2.min()))
        color_probs/=max(color_probs.sum(),1e-10)
        # Trust colour only when its calibrated prototype fit is distinctive.
        weight=.65 if color_probs.max()>=.80 else .55 if color_probs.max()>=.65 else .30 if color_probs.max()>=.50 else .10
    else:weight=0.
    fused=(1-weight)*cnn_probs+weight*color_probs
    result=int(fused.argmax());conf=float(fused[result])
    debug={'cnn_class':int(cnn_probs.argmax()),
           'cnn_conf':round(float(cnn_probs.max()),4),
           'color_class':int(color_probs.argmax()),
           'color_conf':round(float(color_probs.max()),4),
           'color_weight':round(float(weight),3),
           'selected_class':result,'selected_conf':round(conf,4)}
    return (result,conf,debug) if return_debug else (result,conf)
