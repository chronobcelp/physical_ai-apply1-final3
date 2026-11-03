"""3D MuJoCo bento robot with SCARA arm and three-finger crane claw.
The 3 fingers are articulated by hinge joints and visibly close around food.
Attachment is kinematic after closing; this is NOT validated frictional gripping.
"""
import math, os, xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco

CLAW_OPEN = -0.24   # three fingers splay outward
CLAW_CLOSED = 0.43  # three fingers fold inward

NAMES=['ONIGIRI','TAMAGO','BROCCOLI','KARAAGE','TOMATO / FOREIGN']
ITEM_CLASSES=[0,1,2,3,0,4,1,3,0,1]
# Every object is a genuine differently-sized MuJoCo geometry (not video postprocessing).
ITEM_SCALES=(1.00,1.00,1.00,1.00,1.55,1.00,1.50,1.00,1.00,1.00)
# Maximum relative size allowed by the recipe/presentation specification per compartment.
# This is a quality standard, not a simulated hard collision limit.
SIZE_LIMITS={0:1.23,1:1.24,2:1.30,3:1.30}
SOURCE_SURFACE_Z=.051
BENTO_SURFACE_Z=.070

def source_height(i):
    return SOURCE_SURFACE_Z + .031*ITEM_SCALES[i] + .003

def destination_height(i, reject=False):
    return (SOURCE_SURFACE_Z if reject else BENTO_SURFACE_Z) + .031*ITEM_SCALES[i] + .003

def size_fits(cls, scale):
    return cls in SIZE_LIMITS and scale <= SIZE_LIMITS[cls]

# 2-row x 3-column ordered recipe. Compartment ID and food class are distinct.
# The CNN still recognises the same five classes, so previous checkpoints remain compatible.
RECIPE=(0,1,2,3,0,1)  # rice, omelet, broccoli, chicken, rice, omelet
ORDER=tuple(range(len(RECIPE)))
SLOT_CLASSES={slot:cls for slot,cls in enumerate(RECIPE)}
SLOTS={
    0:(.155,.126,.101), 1:(.310,.126,.101), 2:(.465,.126,.101),
    3:(.155,.338,.101), 4:(.310,.338,.101), 5:(.465,.338,.101),
}
# Nine separate food positions; the farthest is still within SCARA's 0.58m reach.
SOURCE=[(.120,-.150),(.265,-.150),(.395,-.150),
        (.120,-.275),(.265,-.275),(.395,-.275),
        (.120,-.385),(.265,-.385),(.395,-.385)]
REJECT=(.065,-.452,.082)
# Episode 1 deliberately lacks broccoli, and two oversize ingredients must be rejected.
# Each scene contains 9 physical items: at least six recipe-relevant ingredients
# except the intentionally missing one; source order differs per episode.
EPISODES=[
    (0,1,2,3,4,5,6,8,9),
    (4,1,3,8,5,6,9,0,7),
    (9,6,2,3,8,4,0,7,1),
    (8,5,6,4,2,9,3,0,1),
]

def quat(yaw):return np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)],dtype=float)

def food_geoms(kind):
    if kind==0:  # rice ball + nori
        return '''<geom type="ellipsoid" size=".035 .031 .028" rgba=".96 .96 .88 1" mass=".09"/>
        <geom type="box" pos=".004 -.018 .024" size=".021 .011 .007" rgba=".07 .14 .12 1" contype="0" conaffinity="0"/>'''
    if kind==1:  # three omelet rolls
        x='''<geom type="box" size=".035 .028 .024" rgba=".96 .72 .15 1" mass=".09"/>'''
        for i in range(3):
            x+=f'<geom type="box" pos="{-.023+i*.023:.4f} 0 .024" size=".009 .027 .005" rgba=".98 .88 .36 1" contype="0" conaffinity="0"/>'
        return x
    if kind==2:
        x='''<geom type="cylinder" size=".014 .022" rgba=".40 .68 .17 1" mass=".09"/>'''
        for k,(dx,dy) in enumerate([(-.019,-.010),(.018,-.007),(-.006,.018),(.012,.019),(0,0)]):
            x+=f'<geom type="sphere" pos="{dx} {dy} .022" size=".018" rgba=".11 {(.47+.05*(k%3)):.3f} .17 1" contype="0" conaffinity="0"/>'
        return x
    if kind==3:
        x='''<geom type="ellipsoid" size=".032 .027 .027" rgba=".67 .33 .11 1" mass=".09"/>'''
        for dx,dy,dz,r in [(-.019,.008,.015,.014),(.017,.008,.018,.016),(.003,-.018,.019,.014),(-.006,.009,.030,.012)]:
            x+=f'<geom type="sphere" pos="{dx} {dy} {dz}" size="{r}" rgba=".80 .46 .18 1" contype="0" conaffinity="0"/>'
        return x
    return '''<geom type="sphere" size=".030" rgba=".88 .08 .09 1" mass=".09"/>
       <geom type="box" pos="0 0 .030" size=".019 .006 .003" rgba=".16 .61 .17 1" contype="0" conaffinity="0"/>
       <geom type="box" pos="0 0 .030" euler="0 0 1.57" size=".019 .006 .003" rgba=".16 .61 .17 1" contype="0" conaffinity="0"/>'''

def scaled_food_geoms(kind, scale):
    """Scale MuJoCo geom size, positions, capsule endpoints and approximate masses."""
    frag=ET.fromstring('<root>'+food_geoms(kind)+'</root>')
    for geom in frag:
        for attr in ('size','pos','fromto'):
            val=geom.get(attr)
            if val:
                nums=[float(x)*scale for x in val.split()]
                geom.set(attr, ' '.join(f'{v:.7f}' for v in nums))
        if geom.get('mass'):
            geom.set('mass',f'{float(geom.get("mass"))*scale**3:.7f}')
    return ''.join(ET.tostring(g,encoding='unicode') for g in frag)

def scene_xml():
    foods=[]
    for i,cls in enumerate(ITEM_CLASSES):
        foods.append(f'''<body name="food_{i}" pos="{.13+.04*i:.3f} -.25 .080"><freejoint name="food_joint_{i}"/>
           {scaled_food_geoms(cls,ITEM_SCALES[i])}</body>''')
    # Three articulated fingers arranged around the vertical tool axis.
    # The hinges are real MuJoCo joints; gripping attachment is idealized.
    finger_bodies=[]
    for k in range(3):
        az=2*math.pi*k/3
        px=.059*math.cos(az);py=.059*math.sin(az)
        finger_bodies.append(f'''<body name="claw_finger_{k}" pos="{px:.6f} {py:.6f} -.048" euler="0 0 {az:.7f}">
          <joint name="claw_joint_{k}" type="hinge" axis="0 1 0" range="-.27 .56" limited="true" damping=".06"/>
          <geom type="sphere" size=".015" rgba=".14 .24 .33 1" mass=".007" contype="0" conaffinity="0"/>
          <geom type="capsule" fromto="0 0 -.003 0 0 -.079" size=".009" rgba=".98 .69 .12 1" mass=".024" contype="0" conaffinity="0"/>
          <geom type="sphere" pos="0 0 -.079" size=".013" rgba=".20 .36 .39 1" mass=".005" contype="0" conaffinity="0"/>
         </body>''')
    claw_xml=''.join(finger_bodies)
    divs='''<geom name="bento_bottom" type="box" pos=".3105 .233 .058" size=".224 .209 .011" rgba=".87 .70 .42 1"/>
    <geom name="bento_wall_l" type="box" pos=".082 .233 .103" size=".006 .216 .048" rgba=".65 .42 .19 1"/>
    <geom name="bento_wall_r" type="box" pos=".539 .233 .103" size=".006 .216 .048" rgba=".65 .42 .19 1"/>
    <geom name="bento_wall_front" type="box" pos=".3105 .018 .103" size=".235 .006 .048" rgba=".65 .42 .19 1"/>
    <geom name="bento_wall_back" type="box" pos=".3105 .447 .103" size=".235 .006 .048" rgba=".65 .42 .19 1"/>
    <geom name="divider_x_left" type="box" pos=".233 .233 .080" size=".005 .210 .026" rgba=".56 .38 .20 1"/>
    <geom name="divider_x_right" type="box" pos=".387 .233 .080" size=".005 .210 .026" rgba=".56 .38 .20 1"/>
    <geom name="divider_y" type="box" pos=".3105 .233 .080" size=".224 .006 .026" rgba=".56 .38 .20 1"/>'''
    return f'''<mujoco model="Bento_Lab_3D"><compiler angle="radian"/>
    <option timestep=".005" gravity="0 0 -9.81" integrator="implicitfast"/>
    <visual><global offwidth="1024" offheight="768"/><quality offsamples="4"/></visual>
    <worldbody>
     <light name="key" pos=".0 -.25 2.0" dir="0 0 -1" directional="true" diffuse=".95 .95 .95"/>
     <light name="fill" pos=".65 .5 1.5" dir="-.3 -.4 -1" diffuse=".45 .48 .5"/>
     <camera name="top" pos=".30 0 1.36" quat="1 0 0 0" fovy="43"/>
     <!-- Matte studio surface with no grid; z-offset avoids planar overlap with table. -->
     <geom name="floor" type="plane" pos="0 0 -.045" size="2 2 .1" rgba=".91 .93 .95 1" contype="0" conaffinity="0"/>
     <geom name="table" type="box" pos=".30 -.01 .023" size=".53 .57 .023" rgba=".14 .21 .28 1"/>
     <geom name="ingredients_mat" type="box" pos=".30 -.30 .047" size=".25 .20 .002" rgba=".26 .34 .37 1" contype="0" conaffinity="0"/>
     {divs}
     <geom name="reject_tray" type="box" pos=".065 -.452 .053" size=".067 .073 .007" rgba=".74 .15 .19 1"/>
     <geom name="reject_back" type="box" pos=".065 -.523 .079" size=".066 .005 .030" rgba=".60 .10 .15 1"/>
     <body name="robot_base" pos="0 0 .430">
      <joint name="base" type="hinge" axis="0 0 1" range="-3.1 3.1" damping=".5"/>
      <geom name="hub" type="cylinder" size=".064 .044" rgba=".12 .19 .32 1"/>
      <geom name="arm_a" type="capsule" fromto="0 0 0 .30 0 0" size=".029" rgba=".21 .67 .86 1"/>
      <body name="elbow" pos=".30 0 0">
       <joint name="elbow_joint" type="hinge" axis="0 0 1" range="-2.9 2.9" damping=".3"/>
       <geom type="sphere" size=".044" rgba=".10 .22 .33 1"/>
       <geom name="arm_b" type="capsule" fromto="0 0 0 .28 0 0" size=".025" rgba=".13 .77 .79 1"/>
       <body name="wrist" pos=".28 0 0">
        <joint name="wrist_joint" type="hinge" axis="0 0 1" range="-6.2 6.2" damping=".12"/>
        <geom type="sphere" size=".034" rgba=".12 .25 .36 1"/>
        <body name="lift"><joint name="lift_slide" type="slide" axis="0 0 1" range="-.39 .02" damping="1"/>
         <geom name="crane_cable" type="cylinder" pos="0 0 -.029" size=".010 .045" rgba=".83 .88 .91 1"/>
         <geom name="crane_claw_hub" type="cylinder" pos="0 0 -.067" size=".035 .018" rgba=".16 .27 .38 1"/>
         <geom name="crane_claw_ring" type="cylinder" pos="0 0 -.082" size=".026 .004" rgba=".96 .71 .16 1" contype="0" conaffinity="0"/>
         <site name="tool_tip" pos="0 0 -.105" size=".006" rgba="1 .82 .2 .4"/>
         {claw_xml}
        </body>
       </body>
      </body>
     </body>
     {''.join(foods)}
    </worldbody></mujoco>'''

class BentoSim:
    L1=.30;L2=.28
    def __init__(self, size=(720,480), seed=31031):
        self.model=mujoco.MjModel.from_xml_string(scene_xml())
        self.data=mujoco.MjData(self.model)
        self.w,self.h=size
        self.renderer=None
        self.depth_renderer=None
        self.seed=seed
        self.joints=['base','elbow_joint','wrist_joint','lift_slide']
        self.qadrs=[int(self.model.jnt_qposadr[self.model.joint(j).id]) for j in self.joints]
        self.fadrs=[int(self.model.jnt_qposadr[self.model.joint(f'food_joint_{i}').id]) for i in range(len(ITEM_CLASSES))]
        self.tool=self.model.site('tool_tip').id
        self.claw_joints=[f'claw_joint_{k}' for k in range(3)]
        self.claw_qadrs=[int(self.model.jnt_qposadr[self.model.joint(j).id]) for j in self.claw_joints]
        self.claw_state=CLAW_OPEN
        self.attached=None
        self.phase="IDLE"
        self.active=[]
        self.reset(0,seed)
    def set_food(self,i,xyz,yaw=0.):
        a=self.fadrs[i]
        self.data.qpos[a:a+3]=xyz
        self.data.qpos[a+3:a+7]=quat(yaw)
        j=self.model.joint(f'food_joint_{i}').id
        da=int(self.model.jnt_dofadr[j]);self.data.qvel[da:da+6]=0
    def food_pose(self,i):
        a=self.fadrs[i];p=self.data.qpos[a:a+7]
        return np.array(p[:3]),float(2*math.atan2(float(p[6]),float(p[3])))
    def reset(self,episode=0,seed=31031):
        rng=np.random.default_rng(seed+episode*13)
        mujoco.mj_resetData(self.model,self.data)
        self.active=list(EPISODES[episode%len(EPISODES)])
        for i in range(len(ITEM_CLASSES)):
            self.set_food(i,(2+i*.1,2.0,.08),0)
        for k,i in enumerate(self.active):
            sx,sy=SOURCE[k]
            self.set_food(i,(sx+rng.uniform(-.012,.012),sy+rng.uniform(-.012,.012),source_height(i)),rng.uniform(-1.4,1.4))
        self.set_joints(0.0,1.2,-1.2,-.16)
        self.set_claw(CLAW_OPEN)
        mujoco.mj_forward(self.model,self.data)
        self.attached=None
        self.phase="READY"
    def set_joints(self,*q):
        for name,adr,angle in zip(self.joints,self.qadrs,q):
            rng=self.model.joint(name).range
            self.data.qpos[adr]=float(np.clip(angle,rng[0],rng[1]))
    def set_claw(self, angle, steps=1, callback=None):
        """Animate all three finger joints opening/closing in synchrony.

        MuJoCo renders real hinge rotations, but no force-based gripping is solved.
        """
        start=float(self.claw_state)
        for step in range(1,steps+1):
            t=step/steps
            t=t*t*(3-2*t)
            a=start+(float(angle)-start)*t
            for addr in self.claw_qadrs:
                self.data.qpos[addr]=a
            mujoco.mj_forward(self.model,self.data)
            if callback and (step%2==0 or step==steps):
                callback(self)
        self.claw_state=float(angle)

    def ik(self,x,y,z,yaw=0.):
        d2=x*x+y*y
        if math.sqrt(d2)>self.L1+self.L2-.002 or math.sqrt(d2)<abs(self.L1-self.L2)+.002:
            raise ValueError(f'XY target unreachable: {(x,y,z)}')
        c=float(np.clip((d2-self.L1**2-self.L2**2)/(2*self.L1*self.L2),-1.,1.))
        e=math.acos(c)
        b=math.atan2(y,x)-math.atan2(self.L2*math.sin(e),self.L1+self.L2*c)
        w=((yaw-b-e+math.pi)%(2*math.pi))-math.pi
        slide=z-.430+.105
        if not -.39<=slide<=.02:raise ValueError(f'Z target unreachable: {z}')
        return np.array([b,e,w,slide])
    def move(self,xyz,yaw=0.,steps=24,callback=None):
        target=self.ik(*xyz,yaw)
        start=np.array([self.data.qpos[a] for a in self.qadrs])
        for k in range(1,steps+1):
            t=k/steps;t=t*t*(3-2*t)
            q=start+(target-start)*t
            self.set_joints(*q);mujoco.mj_forward(self.model,self.data)
            if self.attached is not None:
                tip=self.data.site_xpos[self.tool]
                self.set_food(self.attached,(float(tip[0]),float(tip[1]),float(tip[2]-.030)),float(sum(q[:3])))
                mujoco.mj_forward(self.model,self.data)
            if callback and (k%3==0 or k==steps):callback(self)
    def pick(self,i,callback=None,xy_override=None):
        """Open > descend > close three claws > attach > lift.
        The attached food is moved kinematically rather than by contact forces.
        """
        pos,angle=self.food_pose(i)
        # AI control uses XY recovered from the overhead depth image. The item ID
        # and yaw are still provided by the simulator's object-execution adapter.
        if xy_override is None:
            x,y=float(pos[0]),float(pos[1])
        else:
            x,y=map(float,xy_override)
        yaw=float(angle)
        self.phase='OPEN'
        self.set_claw(CLAW_OPEN,steps=9,callback=callback)
        self.phase='APPROACH'
        self.move((x,y,.245),yaw,callback=callback)
        self.phase='LOWER'
        self.move((x,y,max(.115,source_height(i)+.035)),yaw,steps=26,callback=callback)
        self.phase='CLOSE'
        self.set_claw(CLAW_CLOSED,steps=20,callback=callback)
        # Do not count a wildly misplaced IMAGE-BASED grasp as successful.
        # This geometry-only guard checks executor feasibility, not AI planning.
        actual,_ = self.food_pose(i)
        loc_error = math.hypot(x-float(actual[0]),y-float(actual[1]))
        if loc_error > .029:
            raise RuntimeError(f'Image localization too far for kinematic claw pick: '
                               f'item={i} position_error={loc_error:.3f} m. '
                               'Review detection overlay; no truth XY fallback.')
        # Visual claw is articulated; grasp attachment remains an idealized constraint.
        self.attached=i
        self.phase='LIFT'
        self.move((x,y,.252),yaw,callback=callback)
    def place(self,i,target,callback=None):
        """Carry > lower > open claws > release > rise."""
        if self.attached != i:raise RuntimeError('Cannot place ungrasped object')
        x,y,z=target
        self.phase='TRANSPORT'
        self.move((x,y,.252),0,callback=callback)
        self.phase='PLACE_LOWER'
        self.move((x,y,z+.030),0,steps=24,callback=callback)
        self.attached=None
        self.set_food(i,(x,y,z),0)
        mujoco.mj_forward(self.model,self.data)
        self.phase='RELEASE'
        self.set_claw(CLAW_OPEN,steps=20,callback=callback)
        self.phase='RETRACT'
        self.move((x,y,.252),0,steps=18,callback=callback)
        self.phase='READY'
    def _clean_video_render(self):
        """Switch off shadow-map streaks and debug overlays in output frames only.

        The overhead view used to train and evaluate CNN is not altered.
        No object pixels are erased or retouched after rendering.
        """
        scene = self.renderer.scene
        flag_names = ('mjRND_SHADOW', 'mjRND_REFLECTION', 'mjRND_SKYBOX',
                      'mjRND_FOG', 'mjRND_WIREFRAME')
        for flag_name in flag_names:
            flag = getattr(mujoco.mjtRndFlag, flag_name, None)
            if flag is not None:
                scene.flags[int(flag)] = 0

    def render(self,top=False):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,self.h,self.w)
        if top:self.renderer.update_scene(self.data,camera='top')
        else:
            cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:]=[.29,-.028,.13];cam.distance=1.12;cam.azimuth=129;cam.elevation=-52
            self.renderer.update_scene(self.data,camera=cam)
            self._clean_video_render()
        return np.asarray(self.renderer.render()).copy()
    def render_depth(self):
        """Metric top-camera depth in metres, for vision-only size measurement.
        The depth rendering is independent of the beauty camera; no object IDs
        or ITEM_SCALES are read to infer an inspected item's scale.
        """
        if self.depth_renderer is None:
            self.depth_renderer=mujoco.Renderer(self.model,self.h,self.w)
            self.depth_renderer.enable_depth_rendering()
        self.depth_renderer.update_scene(self.data,camera='top')
        return np.asarray(self.depth_renderer.render(),dtype=np.float32).copy()

    def render_empty_depth(self):
        """Clean-background depth calibration, equivalent to observing an empty tray.
        Simulator pose is used only to temporarily clear the scene for this
        background reference; original MuJoCo state is restored exactly.
        """
        qpos=self.data.qpos.copy()
        qvel=self.data.qvel.copy()
        try:
            for i in range(len(ITEM_CLASSES)):
                self.set_food(i,(2.0+i*.1,2.0,.08),0)
            mujoco.mj_forward(self.model,self.data)
            return self.render_depth()
        finally:
            self.data.qpos[:]=qpos
            self.data.qvel[:]=qvel
            mujoco.mj_forward(self.model,self.data)

    def render_claw(self):
        """Real MuJoCo close-up camera tracking the claw tip (not a fake zoom)."""
        if self.renderer is None:
            self.renderer=mujoco.Renderer(self.model,self.h,self.w)
        tip=self.data.site_xpos[self.tool]
        cam=mujoco.MjvCamera()
        cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=[float(tip[0]),float(tip[1]),float(tip[2]-.030)]
        cam.distance=.28
        cam.azimuth=127
        cam.elevation=-32
        self.renderer.update_scene(self.data,camera=cam)
        self._clean_video_render()
        return np.asarray(self.renderer.render()).copy()

    def train_view(self,i,xy,yaw):
        for j in range(len(ITEM_CLASSES)):self.set_food(j,(2+j*.1,2.,.08),0)
        self.set_food(i,(float(xy[0]),float(xy[1]),source_height(i)),yaw)
        mujoco.mj_forward(self.model,self.data)
        return self.render(top=True)
    def slot_class(self, slot):
        return SLOT_CLASSES[int(slot)]

    def size_label(self,i):
        val=ITEM_SCALES[i]
        return ('特大' if val>1.32 else '普通')+f' ({val:.2f}x)'
    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
        if self.depth_renderer is not None:self.depth_renderer.close();self.depth_renderer=None
