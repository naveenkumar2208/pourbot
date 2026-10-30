"""Offline color/depth localization preview; never commands hardware."""
import json
from pathlib import Path
import cv2
import numpy as np
root=Path(__file__).resolve().parent.parent
im=cv2.imread(str(root/'captures/flask-clear.jpg'))
b=(root/'captures/flask-clear.dep').read_bytes();w=int.from_bytes(b[8:16],'big');h=int.from_bytes(b[16:24],'big')
d=np.frombuffer(b[24:],dtype='>u2').reshape(h,w).astype(float)
meta=json.loads((root/'captures/flask-clear.json').read_text());k=meta['intrinsics']
y,x=np.indices(d.shape);xyz=np.stack(((x-k['centerXPx'])*d/k['focalXPx'],(y-k['centerYPx'])*d/k['focalYPx'],d),axis=-1)
mask=(x>200)&(x<800)&(y>260)&(y<680)&(d>300)&(d<1500)&~((x>415)&(x<575)&(y<450))
pts=xyz[mask][::5];rng=np.random.default_rng(9);best=None
for _ in range(200):
 q=pts[rng.choice(len(pts),3,replace=False)];n=np.cross(q[1]-q[0],q[2]-q[0]);l=np.linalg.norm(n)
 if l<1e-6:continue
 n/=l;c=-n@q[0];inliers=np.abs(pts@n+c)<4
 if best is None or inliers.sum()>best[0]:best=(inliers.sum(),n,c,inliers)
p=pts[best[3]];center=p.mean(0);_,_,vt=np.linalg.svd(p-center,full_matrices=False);n=vt[-1];n=n if n[2]<0 else -n;c=-n@center
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);blue=cv2.inRange(hsv,np.array([85,95,35]),np.array([135,255,255]))>0
roi=blue&(x>435)&(x<535)&(y>245)&(y<345);valid=roi&(d>0);p=xyz[valid];heights=p@n+c
report={'status':'PERCEPTION_PREVIEW_NOT_ROBOT_TARGETS','table_plane_camera_normal':n.tolist(),'table_plane_camera_offset_mm':float(c),'table_fit_inlier_fraction':float(best[0]/len(pts)),'flask_blue_region_pixel_center':[float(x[roi].mean()),float(y[roi].mean())],'flask_blue_region_valid_depth_fraction':float(valid.sum()/roi.sum()),'flask_blue_region_height_above_table_percentiles_mm':np.percentile(heights,[10,50,90]).tolist(),'flask_blue_surface_camera_xyz_median_mm':np.median(p,axis=0).tolist(),'cup_opening_pixel_center_approx':[505,102],'limitations':['Blue patch is a visible surface, not neck center','Cup opening center is manually identified in image; world pose unresolved','Camera extrinsic accuracy not verified','No IK, gripper clearance or collision plan computed']}
(root/'planning/flask-localization-preview.json').write_text(json.dumps(report,indent=2)+'\n')
cv2.rectangle(im,(435,245),(535,345),(0,220,255),2);cv2.putText(im,'Flask neck: perception region only',(250,470),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2);cv2.ellipse(im,(505,102),(68,40),0,0,360,(0,220,255),2);cv2.putText(im,'Receiving cup',(580,105),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2);cv2.putText(im,'PREVIEW - no motion targets validated',(30,695),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,255),2)
cv2.imwrite(str(root/'captures/flask-localization-preview.jpg'),im)
print(json.dumps(report,indent=2))
