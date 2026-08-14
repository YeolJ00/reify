"""THREE probes on the same 14-object scene, all with true mesh contact.

  tilt     ramp 0->26 deg          -> slip angle reads friction, topple angle reads CoM
  drop     released 12 cm up       -> rebound reads damping cd
  collide  shoved at 1.2 m/s       -> momentum transfer reads MASS, which tilt cannot see
"""
import sys, json; sys.path.insert(0,'/home/nas5/jooyeolyun/repos/simulation-assestization')
sys.path.insert(0,'/home/nas5/jooyeolyun/repos/simulation-assestization/scripts')
import numpy as np, warp as wp
from src.sim.mesh_scene import MeshProbeScene
from src.sim.tilt_probe import ramp_gravity_seq, onset_angle
from bigscene_sim import THETA, TABLE, GZ, FPS, SUBSTEPS, layout
A=json.load(open("/home/jooyeolyun/.claude/jobs/5cc17d67/tmp/assets.json"))
wp.init(); NF,SET=60,12
xy,_=layout(A); keys=list(A)
names=[f"{A[k]['cat']}/{k}" for k in keys]; sc=[A[k]["scale"] for k in keys]
mus=[THETA[k]["mu"] for k in keys]; cds=[THETA[k]["cd"] for k in keys]
ms=[THETA[k]["mass"] for k in keys]

def build(n_steps, lift=0.0, vel=None):
    s=MeshProbeScene(names,[[0,0,0]]*len(names),vel or [[0.,0.,0.]]*len(names),
        masses=ms,ground_z=GZ,dt=1/(FPS*SUBSTEPS),n_steps=n_steps,k=2500.,cd=cds,mu=mus,
        mesh_scale=sc,faces=900,ground_mu=TABLE["mu"],ground_cd=TABLE["cd"])
    s.pos0=np.array([[xy[k][0],xy[k][1],s.rest_height(i)+0.002+lift]
                     for i,k in enumerate(keys)],np.float32)
    s.calibrate_stiffness(); return s

out={}
with wp.ScopedDevice("cuda:0"):
    # ---- tilt
    s=build(NF*SUBSTEPS); s.gravity_seq=ramp_gravity_seq(0,26,NF,SUBSTEPS,SET); s.rollout()
    P=s.positions(SUBSTEPS)
    print("=== TILT (friction / centre of mass)")
    print(f"  {'object':<32}{'true slip':>10}{'onset':>8}{'err':>7}")
    for i,k in enumerate(keys):
        mue=np.sqrt(mus[i]*TABLE["mu"]); t=np.rad2deg(np.arctan(mue))
        o=onset_angle(P[:,i,:2],0,26,NF,SET,s.sizes[i])
        out.setdefault(k,{})["tilt"]={"true":t,"onset":o}
        if A[k]["probe"] in ("slide(mu)","mixed"):
            print(f"  {k:<32}{t:>10.1f}{(f'{o:.1f}' if o else 'none'):>8}"
                  f"{(f'{o-t:+.1f}' if o else '--'):>7}")
    # ---- drop
    s2=build(40*SUBSTEPS,lift=0.12); s2.rollout(); P2=s2.positions(SUBSTEPS)
    print("\n=== DROP 12 cm (damping cd -> rebound)")
    print(f"  {'object':<32}{'cd':>7}{'rebound cm':>12}")
    for i,k in enumerate(keys):
        z=P2[:,i,2]; lo=int(np.argmin(z)); reb=float(z[lo:].max()-z[lo])*100
        out[k]["drop"]={"cd":cds[i],"rebound_cm":reb}
        if i<7: print(f"  {k:<32}{cds[i]:>7.0f}{reb:>12.2f}")
    r=[(cds[i],out[k]["drop"]["rebound_cm"]) for i,k in enumerate(keys)]
    r=np.array(r); print(f"  rho(log cd, rebound) = {np.corrcoef(np.log10(r[:,0]),r[:,1])[0,1]:+.3f}"
                        f"   (expect NEGATIVE: more damping, less bounce)")
    # ---- collide
    v=[[1.2,0.,0.]]*len(names)
    s3=build(30*SUBSTEPS,vel=v); s3.rollout(); P3=s3.positions(SUBSTEPS)
    print("\n=== COLLIDE, shoved at 1.2 m/s (MASS -> stopping distance)")
    print(f"  {'object':<32}{'mass kg':>9}{'travel cm':>11}")
    d=[]
    for i,k in enumerate(keys):
        tr=float(np.linalg.norm(P3[-1,i,:2]-P3[0,i,:2]))*100
        out[k]["collide"]={"mass":ms[i],"travel_cm":tr}; d.append((ms[i],tr))
        if i<7: print(f"  {k:<32}{ms[i]:>9.2f}{tr:>11.1f}")
    d=np.array(d); print(f"  rho(mass, travel) = {np.corrcoef(d[:,0],d[:,1])[0,1]:+.3f}"
                        f"   (mass is observable here, and invisible to tilt)")
json.dump(out,open("outputs/scene/bigscene/probes.json","w"),indent=1)
