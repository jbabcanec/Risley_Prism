"""Numerical-stability audit; original core code and old observations unchanged."""
import os
import sys
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
sys.dont_write_bytecode=True
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from forward import vec2pat_stable,fast_forward_stable,default_times,PhysicalBranchError
from oracle import reference,error_upper

ROOT=Path(__file__).parents[1]
REPO=Path(os.environ.get("WEDGE_RESEARCH_ROOT",r"C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge"))
sys.path.insert(0,str(REPO))
from risley_lattice.model import LO,HI,RG,vec2pat
from risley_lattice.separable import affine_geometry,LINEAR
from risley_lattice.fmodel import forward_point
from core import PrismParameters,SystemGeometry,_surface_endpoints,_intersect


def parameters(v):
    g=SystemGeometry(source_distance=6.,prism_thickness=3.,
        inter_prism_gap=v[13],workpiece_distance=v[12],beam_angle_x=v[14],
        beam_angle_y=v[15],beam_pos_x=v[16],beam_pos_y=v[17])
    return PrismParameters(3,list(v[:3]),list(v[3:6]),list(v[6:9]),list(v[9:12]),g)


def legacy_ablation(v,mode="acos",trace_index=None):
    """Same old endpoint arithmetic; change only angle extraction to atan2."""
    p=parameters(v)
    speeds,sphix,sphiy,ref,dist=p._build_interface_model()
    zbase=np.cumsum(dist)
    t=default_times()
    gamma=(360*np.outer(speeds,t))%360
    angle=np.radians(gamma+sphiy[:,None])
    tangent=np.tan(np.radians(sphix))[:,None]
    u,w=np.cos(angle)*tangent,np.sin(angle)*tangent
    normal=np.sqrt(u*u+w*w+1)
    effective=[90-np.degrees(np.arccos(np.clip(x/normal,-1,1))) for x in (u,w)]
    output=[]
    records=[]
    for axis,phi in enumerate(effective):
        phi=np.vstack([phi,np.zeros((1,len(t)))])
        r0,theta0=v[16+axis],v[14+axis]
        slope=np.tan(np.radians(theta0))
        a,b,c,d=_surface_endpoints(phi[0],zbase[0])
        pos,zpos=_intersect(r0,0.,r0+slope,1.,a,b,c,d)
        theta=np.full(len(t),theta0)
        for j in range(6):
            tp=np.tan(np.radians(phi[j])); nn=np.sqrt(1+tp*tp)
            n0,nz=tp/nn,-1/nn
            ti=np.tan(np.radians(theta)); sn=np.sqrt(1+ti*ti)
            i0,iz=ti/sn,1/sn
            nr=ref[j]/ref[j+1]
            cy=nz*i0-n0*iz
            root=np.sqrt(np.maximum(0.,1.0-nr**2*cy**2))
            o0=nr*nz*cy-n0*root
            oz=-nr*n0*cy-nz*root
            norm=np.sqrt(o0*o0+oz*oz+1e-30)
            cos=np.clip(oz/norm,-1,1)
            acos=np.sign(o0)*np.degrees(np.arccos(cos))
            atan=np.degrees(np.arctan2(o0,oz))
            theta=np.where(abs(o0)<1e-12,0.,acos if mode=="acos" else atan)
            if trace_index is not None:
                k,selected_axis=trace_index
                if axis==selected_axis:
                    records.append(dict(interface=j,sf0=float(o0[k]),sf2=float(oz[k]),
                        computed_cosine=float(cos[k]),acos_degrees=float(acos[k]),
                        atan2_degrees=float(atan[k]),used_degrees=float(theta[k])))
            tangent_new=np.tan(np.radians(theta))
            a,b,c,d=_surface_endpoints(phi[j+1],zbase[j+1])
            pos,zpos=_intersect(pos,zpos,pos+tangent_new,zpos+1,a,b,c,d)
        output.append(pos)
    return np.stack(output,axis=-1),records


def fixtures():
    shared=json.loads((ROOT/"cases.json").read_text())
    out=[dict(id="shared/"+c["id"],kind="shared",theta=c["truth"]) for c in shared["cases"]]
    rng=np.random.default_rng(920261002)
    for i in range(100):
        out.append(dict(id=f"native_random/{i:03d}",kind="fresh_native_uniform",theta=(LO+RG*rng.random(18)).tolist()))
    saved=json.loads((ROOT/"ambiguity/benchmark_verified_60.json").read_text())
    for i,r in enumerate(saved["records"]):
        for endpoint in ("base","alternative"):
            out.append(dict(id=f"saved/{i}/{endpoint}",kind="saved_nonzero_wedge",theta=r[endpoint]))
    flat=np.array([.7,-1.2,2.1,0,0,0,-7,11,17,1.5,1.5,1.5,137,7,
        (180/np.pi)*2.**-30,0.,0.,0.])
    out.append(dict(id="adversarial/flat_tiny_beam",kind="analytic_flat",theta=flat.tolist()))
    for i,desired in enumerate((0.,2.**-30,-2.**-30,2.**-28,-2.**-28,1e-8)):
        v=np.array([1.3,-.7,2.4,6.,-5.,4.,7.,-11.,17.,1.5,1.6,1.4,137.,7.,0.,2.,1.2,-2.1])
        s=np.sin(np.deg2rad(v[3]))*np.cos(np.deg2rad(v[6]))
        c=np.sqrt(1-s*s)
        q=desired*c+np.sqrt(1-desired*desired)*s
        a=q*c-s*np.sqrt(v[9]**2-q*q)
        v[14]=np.rad2deg(np.arcsin(a))
        out.append(dict(id=f"adversarial/nonzero_cancel_{i}",kind="all_wedges_nonzero",
            desired_first_exit_transverse=desired,theta=v.tolist()))
    return out


def main():
    start=time.perf_counter()
    rows=[]
    times=default_times()
    assert np.array_equal(times,np.arange(0,10.,10./200)[:200])
    time_difference=float(np.max(abs(times-np.arange(200)/20.)))
    all_cases=fixtures()
    for case in all_cases:
        v=np.array(case["theta"])
        assert np.all(v>=LO) and np.all(v<=HI)
        row=dict(case)
        try:
            stable,diag=vec2pat_stable(v,return_diagnostics=True)
        except PhysicalBranchError as exc:
            row.update(status="strict_branch_refused",branch=exc.branch,axis=exc.axis,
                       interface=exc.interface,sample=exc.index,value=exc.value)
            rows.append(row)
            continue
        old=vec2pat(v)
        b,a=affine_geometry(v)
        smooth=(b+a@v[LINEAR]).reshape(200,2)
        ablated,_=legacy_ablation(v,"atan2")
        clone,_=legacy_ablation(v,"acos")
        # The ablation must first reproduce the exact legacy program.
        assert np.array_equal(clone,old)
        assert np.array_equal(fast_forward_stable(parameters(v)),stable)
        assert np.array_equal(vec2pat_stable(v,times=times),stable)
        maximum=np.unravel_index(np.argmax(abs(old-stable)),stable.shape)
        _,trace=legacy_ablation(v,"acos",maximum)
        fm,fr,guards=forward_point(v)
        residual_interval=np.abs(stable.ravel()-fm)-fr
        # Full oracle on all saved nonzero failures and adversarial cases;
        # random/shared cases also compare their worst location plus two fixed times.
        indices=np.arange(200) if case["kind"] in ("saved_nonzero_wedge","analytic_flat","all_wedges_nonzero") else np.unique([0,maximum[0],199])
        oracle,oracle_guards=reference(v,times[indices],70)
        errors=dict(stable=error_upper(stable[indices],oracle),
                    legacy=error_upper(old[indices],oracle),atan2_only=error_upper(ablated[indices],oracle))
        row.update(status="strict_branch_valid",core_stable_max_abs=float(np.max(abs(old-stable))),
            existing_smooth_stable_max_abs=float(np.max(abs(smooth-stable))),
            atan2_only_stable_max_abs=float(np.max(abs(ablated-stable))),
            highprecision_error_upper=errors,highprecision_sample_indices=indices.tolist(),
            highprecision_exact_binary_timestamps=True,highprecision_decimal_digits=70,
            stable_inside_repository_interval_enclosure=bool(np.all(residual_interval<=0)),
            stable_interval_enclosure_excess=float(np.max(residual_interval)),
            repository_interval_max_radius=float(np.max(fr)),
            physical_margins=guards,oracle_physical_margins=oracle_guards,
            worst_legacy_error_sample=int(maximum[0]),worst_legacy_error_axis=int(maximum[1]),
            legacy_trace_at_worst_sample=trace,smallest_transverse=diag["smallest_transverse"])
        # Actual witnessed error, no claim this threshold is a global guarantee.
        assert errors["stable"]<1e-8
        assert row["existing_smooth_stable_max_abs"]<1e-9
        assert row["stable_inside_repository_interval_enclosure"]
        rows.append(row)
        if case["kind"]!="fresh_native_uniform" or int(case["id"].split('/')[-1])%20==0:
            print(json.dumps(dict(id=case["id"],legacy=errors["legacy"],stable=errors["stable"],
                same_float_timestamp=True)),flush=True)
    valid=[r for r in rows if r["status"]=="strict_branch_valid"]
    summary=dict(total=len(rows),strict_valid=len(valid),strict_refused=len(rows)-len(valid),
        fresh_uniform_count=100,full_native_box=True,prism_order_never_sorted=True,
        all18_parameters_supplied=True,
        worst_core_stable_discrepancy=max(r["core_stable_max_abs"] for r in valid),
        worst_stable_highprecision_error_upper=max(r["highprecision_error_upper"]["stable"] for r in valid),
        worst_legacy_highprecision_error_upper=max(r["highprecision_error_upper"]["legacy"] for r in valid),
        worst_atan2_only_highprecision_error_upper=max(r["highprecision_error_upper"]["atan2_only"] for r in valid),
        all_stable_inside_repository_interval_enclosures=all(r["stable_inside_repository_interval_enclosure"] for r in valid),
        arange_vs_k_over_20_max_time_difference=time_difference,
        default_time_grid_exactly_preserved=True,
        note="Finite test evidence only; no uniform error bound near critical/grazing optics. Invalid samples were recorded, not silently clipped.")
    result=dict(summary=summary,records=rows,timestamps=times.tolist(),
        versions=dict(numpy=np.__version__),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name("forward.py"),Path(__file__).with_name("oracle.py"),
             REPO/"reverse_problem_v2/core.py",REPO/"risley_lattice/fmodel.py"]},
        elapsed_seconds=time.perf_counter()-start)
    destination=Path(__file__).with_name("results.json")
    destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
