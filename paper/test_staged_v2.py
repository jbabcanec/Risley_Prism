#!/usr/bin/env python3
"""
Staged recovery v2: honest, no tight-bound tricks.

Stage 1: FFT → 3 speed magnitudes
Stage 2: Try all P! × 2^P = 48 (speed ordering × sign) combos.
         For each: fix speeds, search 15-D (angles+glass+dist+beam).
         4 restarts per combo, 300 gen. Best combo wins.
Stage 3: Unfreeze speeds. Full 19-D DE (6 restarts seeded from Stage 2)
         + 100k NM polish.
Stage 4: Full permutation search with heavy NM per permutation.
"""
import sys, os, time, json, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core import PrismParameters, SystemGeometry, fast_forward
from scipy.optimize import differential_evolution, minimize
from itertools import permutations, product

T_OBS = 10.0
N_WORKERS = 4

TRUE_VEC = np.array([1.5, -1.0, 2.0, 12.0, -8.0, 5.0, 3.0, 10.0, -6.0,
                     1.50, 1.55, 1.60, 100.0, 6.0, 10.0, 5.0, 0.0, 0.0])
TRUE_P = PrismParameters(3, [1.5,-1.0,2.0], [12.0,-8.0,5.0], [3.0,10.0,-6.0],
                          glass_indices=[1.50,1.55,1.60],
                          geometry=SystemGeometry(workpiece_distance=100.0,
                          inter_prism_gap=6.0, beam_angle_x=10.0, beam_angle_y=5.0))
TARGET = fast_forward(TRUE_P, 200, T_OBS)


class Stage2Obj:
    def __init__(self, speeds):
        self.speeds = speeds
    def __call__(self, x):
        try:
            geo = SystemGeometry(workpiece_distance=x[9], inter_prism_gap=x[10],
                                 beam_angle_x=x[11], beam_angle_y=x[12],
                                 beam_pos_x=x[13], beam_pos_y=x[14])
            p = PrismParameters(3, self.speeds, x[:3].tolist(), x[3:6].tolist(),
                                glass_indices=x[6:9].tolist(), geometry=geo)
            return float(np.mean((fast_forward(p, 200, T_OBS) - TARGET)**2))
        except: return 1e6


class FullObj:
    def __call__(self, x):
        try:
            geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13],
                                 beam_angle_x=x[14], beam_angle_y=x[15],
                                 beam_pos_x=x[16], beam_pos_y=x[17])
            p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                                glass_indices=x[9:12].tolist(), geometry=geo)
            return float(np.mean((fast_forward(p, 200, T_OBS) - TARGET)**2))
        except: return 1e6


if __name__ == '__main__':
    t0_total = time.time()
    full_obj = FullObj()

    # ── Stage 1: FFT ──
    T = len(TARGET)
    dt = T_OBS / T
    freqs = np.fft.rfftfreq(T, d=dt)
    cx, cy = np.mean(TARGET[:, 0]), np.mean(TARGET[:, 1])
    fx = np.abs(np.fft.rfft(TARGET[:, 0] - cx))
    fy = np.abs(np.fft.rfft(TARGET[:, 1] - cy))
    power = fx + fy; power[0] = 0

    speed_mags = []
    for _ in range(3):
        idx = np.argmax(power)
        speed_mags.append(float(freqs[idx]))
        power[max(1,idx-3):min(len(power),idx+4)] = 0
    speed_mags.sort()

    beam_ax_hint = np.degrees(np.arctan2(cx, 120.0))
    beam_ay_hint = np.degrees(np.arctan2(cy, 120.0))

    print("=" * 70)
    print("STAGED RECOVERY v2 (48 combos, honest)")
    print("=" * 70)
    print(f"FFT speeds: {speed_mags}")
    print(f"Beam hints: ({beam_ax_hint:.1f}, {beam_ay_hint:.1f}) deg")

    # ── Stage 2: 48 combos (6 orderings × 8 signs) ──
    s2_bounds = ([(-18,18)]*3 + [(-18,18)]*3 + [(1.3,1.8)]*3 +
                 [(50,200)] + [(2,15)] + [(-25,25)]*2 + [(-5,5)]*2)
    s2_seed = np.array([8,-5,3, 2,5,-3, 1.50,1.55,1.60,
                        100,6, beam_ax_hint,beam_ay_hint, 0,0])

    # Track top 5 combos, not just the best
    top_combos = []  # list of (mse, x, speeds)
    n_combos = 0

    print(f"\nStage 2: 48 combos x 6 restarts (15-D each)")
    for perm in permutations(range(3)):
        ordered = [speed_mags[i] for i in perm]
        for signs in product([1,-1], repeat=3):
            signed = [s * sign for s, sign in zip(ordered, signs)]
            obj2 = Stage2Obj(signed)
            n_combos += 1

            combo_best_mse = float('inf')
            combo_best_x = None
            for r in range(6):
                x0 = s2_seed if r == 0 else None
                res = differential_evolution(obj2, s2_bounds,
                                             seed=n_combos*100+r*17,
                                             maxiter=300, popsize=25, tol=1e-12,
                                             polish=False, x0=x0,
                                             workers=N_WORKERS, updating='deferred')
                if res.fun < combo_best_mse:
                    combo_best_mse = res.fun
                    combo_best_x = res.x.copy()

            top_combos.append((combo_best_mse, combo_best_x, signed[:]))

            if n_combos % 8 == 0:
                top_combos.sort(key=lambda t: t[0])
                print(f"  {n_combos}/48 combos done, "
                      f"best MSE={top_combos[0][0]:.2e}", flush=True)

    # Stage 2.5: Polish top 5 combos and pick the winner
    top_combos.sort(key=lambda t: t[0])
    print(f"\nStage 2.5: Polishing top 5 combos")
    best_s2_x, best_s2_mse, best_speeds = None, float('inf'), None

    for rank, (mse, x, speeds) in enumerate(top_combos[:5]):
        obj2 = Stage2Obj(speeds)
        res_nm = minimize(obj2, x, method='Nelder-Mead',
                          options={'maxiter': 50000, 'fatol': 1e-15})
        polished_mse = res_nm.fun
        print(f"  #{rank+1}: speeds={[f'{s:+.1f}' for s in speeds]} "
              f"DE={mse:.2e} -> NM={polished_mse:.2e}", flush=True)
        if polished_mse < best_s2_mse:
            best_s2_mse = polished_mse
            best_s2_x = res_nm.x.copy()
            best_speeds = speeds[:]

    print(f"\n  Stage 2 winner: MSE={best_s2_mse:.2e}")
    print(f"  Best speeds: {best_speeds}")
    print(f"  True speeds: [1.5, -1.0, 2.0]")

    # ── Stage 3: Full 19-D from Stage 2 ──
    print(f"\nStage 3: Full 19-D refinement")

    full_x = np.concatenate([np.array(best_speeds),
                              best_s2_x[:3], best_s2_x[3:6],  # ax, ay
                              best_s2_x[6:9],                  # glass
                              best_s2_x[9:]])                  # dist+beam

    full_bounds = ([(-3.5,3.5)]*3 + [(-18,18)]*6 + [(1.3,1.8)]*3 +
                   [(50,200)] + [(2,15)] + [(-25,25)]*2 + [(-5,5)]*2)

    # Clip to bounds
    for i in range(len(full_x)):
        full_x[i] = np.clip(full_x[i], full_bounds[i][0], full_bounds[i][1])

    best_full_x, best_full_mse = full_x.copy(), full_obj(full_x)

    # DE restarts seeded from Stage 2
    for r in range(6):
        res = differential_evolution(full_obj, full_bounds, seed=500+r,
                                     maxiter=300, popsize=30, tol=1e-14,
                                     polish=False, x0=best_full_x,
                                     workers=N_WORKERS, updating='deferred')
        tag = ''
        if res.fun < best_full_mse:
            best_full_x, best_full_mse = res.x.copy(), res.fun
            tag = ' *'
        print(f"  DE {r+1}/6: MSE={res.fun:.2e}{tag}", flush=True)

    # Heavy NM
    res3 = minimize(full_obj, best_full_x, method='Nelder-Mead',
                    options={'maxiter': 100000, 'fatol': 1e-16, 'xatol': 1e-16})
    if res3.fun < best_full_mse:
        best_full_x, best_full_mse = res3.x.copy(), res3.fun
    print(f"  NM: MSE={best_full_mse:.2e}")

    # ── Stage 4: Permutation search with heavy polish ──
    print(f"\nStage 4: Permutation search")
    sp=best_full_x[:3]; ax=best_full_x[3:6]; ay=best_full_x[6:9]
    gl=best_full_x[9:12]; rest=best_full_x[12:]

    perm_idx = 0
    for perm in permutations(range(3)):
        if list(perm)==[0,1,2]: continue
        perm_idx += 1
        xp = np.concatenate([sp[list(perm)], ax[list(perm)], ay[list(perm)],
                              gl[list(perm)], rest])
        for j in range(len(xp)):
            xp[j] = np.clip(xp[j], full_bounds[j][0], full_bounds[j][1])
        # DE + heavy NM per permutation
        rp_de = differential_evolution(full_obj, full_bounds, seed=700+perm_idx*13,
                                        maxiter=250, popsize=30, tol=1e-14,
                                        polish=False, x0=xp,
                                        workers=N_WORKERS, updating='deferred')
        rp = minimize(full_obj, rp_de.x, method='Nelder-Mead',
                      options={'maxiter': 80000, 'fatol': 1e-16, 'xatol': 1e-16})
        if rp.fun < best_full_mse:
            best_full_x, best_full_mse = rp.x.copy(), rp.fun
            print(f"  perm {list(perm)}: MSE={best_full_mse:.2e} *", flush=True)
        else:
            print(f"  perm {list(perm)}: MSE={rp.fun:.2e}", flush=True)

    # ── Final report ──
    wt = time.time() - t0_total
    errors = np.abs(best_full_x - TRUE_VEC)

    print(f"\n{'='*70}")
    print(f"FINAL: MSE={best_full_mse:.2e}, max_err={np.max(errors):.2e} ({wt:.0f}s)")
    print(f"{'='*70}")
    names = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
             'ng1','ng2','ng3','d_W','gap','beam_ax','beam_ay','beam_px','beam_py']
    for n,e,t,r in zip(names, errors, TRUE_VEC, best_full_x):
        s = 'OK' if e<1e-3 else 'CLOSE' if e<0.1 else 'MEH' if e<1.0 else 'BAD'
        print(f'  {n:10s}: err={e:.4e}  true={t:.4f}  rec={r:.4f}  [{s}]')

    with open('staged_v2_results.json','w') as f:
        json.dump({'mse':float(best_full_mse),'errors':errors.tolist(),
                   'recovered':best_full_x.tolist(),'time_s':round(wt,1)},f,indent=2)
    print(f"\nSaved to staged_v2_results.json")
