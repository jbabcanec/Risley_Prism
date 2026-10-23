#!/usr/bin/env python3
"""Cluster sweep harness: the large-scale experimental program as
embarrassingly parallel single-core tasks with resume-safe JSONL output.

Experiments
-----------
adaptive   solve the full 18-D problem at each T in --T (stop at first
           success); records per-T outcome, truth margins, certificates on
           the final failure. The identifiability-atlas workhorse.
fixedT     solve at every T in --T without early stop (scaling-law cohorts).
mint       bisect the minimal recovering T in [Tmin, Tmax] for cases that
           fail at T[0] (prescription-validation quantiles).
speedsP    signed-speed extraction only, arbitrary prism count --P
           (the P x T frontier; completion is P=3-only).
noise      adaptive protocol under additive white noise --snr (dB).

Sharding: --start/--count index into the case stream (case_at: order-free,
any shard generates any index). Output: results/sweep_<exp>_<tag>_<start>.jsonl,
one flushed row per task, resumed by (case, T, snr) key on restart.

Examples
--------
  python experiments/sweep.py --exp adaptive --start 0 --count 100 --tag atlas
  python experiments/sweep.py --exp speedsP --P 4 --T 10 40 --start 0 --count 50
See experiments/slurm_sweep.sh for the cluster template.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import re
import sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import (DT, FS, vec2pat, pat_P, case_at, truth_margins,
                            solve18, extract_speeds, spectral_certificate,
                            spectral_fisher, certify_success)
from risley_lattice.model import battery_cases_P

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Directories holding results from EARLIER runs (possibly a different tag or a
# different cases-per-file layout). Cases already recorded there are skipped,
# so a re-launch never recomputes work you have already paid for. Only rows
# that carry the full modern schema (x18 present) count as done.
RESUME_DIRS = []
_LISTDIR_CACHE = {}


def make_pattern(tc, T, snr_db=None, noise_rng=None):
    pat = vec2pat(tc, int(round(T * FS)), T)
    if snr_db is not None and np.isfinite(snr_db):
        prms = np.std(pat - pat.mean(0))
        sig = prms / (10 ** (snr_db / 20.0))
        pat = pat + noise_rng.normal(0.0, sig, pat.shape)
    return pat


def condense_cert(reasons):
    """First token of each certificate line: the machine-readable classes."""
    out = []
    for r in (reasons or [])[:6]:
        out.append(r.split(':')[0].strip())
    return out


def _parse_t_req(reasons):
    """Extract the maximum certified T_req from reason strings."""
    vals = [int(m.group(1))
            for r in reasons
            for m in [re.search(r'need T >= (\d+)', r)] if m]
    return max(vals) if vals else None


def solve_once(tc, T, snr_db, noise_rng, with_bounds=False):
    t0 = time.time()
    pat = make_pattern(tc, T, snr_db, noise_rng)
    x18, mse, how, info = solve18(pat)
    err = float(np.max(np.abs(x18 - tc))) if x18 is not None else 1e9
    row = {'T': T, 'err': err, 'mse': float(mse), 'how': how,
           'x18': x18.tolist() if x18 is not None else None,
           'min_cycles': float(np.min(np.abs(tc[:3])) * T),
           'secs': round(time.time() - t0, 2),
           'res_clean': info.get('res_clean'),
           'overload': bool(info.get('overload_retry', False)),
           'masked': info.get('masked', 0)}
    if err >= 1e-3:
        N_est = info.get('N')
        if N_est is None:
            N_est = info.get('gens_partial')
        if N_est is not None and len(np.atleast_1d(N_est)):
            try:
                reasons, sig_g = spectral_certificate(pat, N_est, info)
                row['cert'] = condense_cert(reasons)
                row['T_req'] = _parse_t_req(reasons)
                row['sigma_N'] = float(sig_g.max()) if len(sig_g) else None
            except Exception as ex:
                row['cert'] = [f'cert-error {type(ex).__name__}']
                row['T_req'] = None
                row['sigma_N'] = None
            # min_amp: minimum prism fundamental amplitude from Fisher fit
            try:
                mask = info.get('mask', np.ones(len(pat), bool))
                _, _, amp, _, _ = spectral_fisher(pat, N_est, mask)
                row['min_amp'] = float(amp.min()) if len(amp) else None
            except Exception:
                row['min_amp'] = None
        else:
            row['cert'] = ['no-generators']
            row['T_req'] = None
            row['sigma_N'] = None
            row['min_amp'] = None
    elif with_bounds:
        smask = info.get('mask', np.ones(len(pat), bool))
        rmask = np.repeat(smask, 2)
        try:
            b = certify_success(x18, pat, rmask)
            errs = np.abs(x18 - tc)
            row['bounds_cover'] = bool(np.all(errs <= np.maximum(b, 1e-14)))
            row['bound_max'] = float(np.max(b))
            row['tightness_med'] = float(np.median(
                b / np.maximum(errs, 1e-16)))
        except Exception as ex:
            row['bounds_cover'] = None
    return row, err


def _resume_paths(path, args):
    """This shard's own file, plus any file in RESUME_DIRS whose case range
    overlaps [start, start+count). Filenames encode their start index, so the
    overlap test is a cheap string parse -- no need to open 40k files."""
    yield path
    lo, hi = args.start, args.start + args.count
    pref = f'sweep_{args.exp}_'
    for d in RESUME_DIRS:
        if not os.path.isdir(d):
            continue
        names = _LISTDIR_CACHE.get(d)
        if names is None:
            try:
                names = os.listdir(d)
            except OSError:
                names = []
            _LISTDIR_CACHE[d] = names   # legacy dir can hold 50k+ entries
        for nm in names:
            if not (nm.startswith(pref) and nm.endswith('.jsonl')):
                continue
            try:
                s0 = int(nm[:-len('.jsonl')].rsplit('_', 1)[1])
            except (ValueError, IndexError):
                continue
            if lo <= s0 < hi:
                yield os.path.join(d, nm)


def _load_done(path, args):
    """(case, snr) keys already computed. A row only counts if it carries the
    modern schema -- rows from before x18 was added are recomputed, which is
    the whole point of the re-run."""
    done = set()
    for p in _resume_paths(path, args):
        if not os.path.exists(p):
            continue
        try:
            fh = open(p)
        except OSError:
            continue
        with fh as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                trail = r.get('trail') or []
                if trail and 'x18' not in trail[0]:
                    continue          # pre-x18 row: redo it
                try:
                    done.add((r['case'], r.get('snr')))
                except KeyError:
                    pass
    return done


def run(args):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(
        OUTDIR, f'sweep_{args.exp}_{args.tag}_{args.start:07d}.jsonl')
    done = _load_done(path, args)
    fout = open(path, 'a')
    snrs = args.snr if args.exp == 'noise' else [None]
    for i in range(args.start, args.start + args.count):
        tc = case_at(i, args.seed0)
        marg = truth_margins(tc) if args.exp != 'speedsP' else {}
        for snr in snrs:
            if (i, snr) in done:
                continue
            noise_rng = np.random.default_rng([args.seed0, i, 777])
            rec = {'exp': args.exp, 'case': i, 'seed0': args.seed0,
                   'snr': snr, **marg}
            if args.exp in ('adaptive', 'noise'):
                trail = []
                for T in args.T:
                    row, err = solve_once(tc, T, snr, noise_rng,
                                          with_bounds=args.bounds)
                    trail.append(row)
                    if err < 1e-3:
                        rec['T_solved'] = T
                        break
                else:
                    rec['T_solved'] = None
                rec['trail'] = trail
            elif args.exp == 'fixedT':
                rec['trail'] = [solve_once(tc, T, snr, noise_rng,
                                           with_bounds=args.bounds)[0]
                                for T in args.T]
            elif args.exp == 'mint':
                row0, err0 = solve_once(tc, args.T[0], snr, noise_rng)
                rec['trail'] = [row0]
                if err0 < 1e-3:
                    rec['T_min'] = args.T[0]
                else:
                    lo, hi = args.T[0], args.Tmax
                    row_hi, err_hi = solve_once(tc, hi, snr, noise_rng)
                    rec['trail'].append(row_hi)
                    if err_hi >= 1e-3:
                        rec['T_min'] = None      # infeasible below Tmax
                    else:
                        for _ in range(args.bisect):
                            mid = round((lo * hi) ** 0.5, 1)
                            rowm, errm = solve_once(tc, mid, snr, noise_rng)
                            rec['trail'].append(rowm)
                            if errm < 1e-3:
                                hi = mid
                            else:
                                lo = mid
                        rec['T_min'] = hi
            elif args.exp == 'speedsP':
                sp, ax, ay, ng, geom9 = battery_cases_P(
                    args.P, n=1, seed=args.seed0 + i)[0]
                pair = min(min(abs(sp[a] - sp[b]), abs(sp[a] + sp[b]))
                           for a in range(args.P)
                           for b in range(a + 1, args.P)) \
                    if args.P > 1 else np.inf
                rec.update(P=args.P, pair_sep=float(pair),
                           ax_min=float(np.min(np.abs(ax))),
                           cyc=float(np.min(np.abs(sp)) * 10.0))
                trail = []
                for T in args.T:
                    t0 = time.time()
                    pat = pat_P(sp, ax, ay, ng, geom9,
                                int(round(T * FS)), T)
                    N, info = extract_speeds(pat, DT, n_gen=args.P)
                    e = float(np.max(np.abs(N - sp))) if N is not None \
                        else 1e9
                    trail.append({'T': T, 'err': e,
                                  'secs': round(time.time() - t0, 2),
                                  'overload': bool(
                                      info.get('overload_retry', False))})
                    if e < 0.02:
                        rec['T_solved'] = T
                        break
                else:
                    rec['T_solved'] = None
                rec['trail'] = trail
            fout.write(json.dumps(rec) + '\n')
            fout.flush()
    fout.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True,
                    choices=['adaptive', 'fixedT', 'mint', 'speedsP',
                             'noise'])
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--seed0', type=int, default=424242)
    ap.add_argument('--T', type=float, nargs='+',
                    default=[10.0, 20.0, 40.0, 80.0])
    ap.add_argument('--Tmax', type=float, default=320.0)
    ap.add_argument('--bisect', type=int, default=5)
    ap.add_argument('--P', type=int, default=3)
    ap.add_argument('--snr', type=float, nargs='+', default=[60, 50, 40, 30])
    ap.add_argument('--bounds', action='store_true',
                    help='certify successes (adds ~37 forward evals each)')
    ap.add_argument('--tag', default='run')
    run(ap.parse_args())
