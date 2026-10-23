# Oct 6 meeting follow-ups

These scripts address the action items from the Oct 6, 2026 Risley prism meeting. Everything was run locally in Python, with no Modal. The write-up is `Ben/risley_prism_analysis_v2.docx` (sections C1–C6).

| Script | What it does | Output |
|---|---|---|
| `compact_ready.py`, `add_truth.py` | Compact the 669,574-case ready dataset to one numpy table, and regenerate each case's truth from its seed. | `~/work/camp/campaign.npz` (scratch) |
| `tir.py` | Exact worst-case TIR margin (closed form) and a forward-model TIR detector, used to validate it. | — |
| `tir_campaign.py` | What TIR explains in the campaign, plus restricted-box recovery rates. | `out/tir_campaign.json`, `figures/T1_*`, `figures/T2_*` |
| `local_campaign.py` | Small checkpointed runs: `box10`, `ben`, `ben066`, `calib`. | `~/work/lc/*.jsonl` |
| `sign_sweep_local.py` | Sign-sweep rung on all of G1 and a random 60 from G3. | `~/work/ss/*.jsonl`, `out/sign_sweep_local.json` |
| `nonunique.py` | ε-equivalence, twins, multistart and equal-speed analyses. | `~/work/nu/*.jsonl` |
| `atlas_overlap.py` | Atlas overlap and reproducibility check. | `out/atlas_overlap.json` |
| `accepted_wrong.py` | Trials that pass the MSE acceptance test but are more than 1e-3 from the truth. | `out/mse_accepted_wrong.json` |
| `regress_solve18.py` | Regression test for the new `sign_sweep` rung in `risley_lattice/solve.py`. | `out/regress_solve18.json` |
| `export_local.py` | Summaries and figures for all the local runs. | `out/local_summary.json`, `figures/C*` |
| `fill_paper.py` | Fills paper placeholders E1–E4 without touching `main.tex`. | `paper/main_campaign.tex`, `paper/campaign_E1_E4_parts/`, `paper/figures/E*` |
| `rerun_ben4.py` | Re-runs the 4 unsolved Ben-regime machines with the sign-flip fix on, viewing times up to 320 s (`--all` has no time limit; run it in Terminal). | `out/rerun_ben4.jsonl` |
| `warmstart.py` | Warm-start test: reuses the answer from a shorter viewing time as the starting guess for the next one (machine 87 plus a sample of September failures). | `out/warmstart.json`, `~/work/warm/` |
| `glassknown.py` | Re-solves lookalike machines and flat-wedge controls with the glass type unknown vs. known. | `out/glassknown.json`, `~/work/glass/` |
| `build_plain_docx.py` | Builds the plain-language companion document `Risley_October_Followups_Plain_Language.docx`. | — |
| `build_docx.py` | Builds `risley_prism_analysis_v2.docx` from the original document plus the outputs above. | — |
| `round.sh`, `progress.py` | Drive the local runs in ~160 s rounds on 4 cores, and report progress. | — |

## Code change

`risley_lattice/solve.py` now has a final `sign_sweep` rung in `solve18`, covering pair, triple and single flips plus a polish step. It's on by default, and `RISLEY_SIGN_SWEEP=0` disables it. The original file is at `backup/solve.py.orig`. All the local campaigns above ran with the rung off, so they stay comparable to the 669,574-case campaign.

## Reproduce

Run from this folder:

```bash
python3 compact_ready.py --part 0 --out ~/work/camp   # ... parts 0-3, then:
python3 compact_ready.py --merge --out ~/work/camp
python3 add_truth.py --dir ~/work/camp --part 0       # ... parts 0-3, then --merge
python3 tir_campaign.py; python3 atlas_overlap.py; python3 accepted_wrong.py
python3 sign_sweep_local.py build --work ~/work/ss --g3 60
bash round.sh            # repeat until progress.py shows everything finished
python3 export_local.py; python3 regress_solve18.py report
python3 fill_paper.py; python3 build_docx.py
```

## Raw run logs

`run_logs/` is a copy of the per-test logs from the local runs, which originally lived in a temporary working folder (`~/work/`):

- `lc/`: the box10, ben, ben066 and calib campaigns.
- `ss/`: the sign-flip runs.
- `nu/`: the uniqueness tests.
- `warm/`: the warm-start test.
- `glass/`: the glass-known test.
- `regress/`: the 30-machine regression check.

The `start_*` files and `long.json` in `glass/` are only bookkeeping markers. To re-run a report from these copies, point the script at them. For example: `LC_WORK=run_logs/lc python3 export_local.py`. For `sign_sweep_local.py`, `report` takes `--work run_logs/ss`.

The large compacted campaign table (`~/work/camp/campaign.npz`, about 570 MB) is not copied, because `compact_ready.py` and `add_truth.py` regenerate it in seconds.
