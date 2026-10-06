"""Predeclared separate repeated-speed stratum; no case-specific tuning."""
from contract import *

def main():
    rng=np.random.default_rng(20261004);rows=[];rejects=[]
    for draw in range(100):
        v=LO+(HI-LO)*rng.random(18)
        pair=rng.choice(3,size=2,replace=False);v[pair[1]]=v[pair[0]]
        try:y,m,floor=checked(v)
        except Exception as exc:
            rejects.append({'draw':draw,'truth':v.tolist(),'reason':str(exc)});continue
        rows.append({'id':f'collision_holdout_{len(rows):02d}','draw':draw,'truth':v.tolist(),'observed':y.tolist(),'margins':m,'canonical_vs_interval_max_bound':floor})
        if len(rows)==2:break
    base=json.loads((WORK/'cases.json').read_text())
    base.update(seed=20261004,cases=rows,rejected=rejects,stratum='Two randomly selected physical prism positions share a sampled signed speed. All other coordinates sampled independently across original native bounds; no speed sorting.')
    (WORK/'collision_holdout.json').write_text(json.dumps(base,indent=2))
    (WORK/'collision_holdout_observations.json').write_text(json.dumps({'timestamps':base['timestamps'],'lower':base['lower'],'upper':base['upper'],'cases':[{'id':r['id'],'observed':r['observed']} for r in rows]},indent=2))
    print(json.dumps({'cases':len(rows),'rejections':len(rejects),'output':'collision_holdout_observations.json'}))

if __name__=='__main__':main()
