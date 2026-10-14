"""One-off frozen cross-visitation diagnostic; no training components constructed."""
import argparse, csv, gzip, hashlib, json, sys, time
from pathlib import Path
from random import Random
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from tetris_gymnasium.envs.tetris import Tetris
from tetris_agent.planner import TacticalPlanner, PlannerStatus
from tetris_agent.rl.runner import RLRunner
from tetris_agent.rl.policy import EpsilonGreedyPolicy, valid_ids
from tetris_agent.rl.state import encode_states, SPACE, geometric_mask
from tetris_agent.rl.evaluation import (load_checkpoint_network, serialize_state,
                                       deserialize_state, state_hash)
from tetris_agent.rl.evaluation_build import checked_plan, native_context
from tetris_agent.rl.reward import hole_count

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/geometric/penalty_020/seed_42/500k'
MODELS = ('M150', 'M250')
PATHS = {m: RUN / f'checkpoint_{k:09d}.pt' for m,k in zip(MODELS,(150000,250000))}
def dump(v): return json.dumps(v, sort_keys=True, separators=(',',':'))
def digest(v): return hashlib.sha256(dump(v).encode()).hexdigest()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(path, rows):
    if not rows: raise ValueError('Empty table')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def known_states():
    result={}
    for p in (ROOT/'benchmarks/ddqn_frozen_policy_gameplay_20261005/trap_entry_contexts').glob('*.json'):
        d=json.loads(p.read_text(encoding='utf-8'))
        s=deserialize_state(d['state']); h=state_hash(s)
        assert h==d['state_hash']
        assert list(np.flatnonzero(geometric_mask(s)))==d['valid_action_ids']
        labels={a:('R' if a in d['reachable_action_ids'] else 'U') for a in d['valid_action_ids']}
        r=result.setdefault(h,dict(labels=labels,original_ids=set(),known_u_ids=set()))
        assert r['labels']==labels
        r['original_ids'].add(d['action_id']); r['known_u_ids'].update(a for a,v in labels.items() if v=='U')
    return result
def collect(source, seeds, cap, directory):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    out=Path(directory); out.mkdir(parents=True,exist_ok=True)
    hashes={m:sha(p) for m,p in PATHS.items()}
    nets={m:load_checkpoint_network(p).eval().requires_grad_(False) for m,p in PATHS.items()}
    weights={m:{k:v.clone() for k,v in n.state_dict().items()} for m,n in nets.items()}
    known=known_states(); cache={}; rows=[]; episodes=[]; times=[]
    stats=dict(trajectory_calls=0,extra_selected_calls=0,duplicate_validation_calls=0,
               reused_prior_labels=0,same_state_id_checks=0,branch_rng_checks=0,
               label_order_checks=0,duplicate_states_validated=0,different_native_validated=0)
    basepolicy=EpsilonGreedyPolicy(nets[source],seed=44)
    planner=TacticalPlanner()
    context_path=out/f'{source}_native_contexts.jsonl.gz'
    started=time.perf_counter()
    pending={}; streak=0
    class ObservedPolicy:
        def select(self, state, mask):
            nonlocal pending
            # IDs selected before any Planner or label-cache access.
            ids={}
            x=encode_states([state])
            for m,n in nets.items():
                q=n(x)[0]
                ids[m]=int(q.masked_fill(~torch.as_tensor(mask),-torch.inf).argmax())
            shadow=Random(); shadow.setstate(basepolicy._rng.getstate())
            random_branch=shadow.random()<.05
            expected=shadow.choice(valid_ids(mask)) if random_branch else ids[source]
            chosen=basepolicy.select(state,mask,epsilon=.05)
            assert chosen==expected and shadow.getstate()==basepolicy._rng.getstate()
            stats['branch_rng_checks']+=1
            h=state_hash(state)
            pending=dict(state=state,state_hash=h,greedy_ids=ids,chosen=chosen,
                         branch='epsilon_random' if random_branch else 'greedy',selected=True)
            return chosen
    class ObservedPlanner:
        def plan(self, env, target):
            assert pending['selected']
            stats['label_order_checks']+=1
            h=pending['state_hash']; ids=pending['greedy_ids']
            context=native_context(env); ch=digest(context)
            pending['context']=context; pending['context_hash']=ch
            plan=checked_plan(env,target,planner,times)
            stats['trajectory_calls']+=1
            actual='U' if plan.status==PlannerStatus.UNREACHABLE else 'R'
            pending['outcome']=actual
            if h in cache:
                c=cache[h]
                assert c['ids']==ids
                stats['same_state_id_checks']+=1
                c['seen']+=1
                sample=c['seen']==2 or (ch!=c['context_hash'] and not c['different_validated'])
                if sample:
                    for a in sorted(set(ids.values())):
                        p=checked_plan(env,SPACE.decode(a),planner,times)
                        value='U' if p.status==PlannerStatus.UNREACHABLE else 'R'
                        stats['duplicate_validation_calls']+=1
                        assert c['labels'][a]==value, ('SEMANTIC_LABEL_CONFLICT',h,a)
                    stats['duplicate_states_validated']+=1
                    if ch!=c['context_hash']:
                        c['different_validated']=True; stats['different_native_validated']+=1
            else:
                c=dict(ids=ids.copy(),labels={},seen=1,context_hash=ch,different_validated=False)
                for a in sorted(set(ids.values())):
                    if h in known:
                        c['labels'][a]=known[h]['labels'][a]
                        stats['reused_prior_labels']+=1
                    elif a==pending['chosen']:
                        c['labels'][a]=actual
                    else:
                        p=checked_plan(env,SPACE.decode(a),planner,times)
                        c['labels'][a]='U' if p.status==PlannerStatus.UNREACHABLE else 'R'
                        stats['extra_selected_calls']+=1
                cache[h]=c
            if pending['chosen'] in c['labels']:
                assert actual==c['labels'][pending['chosen']], ('SEMANTIC_LABEL_CONFLICT',h,pending['chosen'])
            pending['labels']={m:c['labels'][ids[m]] for m in MODELS}
            return plan
    env=Tetris(gravity=True)
    runner=RLRunner(ObservedPolicy(),reward_mode='minimal',unreachable_penalty=-.20,planner=ObservedPlanner())
    try:
        with gzip.open(context_path,'wt',encoding='utf-8') as context_file:
            for seed in seeds:
                local=0; streak=0
                def transition(t):
                    nonlocal local,streak
                    local+=1; u=pending['outcome']=='U'; streak=streak+1 if u else 0
                    s=pending['state']; h=pending['state_hash']; semantic=serialize_state(s)
                    assert state_hash(t.state)==h
                    assert (state_hash(t.next_state)==h) if u else True
                    ref=len(rows)+1
                    context_file.write(dump(dict(arm_occurrence_index=ref,episode_seed=seed,
                        decision_index=local,state_hash=h,native_context=pending['context']))+'\n')
                    k=known.get(h)
                    r=dict(source_policy=source,episode_seed=seed,arm_occurrence_index=ref,
                        global_occurrence_index=0,decision_index=local,state_hash=h,
                        encoded_state=dump(encode_states([s])[0].tolist()),board=dump(semantic['board']),
                        current_piece=s.current_piece,next4=dump(list(s.next_queue)),
                        chosen_action_id=t.action_id,selection_branch=pending['branch'],
                        chosen_planner_outcome=pending['outcome'],reward=t.reward,
                        self_loop=u,terminated=t.terminated,truncated=t.truncated,
                        consecutive_U=streak,next_state_hash=state_hash(t.next_state),
                        native_context_file=context_path.name,native_context_record=ref,
                        native_context_hash=pending['context_hash'],holes=hole_count(s))
                    for m in MODELS:
                        a=pending['greedy_ids'][m]; r[m+'_greedy_id']=a
                        r[m+'_label']=pending['labels'][m]
                        r[m+'_historical_original_U']=bool(k and a in k['original_ids'])
                        r[m+'_historical_known_competing_U']=bool(k and a in k['known_u_ids'] and a not in k['original_ids'])
                    rows.append(r)
                metrics=runner.run(env,max_decisions=cap,env_seed=seed,on_transition=transition)
                assert not metrics.diagnostic, metrics.diagnostic
                assert local==metrics.decision_count==metrics.stored_transitions
                episodes.append(dict(source_policy=source,episode_seed=seed,**asdict(metrics)))
                print(dump(dict(source=source,seed=seed,decisions=local,total=len(rows),
                                seconds=round(time.perf_counter()-started,2))),flush=True)
    except Exception as exc:
        (out/f'{source}_STOP.json').write_text(dump(dict(error=str(exc),state_hash=pending.get('state_hash'))))
        if rows: write_csv(out/f'{source}_partial.csv',rows)
        raise
    finally: env.close()
    for m,n in nets.items():
        assert all(torch.equal(weights[m][k],v) for k,v in n.state_dict().items())
        assert all(p.grad is None and not p.requires_grad for p in n.parameters())
        assert sha(PATHS[m])==hashes[m]
    write_csv(out/f'{source}_occurrences.csv',rows)
    (out/f'{source}_episodes.json').write_text(dump(episodes),encoding='utf-8')
    validation=dict(source=source,seeds=seeds,cap=cap,seconds=time.perf_counter()-started,
        counts=stats,checkpoint_hashes=hashes,weights_unchanged=True,no_gradients=True,
        optimizer_instances=0,replay_instances=0,learner_instances=0,target_network_instances=0,
        planner_call_count=len(times),planner_seconds=sum(times),occurrences=len(rows),
        unique_states=len(cache),trajectory_digest=digest(rows))
    (out/f'{source}_validation.json').write_text(dump(validation),encoding='utf-8')
    return validation
def summarize(directory):
    import csv,json,math,sys,collections,gzip,hashlib,subprocess
    from pathlib import Path
    import numpy as np
    sys.path.insert(0,str(Path.cwd()/'src'))
    from tetris_agent.rl.evaluation import deserialize_state,state_hash
    from tetris_agent.rl.state import encode_states,geometric_mask
    P=Path(directory)
    models=('M150','M250')
    def write(name,rows):
        with (P/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def read(name):return list(csv.DictReader((P/name).open(encoding='utf-8',newline='')))
    def jread(name):return json.loads((P/name).read_text(encoding='utf-8'))
    def true(x):return x=='True'
    def ratio(n,d):return 100*n/d if d else None
    rows=[];sets={};by={};contexts={}
    validation=dict()
    for m in models:
        rr=read(m+'_occurrences.csv');by[m]=rr;sets[m]={r['state_hash'] for r in rr}
        for r in rr:
            r['global_occurrence_index']=len(rows)+1;rows.append(r)
        vv=jread(m+'_validation.json')
        ee=jread(m+'_episodes.json')
        assert len(rr)==sum(e['decision_count'] for e in ee)==vv['occurrences']
        assert sum(r['chosen_planner_outcome']=='U' for r in rr)==sum(e['unreachable_count'] for e in ee)
        assert all(e['diagnostic'] is None for e in ee)
        assert sum(e['terminated']+e['truncated'] for e in ee)==len(vv['seeds'])
        with gzip.open(P/(m+'_native_contexts.jsonl.gz'),'rt',encoding='utf-8') as f:
            cr=[json.loads(line) for line in f]
        assert len(cr)==len(rr)
        for r,c in zip(rr,cr):
            assert r['state_hash']==c['state_hash']
            dh=hashlib.sha256(json.dumps(c['native_context'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert dh==r['native_context_hash']
        validation[m]=vv
    groups=collections.defaultdict(list)
    for r in rows:groups[r['state_hash']].append(r)
    unique=[]
    for h,rr in sorted(groups.items()):
        f=rr[0];s=deserialize_state(dict(board=json.loads(f['board']),current_piece=int(f['current_piece']),next_queue=json.loads(f['next4'])))
        assert state_hash(s)==h
        assert np.array_equal(encode_states([s])[0].numpy(),json.loads(f['encoded_state']))
        mask=geometric_mask(s)
        for r in rr:
            assert all(r[k]==f[k] for k in ('encoded_state','board','current_piece','next4','M150_greedy_id','M250_greedy_id','M150_label','M250_label'))
            assert mask[int(r['chosen_action_id'])]
            for m in models:assert mask[int(r[m+'_greedy_id'])]
            if r['selection_branch']=='greedy':
                assert r['chosen_action_id']==r[r['source_policy']+'_greedy_id']
                assert r['chosen_planner_outcome']==r[r['source_policy']+'_label']
            assert true(r['self_loop'])==(r['chosen_planner_outcome']=='U')
            if true(r['self_loop']):assert r['next_state_hash']==h and float(r['reward'])==-.2
        u=dict(state_hash=h,board=f['board'],current_piece=f['current_piece'],next4=f['next4'],encoded_state=f['encoded_state'],
            geometric_valid_ids=json.dumps(np.flatnonzero(mask).tolist()),holes=f['holes'],
            native_context_file=f['native_context_file'],native_context_record=f['native_context_record'],
            native_context_variants=len({r['native_context_hash'] for r in rr}))
        counts={m:sum(r['source_policy']==m for r in rr) for m in models}
        u['subset']='shared' if all(counts.values()) else ('D150-only' if counts['M150'] else 'D250-only')
        for m in models:
            u[m+'_occurrences']=counts[m]
            u[m+'_actual_U_occurrences']=sum(r['source_policy']==m and r['chosen_planner_outcome']=='U' for r in rr)
            for suffix in ('greedy_id','label','historical_original_U','historical_known_competing_U'):u[m+'_'+suffix]=f[m+'_'+suffix]
        unique.append(u)
    write('occurrences.csv',rows);write('unique_states.csv',unique)
    cells=[];rates={}
    for weighting in ('occurrence','unique'):
        rates[weighting]={}
        for model in models:
            for source in models:
                selected=by[source] if weighting=='occurrence' else [u for u in unique if u[source+'_occurrences']>0]
                nu=sum(r[model+'_label']=='U' for r in selected);n=len(selected)
                cell={('M150','M150'):'A',('M150','M250'):'B',('M250','M150'):'C',('M250','M250'):'D'}[model,source]
                cells.append(dict(weighting=weighting,cell=cell,model=model,distribution='D'+source[1:],
                    occurrence_count=len(by[source]),unique_state_count=len(sets[source]),denominator=n,
                    reachable_count=n-nu,unreachable_count=nu,U_percent=ratio(nu,n)))
                rates[weighting][cell]=ratio(nu,n)
    write('cross_visitation_2x2.csv',cells)
    effects=[]
    for weighting,r in rates.items():
        for kind,label,x,y in [('distribution','M150 B-A','B','A'),('distribution','M250 D-C','D','C'),
                               ('model','D150 C-A','C','A'),('model','D250 D-B','D','B')]:
            effects.append(dict(weighting=weighting,kind=kind,contrast=label,delta_percentage_points=r[x]-r[y]))
    write('matched_contrasts.csv',effects)
    s1,s2=sets.values();inter=s1&s2;union=s1|s2
    overlap=dict(D150=len(s1),D250=len(s2),intersection=len(inter),union=len(union),
                 Jaccard=len(inter)/len(union),D150_only=len(s1-s2),D250_only=len(s2-s1))
    write('state_overlap.csv',[dict(metric=k,value=v) for k,v in overlap.items()])
    subset=[]
    for name in ('D150-only','shared','D250-only'):
        selected=[u for u in unique if u['subset']==name]
        for model in models:
            nu=sum(u[model+'_label']=='U' for u in selected)
            subset.append(dict(subset=name,model=model,unique_states=len(selected),unreachable_count=nu,
                               reachable_count=len(selected)-nu,U_percent=ratio(nu,len(selected)),
                               both_U_states=sum(all(u[m+'_label']=='U' for m in models) for u in selected)))
    write('subset_difficulty.csv',subset)
    mult=[];top=[]
    for source in models:
        uu=[u for u in unique if u[source+'_occurrences']]
        for metric in ('actual_chosen','M150_greedy','M250_greedy'):
            weights={u['state_hash']:u[source+'_actual_U_occurrences'] if metric=='actual_chosen' else
                     (u[source+'_occurrences'] if u[metric[:4]+'_label']=='U' else 0) for u in uu}
            ranked=sorted(weights,key=lambda h:(-weights[h],h));total=sum(weights.values())
            nU=sum(v>0 for v in weights.values())
            base=dict(source=source,metric=metric,occurrences=len(by[source]),unique_states=len(uu),
                U_occurrences=total,unique_U_states=nU,occurrence_U_percent=ratio(total,len(by[source])),
                unique_U_percent=ratio(nU,len(uu)),max_state_multiplicity=max(u[source+'_occurrences'] for u in uu),
                mean_multiplicity=len(by[source])/len(uu))
            for frac in (.01,.05,.10):
                k=math.ceil(frac*len(uu))
                base[f'top_{int(frac*100)}pct_state_count']=k
                base[f'top_{int(frac*100)}pct_U_share_percent']=ratio(sum(weights[h] for h in ranked[:k]),total)
            mult.append(base)
        for u in sorted(uu,key=lambda u:(-u[source+'_occurrences'],u['state_hash']))[:20]:
            top.append(dict(source=source,state_hash=u['state_hash'],occurrences=u[source+'_occurrences'],
                actual_U=u[source+'_actual_U_occurrences'],M150_label=u['M150_label'],M250_label=u['M250_label']))
    write('multiplicity_summary.csv',mult);write('top_repeated_states.csv',top)
    explore=[]
    for source in models:
        for branch in ('greedy','epsilon_random'):
            rr=[r for r in by[source] if r['selection_branch']==branch];nu=sum(r['chosen_planner_outcome']=='U' for r in rr)
            explore.append(dict(source=source,branch=branch,occurrences=len(rr),U_count=nu,R_count=len(rr)-nu,
                branch_U_percent=ratio(nu,len(rr)),share_all_actual_U_percent=ratio(nu,sum(r['chosen_planner_outcome']=='U' for r in by[source])),
                random_coincides_source_greedy=sum(r['chosen_action_id']==r[source+'_greedy_id'] for r in rr) if branch=='epsilon_random' else 0))
    write('exploration_summary.csv',explore)
    desc=[]
    for source in models:
        for weighting in ('occurrence','unique'):
            rr=by[source] if weighting=='occurrence' else [u for u in unique if u[source+'_occurrences']]
            desc.append(dict(source=source,weighting=weighting,feature='holes_mean',value=np.mean([int(r['holes']) for r in rr])))
            for position in range(5):
                cc=collections.Counter(int(r['current_piece']) if position==0 else json.loads(r['next4'])[position-1] for r in rr)
                for piece in range(7):
                    desc.append(dict(source=source,weighting=weighting,feature=('current' if position==0 else f'next{position}')+f'_piece_{piece}_fraction',value=cc[piece]/len(rr)))
    write('distribution_descriptors.csv',desc)
    summary=dict(rates=rates,overlap=overlap,effects=effects,multiplicity=mult,exploration=explore,
                 subset=subset,top=top,episodes={m:jread(m+'_episodes.json') for m in models},
                 arm_validation=validation,unique_union=len(unique),occurrences=len(rows),
                 different_native_semantic_states=sum(u['native_context_variants']>1 for u in unique))
    (P/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('rates','overlap','effects','multiplicity','exploration','subset','different_native_semantic_states')}))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True)
    ap.add_argument('--seeds',type=int,default=2); ap.add_argument('--cap',type=int,default=40)
    ap.add_argument('--summarize',action='store_true')
    args=ap.parse_args(); seeds=list(range(1000,1000+args.seeds))
    if args.summarize:
        summarize(args.output)
        return
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(collect,m,seeds,args.cap,args.output) for m in MODELS]
        result=[j.result() for j in jobs]
    print(dump(result),flush=True)
if __name__=='__main__': main()
