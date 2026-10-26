"""EV hosting extension: synthetic service contracts and perfect-information bounds.

Frozen protocol (2026-10-26, before these new EV outcomes):
6 existing SimBench archetypes x OU/AU/OP100/AP100 x 2024/2025 x
0/2/5 installed 1000 L modules x immediate/managed charging.  A module is
23.26 kWh thermal and 20 kW thermal; one module is unavailable but kept hot.
The 0-module arm has no tank.  The hot spare is replenished through the same
eta=.95 charge transfer as available inventory (a small correction relative
to the older heat-only experiment's direct-maintenance convention).
Existing 2024 heat scale is held fixed, rather
than re-normalised to admit EVs.  Each synthetic equivalent EV needs 12 kWh
grid-side per day, can draw 3.7 kW, arrives at model-clock hour 18 and leaves
at hour 07.  Every day has the same contract; the year is periodic.  No V2G,
travel observations, battery efficiency, or battery ageing are inferred.

Maximum continuous equivalent EV count is the objective.  Both EV management
and TES receive the complete annual chronology, so these are operational
upper bounds rather than deployable controllers.  Daily EV energy, hourly
heat service, power and inventory limits are hard constraints.  Scalar AC
envelopes refer to the same remote connection and common assumed PF=.97.
The displayed robust hosting value is the smaller of the two development
year values; 2026 is deliberately not re-used as a new holdout.

Prespecified sensitivity: OU, installed modules 0/5, both charge modes,
both development years, and (8,20) kWh/day at 18--07 or 12 kWh/day at
16--08 and 20--06.  All selected cases, including failures, are retained.

Run --pilot first; --main resumes a complete main matrix; --sensitivity
resumes sensitivities; --ac validates six limiting OU/5-module/managed
witnesses with the existing independent annual solver.  Results stay in
this directory's results_ev; all source caches are read-only.  This protocol
was specified before the EV pilot; it is not a public preregistration.
Set PROJECT09_EV_DATA_ROOT to a portable input directory containing
development_cache.npz and catalog/operations_*.csv,capacity_*.npz.  Core LPs
need only Python, NumPy, pandas and SciPy; --ac additionally needs the full
original project environment and network libraries.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path(os.environ.get('PROJECT09_EV_DATA_ROOT',str(HERE.parent)))
LEGACY_ROOT=HERE.parent
OUT=HERE/'results_ev'
VERSION='ev_v2_spare_eta'

def profiles(values,lo=1.,hi=8.):
    """Exact cohort extraction from revision_v2/measured_planning.py.

    values[:, :, 0/1] are heat/electric watts for 24 systems; summing then
    dividing by 1e6 yields MWh per one-hour interval. COP-eligible heat and
    electricity have ratio within [1,8]; other electricity remains fixed.
    """
    h=values[:,:,0];e=values[:,:,1]
    ratio=np.divide(h,e,out=np.zeros_like(h),where=e>0)
    eligible=(h>0)&(e>0)&(ratio>=lo)&(ratio<=hi)
    heat=np.where(eligible,h,0).sum(0)/1e6
    flex=np.where(eligible,e,0).sum(0)/1e6
    fixed=np.where(eligible,0,e).sum(0)/1e6
    cop=np.divide(heat,flex,out=np.ones_like(heat),where=flex>0)
    return heat,cop,fixed,flex

KEYS=['rural1','rural2','rural3','semiurb4','semiurb5','urban6']
ASSETS=['OU','AU','OP100','AP100']
ALPHA=.8; RHO=.998; ETA=.95
UNIT_E=23.26; UNIT_Q=20.
N=8760

def contract(arrival=18,departure=7,energy=12.,rate=3.7):
    """Model-clock hours; cyclic overnight sessions, not measured commuting."""
    hour=np.arange(N)%24
    day=np.arange(N)//24
    available=(hour>=arrival)|(hour<departure)
    sid=(day-(hour<departure))%365
    shape=np.zeros(N)
    for d in range(365):
        ix=(d*24+np.arange(arrival,24+departure))%N
        rem=energy
        for t in ix:
            shape[t]=min(rate,rem);rem-=shape[t]
        if rem>1e-10:raise ValueError('Service exceeds connection window')
    assert abs(shape.sum()-365*energy)<1e-8
    assert np.max(abs(np.bincount(sid,weights=shape,minlength=365)-energy))<1e-10
    return available,sid,shape

def model_inputs(key,year,modules):
    z=np.load(ROOT/'development_cache.npz')
    h,c,b,e=profiles(z['values'])
    row=pd.read_csv(ROOT/'catalog'/f'operations_{key}_0.00.csv').iloc[0]
    base=float(row.base_scale);pmax=float(row.pmax)*1000
    sl=slice(0,N) if year==2024 else slice(N,2*N)
    hh=h[sl]*base*1000;cc=c[sl];bb=b[sl]*base*1000
    if modules:bb=bb+UNIT_E*(1-RHO)/(ETA*ALPHA*cc)
    return hh,cc,bb,pmax,base

def no_storage(h,c,b,cap,mode,energy,rate,arrival,departure):
    """Exact one-dimensional feasibility for the zero-TES arms."""
    start=time.perf_counter();avail,sid,shape=contract(arrival,departure,energy,rate)
    hp=h/c;margin=cap-b-hp
    if margin.min() < -1e-6:
        return dict(status='heat_infeasible',message='Fixed heat alone violates envelope',seconds=time.perf_counter()-start)
    margin=np.maximum(margin,0)
    if mode=='immediate':
        ix=shape>0
        count=float(np.min(margin[ix]/shape[ix]))
        ev=shape*count
    else:
        upper=float(np.min(np.bincount(sid,weights=margin*avail,minlength=365)/energy))
        lo=0.;hi=upper
        for _ in range(65):
            mid=(lo+hi)/2
            room=np.minimum(margin,mid*rate)*avail
            good=np.bincount(sid,weights=room,minlength=365)>=mid*energy-1e-10
            if good.all():lo=mid
            else:hi=mid
        count=lo*(1-1e-10)
        # A deterministic earliest feasible witness under the complete envelope.
        ev=np.zeros(N)
        for d in range(365):
            ix=(d*24+np.arange(arrival,24+departure))%N
            remain=count*energy
            for t in ix:
                ev[t]=min(margin[t],count*rate,remain);remain-=ev[t]
            assert remain<1e-6
    return dict(status='optimal',count=count,hp=hp,ev=ev,soc=np.zeros(N),charge=np.zeros(N),discharge=np.zeros(N),seconds=time.perf_counter()-start,secondary=False)

def with_storage(h,c,b,cap,pmax,modules,mode,energy,rate,arrival,departure,fixed_count=None):
    """Sparse annual LP; units kW/kWh.  The EV count itself is a variable."""
    start=time.perf_counter();avail,sid,shape=contract(arrival,departure,energy,rate)
    t=np.arange(N);managed=mode=='managed';nv=(5 if managed else 4)*N+1;nc=nv-1
    E=(modules-1)*UNIT_E;Q=(modules-1)*UNIT_Q
    # hp electricity, thermal charge, thermal discharge, end-of-hour inventory.
    rows=np.r_[t,t,t,N+t,N+t,N+t,N+t]
    cols=np.r_[t,N+t,2*N+t,3*N+t,3*N+(t-1)%N,N+t,2*N+t]
    vals=np.r_[c,np.full(N,-1/ALPHA),np.ones(N),np.ones(N),np.full(N,-RHO),np.full(N,-ETA),np.full(N,1/ETA)]
    if managed:
        ix=np.where(avail)[0]
        rows=np.r_[rows,2*N+sid[ix],2*N+np.arange(365)]
        cols=np.r_[cols,4*N+ix,np.full(365,nc)]
        vals=np.r_[vals,np.ones(len(ix)),np.full(365,-energy)]
    eq=coo_matrix((vals,(rows,cols)),shape=(2*N+(365 if managed else 0),nv)).tocsr()
    rhs=np.r_[h,np.zeros(N+(365 if managed else 0))]
    ur=np.r_[t,t];uc=np.r_[t,4*N+t if managed else np.full(N,nc)]
    uv=np.r_[np.ones(N),np.ones(N) if managed else shape]
    if managed:
        ix=np.where(avail)[0]
        ur=np.r_[ur,N+np.arange(len(ix)),N+np.arange(len(ix))]
        uc=np.r_[uc,4*N+ix,np.full(len(ix),nc)]
        uv=np.r_[uv,np.ones(len(ix)),np.full(len(ix),-rate)]
    ub=coo_matrix((uv,(ur,uc)),shape=(N+(len(ix) if managed else 0),nv)).tocsr()
    limit=np.r_[cap-b,np.zeros(len(ix))] if managed else cap-b
    bounds=list(zip(np.zeros(N),np.where(h>0,pmax,0)))
    bounds+=list(zip(np.zeros(N),np.where(h>0,Q,0)))*2
    bounds += [(0,E)]*N
    if managed:bounds+=list(zip(np.zeros(N),np.where(avail,np.inf,0)))
    bounds += [(0,None)]
    objective=np.zeros(nv);objective[nc]=-1.
    if fixed_count is not None:
        # Optional exploratory fixed-service evaluation.  This does not change
        # default maximum-hosting experiments or write their output files.
        bounds[nc]=(float(fixed_count),float(fixed_count))
        objective[:N]=1.;objective[N:3*N]=1e-5;objective[nc]=0.
    res=linprog(objective,A_ub=ub,b_ub=limit,A_eq=eq,b_eq=rhs,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not res.success:return dict(status='infeasible' if res.status==2 else 'solver_failed',message=res.message,seconds=time.perf_counter()-start)
    count=float(res.x[nc]);secondary=False
    # Eliminate arbitrary LP degeneracy with a fixed hosting count and positive
    # electricity/cycling costs.  Never report simultaneous cycling as hardware.
    fixed=list(bounds);fixed[nc]=(max(0,count-1e-7),max(0,count-1e-7))
    cost=np.zeros(nv);cost[:N]=1.;cost[N:3*N]=1e-5
    op=res if fixed_count is not None else linprog(cost,A_ub=ub,b_ub=limit,A_eq=eq,b_eq=rhs,bounds=fixed,method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not op.success:
        return dict(status='witness_failed',message=op.message,count=count,seconds=time.perf_counter()-start)
    x=op.x;secondary=True
    return dict(status='optimal',count=count,witness_count=float(x[nc]),hp=x[:N],ev=x[4*N:5*N] if managed else shape*x[nc],soc=x[3*N:4*N],charge=x[N:2*N],discharge=x[2*N:3*N],seconds=time.perf_counter()-start,secondary=secondary)

def validate(o,h,c,b,cap,pmax,modules,mode,energy,rate,arrival,departure):
    avail,sid,shape=contract(arrival,departure,energy,rate)
    count=o.get('witness_count',o['count']);hp=o['hp'];ev=o['ev'];soc=o['soc'];ch=o['charge'];dis=o['discharge']
    E=max(0,modules-1)*UNIT_E;Q=max(0,modules-1)*UNIT_Q
    residual=dict(heat_balance_kw=float(abs(c*hp-ch/ALPHA+dis-h).max()),inventory_balance_kwh=float(abs(soc-RHO*np.roll(soc,1)-ETA*ch+dis/ETA).max()),departure_energy_kwh=float(abs(np.bincount(sid,weights=ev,minlength=365)-count*energy).max()),network_excess_kw=float(max(0,(hp+ev+b-cap).max())),ev_rate_excess_kw=float(max(0,(ev-count*rate*avail).max())),hp_rate_excess_kw=float(max(0,(hp-pmax).max())),soc_min_kwh=float(soc.min()),soc_excess_kwh=float(max(0,soc.max()-E)),transfer_excess_kw=float(max(0,ch.max()-Q,dis.max()-Q)),simultaneous_cycling_kw=float(np.minimum(ch,dis).max()),ev_outside_window_kw=float(ev[~avail].max()),thermal_electricity_mwh=float((hp+b).sum()/1000),ev_electricity_mwh=float(ev.sum()/1000),peak_connection_kw=float((hp+ev+b).max()))
    critical=[residual[k] for k in ['heat_balance_kw','inventory_balance_kwh','departure_energy_kwh','network_excess_kw','ev_rate_excess_kw','hp_rate_excess_kw','soc_excess_kwh','transfer_excess_kw','simultaneous_cycling_kw','ev_outside_window_kw']]
    assert max(critical)<2e-5 and residual['soc_min_kwh']>-2e-5,residual
    if mode=='immediate':assert np.max(abs(ev-shape*count))<2e-5
    return residual

def run_case(key,asset,year,modules,mode,energy=12.,arrival=18,departure=7,sensitivity='main'):
    rate=3.7;h,c,b,pmax,base=model_inputs(key,year,modules)
    cap=np.load(ROOT/'catalog'/f'capacity_{key}_{asset}.npz')['cap']*1000
    row=dict(model_version=VERSION,network=key,asset=asset,year=year,installed_modules=modules,available_thermal_kwh=max(0,modules-1)*UNIT_E,available_transfer_kw=max(0,modules-1)*UNIT_Q,mode=mode,energy_kwh=energy,rate_kw=rate,arrival=arrival,departure=departure,sensitivity=sensitivity,base_scale=base)
    o=no_storage(h,c,b,cap,mode,energy,rate,arrival,departure) if modules==0 else with_storage(h,c,b,cap,pmax,modules,mode,energy,rate,arrival,departure)
    row.update({k:v for k,v in o.items() if np.isscalar(v)})
    if o['status']=='optimal':
        row.update(validate(o,h,c,b,cap,pmax,modules,mode,energy,rate,arrival,departure))
        row['whole_ev_floor']=int(math.floor(o.get('witness_count',o['count'])))
        # Keep all annual witnesses: modest outputs, and no hidden selected trace.
        label=f'{key}_{asset}_{year}_{modules}_{mode}_{energy:g}_{arrival}_{departure}'
        dest=OUT/'witnesses'/f'{label}.npz'
        np.savez_compressed(dest,hp_kw=o['hp'],ev_kw=o['ev'],soc_kwh=o['soc'],charge_kw=o['charge'],discharge_kw=o['discharge'],fixed_kw=b,cap_kw=cap,count=o.get('witness_count',o['count']),heat_kwh=h,cop=c)
        row['witness']=dest.name
    return row

def aggregate():
    f=OUT/'main.csv'
    if f.exists():
        df=pd.read_csv(f)
        cols=['network','asset','installed_modules','mode','energy_kwh','arrival','departure']
        out=[]
        for label,group in df.groupby(cols):
            row=dict(zip(cols,label));coverage=set(group.year)=={2024,2025} and len(group)==2
            passed=coverage and bool((group.status=='optimal').all())
            row.update(development_years=int(group.year.nunique()),two_year_feasible=passed,status='optimal' if passed else '|'.join(sorted(set(group.status))))
            if passed:
                feasible=group.witness_count.fillna(group['count']) if 'witness_count' in group else group['count']
                row.update(equivalent_evs=float(group['count'].min()),feasible_equivalent_evs=float(feasible.min()),whole_ev_floor=int(np.floor(feasible.min())),worst_network_excess_kw=float(group.network_excess_kw.max()),worst_heat_balance_kw=float(group.heat_balance_kw.max()),worst_departure_energy_kwh=float(group.departure_energy_kwh.max()))
            else:
                row.update(equivalent_evs=np.nan,feasible_equivalent_evs=np.nan,whole_ev_floor=np.nan)
            out.append(row)
        agg=pd.DataFrame(out)
        agg.to_csv(OUT/'hosting_two_year.csv',index=False)
    manifest=dict(model_version=VERSION,protocol='Synthetic daily grid-side EV contract, independently periodic 8760-hour perfect-information capacity bounds. Min of 2024/2025, not continuous two-year SOC or a 2026 holdout. No EV economic optimisation.',modules=[0,2,5],assets=ASSETS,keys=KEYS,spare_maintenance='unitE*(1-rho)/(eta*alpha*COP)',units='kW, kWh in solver; equivalent continuous EV count in hosting results',raw_inputs_modified=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def write_summary():
    """Sourceable numeric summaries; no outcome filtering or economic claims."""
    if not (OUT/'hosting_two_year.csv').exists():return
    main=pd.read_csv(OUT/'main.csv');agg=pd.read_csv(OUT/'hosting_two_year.csv')
    summary=dict(model_version=VERSION,main_cases=len(main),expected_main_cases=288,main_statuses=main.status.value_counts().to_dict(),two_year_groups=len(agg),two_year_feasible=int(agg.two_year_feasible.sum()),ou_comparison=[],switch_intervals=[],asset_metadata=[])
    for key in KEYS:
        row=dict(network=key)
        for m in [0,2,5]:
            for mode in ['immediate','managed']:
                x=agg[(agg.network==key)&(agg.asset=='OU')&(agg.installed_modules==m)&(agg['mode']==mode)]
                if len(x)==1 and bool(x.iloc[0].two_year_feasible):
                    row[f'{mode}_{m}']=float(x.iloc[0].equivalent_evs)
                    row[f'{mode}_{m}_whole']=int(x.iloc[0].whole_ev_floor)
        summary['ou_comparison'].append(row)
        if all(k in row for k in ['managed_0','managed_5']):
            row['tes_increment_to_managed']=row['managed_5']-row['managed_0']
            row['tes_increment_percent']=100*(row['managed_5']/row['managed_0']-1)
            low=row['managed_0_whole']+1;high=row['managed_5_whole']
            summary['switch_intervals'].append(dict(network=key,synthetic_ev_first=low,synthetic_ev_last=high,interpretation='Original grid can serve these integer fleets with managed EV+5 TES modules but not managed EV alone; capacity-only, not least-cost selection.'))
        for asset in ASSETS:
            z=np.load(ROOT/'catalog'/f'capacity_{key}_{asset}.npz')
            summary['asset_metadata'].append(dict(network=key,asset=asset,transformer_mva=float(z['trafo']),parallel_route_km=float(z['length']),connection_bus=int(z['bus'])))
    residuals=['heat_balance_kw','inventory_balance_kwh','departure_energy_kwh','network_excess_kw','ev_rate_excess_kw','hp_rate_excess_kw','soc_excess_kwh','transfer_excess_kw','simultaneous_cycling_kw','ev_outside_window_kw']
    frames=[main]
    if (OUT/'sensitivity.csv').exists():
        sens=pd.read_csv(OUT/'sensitivity.csv');frames.append(sens)
        summary['sensitivity_cases']=len(sens);summary['sensitivity_statuses']=sens.status.value_counts().to_dict()
        rows=[]
        for label,g in sens.groupby(['network','installed_modules','mode','energy_kwh','arrival','departure']):
            row=dict(zip(['network','installed_modules','mode','energy_kwh','arrival','departure'],label))
            passed=len(g)==2 and set(g.year)=={2024,2025} and bool((g.status=='optimal').all())
            row.update(two_year_feasible=passed,equivalent_evs=float(g['count'].min()) if passed else np.nan)
            rows.append(row)
        ss=pd.DataFrame(rows);ss.to_csv(OUT/'sensitivity_two_year.csv',index=False)
        sensitivity_summary=[]
        for label,g in ss.groupby(['energy_kwh','arrival','departure']):
            pivot=g.pivot(index='network',columns=['installed_modules','mode'],values='equivalent_evs')
            delta=pivot[5,'managed']-pivot[0,'managed']
            percent=100*delta/pivot[0,'managed']
            sensitivity_summary.append(dict(energy_kwh=float(label[0]),arrival=int(label[1]),departure=int(label[2]),managed_only_min=float(pivot[0,'managed'].min()),managed_only_max=float(pivot[0,'managed'].max()),joint_min=float(pivot[5,'managed'].min()),joint_max=float(pivot[5,'managed'].max()),tes_increment_min=float(delta.min()),tes_increment_max=float(delta.max()),tes_increment_percent_min=float(percent.min()),tes_increment_percent_max=float(percent.max())))
        summary['sensitivity_summary']=sensitivity_summary
    both=pd.concat(frames,ignore_index=True)
    summary['maximum_residuals']={k:float(both[k].max()) for k in residuals}
    summary['total_case_seconds']=float(both.seconds.sum())
    summary['binding_sessions']=binding_sessions(main)
    if (OUT/'ac_validation.csv').exists():
        ac=pd.read_csv(OUT/'ac_validation.csv')
        summary['ac_validation']=dict(records=len(ac),intervals=int(ac.intervals.sum()),violations=int(ac.violations.sum()),minimum_voltage_pu=float(ac.min_voltage.min()),maximum_voltage_pu=float(ac.max_voltage.max()),maximum_line_percent=float(ac.max_line.max()),maximum_transformer_percent=float(ac.max_trafo.max()),newton_voltage_error=float(ac.voltage_error.max()),newton_line_percentage_point_error=float(ac.line_loading_error.max()),newton_transformer_percentage_point_error=float(ac.trafo_loading_error.max()))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame(summary['ou_comparison']).to_csv(OUT/'ou_comparison.csv',index=False)
    pd.DataFrame(summary['asset_metadata']).to_csv(OUT/'asset_metadata.csv',index=False)
    print('SUMMARY '+json.dumps(summary),flush=True)

def binding_sessions(df):
    """Critical zero-TES EV session and joint inventory, not extra trials."""
    out=[]
    avail,sid,_=contract()
    for key in KEYS:
        choices=df[(df.network==key)&(df.asset=='OU')&(df.installed_modules==0)&(df['mode']=='managed')&(df.status=='optimal')]
        if len(choices)!=2:continue
        row=choices.sort_values('count').iloc[0];year=int(row.year);count=float(row['count'])
        h,c,b,_,_=model_inputs(key,year,0)
        cap=np.load(ROOT/'catalog'/f'capacity_{key}_OU.npz')['cap']*1000
        margin=cap-b-h/c
        slack=np.bincount(sid,weights=np.minimum(np.maximum(margin,0),count*3.7)*avail,minlength=365)-12*count
        days=np.flatnonzero(slack<1e-6)
        dates=pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='h',inclusive='left')
        dates=dates[~((dates.month==2)&(dates.day==29))]
        jointrow=df[(df.network==key)&(df.asset=='OU')&(df.installed_modules==5)&(df['mode']=='managed')&(df.year==year)].iloc[0]
        z=np.load(OUT/'witnesses'/jointrow.witness);thermal=[]
        for day in days:
            window=(int(day)*24+np.arange(18,31))%8760
            thermal.append(dict(session_day_zero_based=int(day),arrival_model_time=dates[int(day)*24+18].isoformat(),soc_before_kwh=float(z['soc_kwh'][(int(day)*24+17)%8760]),soc_at_departure_kwh=float(z['soc_kwh'][(int(day)*24+30)%8760]),heat_during_window_kwh=float(z['heat_kwh'][window].sum()),net_discharge_in_window_kwh=float((z['discharge_kw'][window]-z['charge_kw'][window]).sum()),thermal_electricity_freed_kwh=float((h[window]/c[window]+b[window]-z['hp_kw'][window]-z['fixed_kw'][window]).sum())))
        out.append(dict(network=key,limiting_year=year,base_managed_count=count,critical_session_days_zero_based=days.tolist(),minimum_slack_kwh=float(slack.min()),joint_inventory_at_baseline_critical_sessions=thermal))
    (OUT/'binding_sessions.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out

def ac_check():
    if not (LEGACY_ROOT/'catalog_options.py').exists():
        raise RuntimeError('--ac requires the original project network environment; portable core LP recomputation does not.')
    sys.path.insert(0,str(LEGACY_ROOT))
    import catalog_options as co
    df=pd.read_csv(OUT/'main.csv');rows=[]
    for key in KEYS:
        choices=df[(df.network==key)&(df.asset=='OU')&(df.installed_modules==5)&(df['mode']=='managed')&(df.status=='optimal')]
        if len(choices)!=2:continue
        row=choices.sort_values('count').iloc[0];z=np.load(OUT/'witnesses'/row.witness)
        base=co.a.simbench.get_simbench_net(co.a.m.NETWORKS[key][0]);lp,lq,sg,_=co.a.m.simbench_profiles(base);bus=co.a.m.farthest_bus(base)
        power=(z['hp_kw']+z['ev_kw']+z['fixed_kw'])/1000
        stats,_=co.a.annual(base,lp,lq,sg,np.repeat(power,4),bus,factor=1.)
        result=dict(model_version=VERSION,network=key,asset='OU',modules=5,mode='managed',year=int(row.year),equivalent_evs=float(row['count']),**stats)
        assert stats['violations']==0,result
        rows.append(result);pd.DataFrame(rows).to_csv(OUT/'ac_validation.csv',index=False)
        print('AC '+json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--main',action='store_true');p.add_argument('--sensitivity',action='store_true');p.add_argument('--ac',action='store_true');p.add_argument('--modules',default='0,2,5');p.add_argument('--keys',default=','.join(KEYS));args=p.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'witnesses').mkdir(exist_ok=True)
    (OUT/'README.txt').write_text(__doc__,encoding='utf-8')
    if args.pilot:
        matrix=[dict(key='rural1',asset='OU',year=2024,modules=m,mode=mode,arrival=18,departure=7) for m in [0,2,5] for mode in ['immediate','managed']]
        # Keys are deliberately kept consistent with result column names below.
        for kw in matrix:
            row=run_case(**kw);print('PILOT '+json.dumps(row),flush=True)
    if args.main:
        matrix=[dict(key=k,asset=a,year=y,modules=m,mode=mode,arrival=18,departure=7) for k in args.keys.split(',') for a in ASSETS for y in [2024,2025] for m in map(int,args.modules.split(',')) for mode in ['immediate','managed']]
        # Resume identity uses result-style field names.
        f=OUT/'main.csv';rows=pd.read_csv(f).to_dict('records') if f.exists() else []
        rows=[r for r in rows if r.get('model_version')==VERSION]
        def sig(r):return (r['network'],r['asset'],int(r['year']),int(r['installed_modules']),r['mode'])
        done={sig(r) for r in rows}
        for kw in matrix:
            sk=(kw['key'],kw['asset'],kw['year'],kw['modules'],kw['mode'])
            if sk in done:continue
            row=run_case(**kw);rows.append(row);done.add(sk);pd.DataFrame(rows).to_csv(f,index=False)
            print(json.dumps({k:row.get(k) for k in ['network','asset','year','installed_modules','mode','status','count','seconds']}),flush=True)
        aggregate()
    if args.sensitivity:
        f=OUT/'sensitivity.csv';rows=pd.read_csv(f).to_dict('records') if f.exists() else []
        rows=[r for r in rows if r.get('model_version')==VERSION]
        def sig(r):return (r['network'],int(r['year']),int(r['installed_modules']),r['mode'],float(r['energy_kwh']),int(r['arrival']),int(r['departure']))
        done={sig(r) for r in rows}
        for k in args.keys.split(','):
            for y in [2024,2025]:
                for m in [0,5]:
                    for mode in ['immediate','managed']:
                        for energy,arrival,departure in [(8.,18,7),(20.,18,7),(12.,16,8),(12.,20,6)]:
                            sk=(k,y,m,mode,energy,arrival,departure)
                            if sk in done:continue
                            row=run_case(k,'OU',y,m,mode,energy,arrival,departure,'sensitivity');rows.append(row);done.add(sk);pd.DataFrame(rows).to_csv(f,index=False)
                            print('SENS '+json.dumps({k:row.get(k) for k in ['network','year','installed_modules','mode','energy_kwh','arrival','departure','status','count','seconds']}),flush=True)
    if args.ac:ac_check()
    if args.main or args.sensitivity or args.ac:write_summary()

if __name__=='__main__':main()
