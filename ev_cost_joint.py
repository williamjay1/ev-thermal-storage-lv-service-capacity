"""Exploratory joint HP/EV/network-loss optimisation at fixed EV service.

Retains the six post-hosting boundary fleets, all costs, service, assets and
physical constraints of ev_cost_screen.py.  The earlier minimum-HP-energy
screen remains a diagnostic result.  This extension jointly minimises heat
pump electricity plus the convex cached network-loss quartic with a tangent
epigraph LP.  Seventeen tangents delimit 16 equal segments inside each
hour's physical connection interval.  The LP objective gives a lower bound;
the same feasible dispatch evaluated in the true quartic gives an upper
bound.  Both include the assumed +/-1e-6 MW/hour loss-fit allowance, and the
lower bound subtracts the worst possible positive cycling regulariser.

All four assets and three storage sizes are retained.  Begin with 16
segments; if wire-vs-TES NPC intervals overlap, refine only options whose
lower NPC is no greater than the best feasible NPC to 32 then 64 segments.
This is computational interval refinement, not fleet/cost/result selection.
No growth or deferral year is introduced.  Annual operations use independent
periodic perfect-information years, with the same EV and heat service.

python ev_cost_joint.py --pilot ; python ev_cost_joint.py --run
Portable inputs and dependencies are those of ev_cost_screen.py.
"""
from __future__ import annotations
import argparse,json,pathlib,time,os,sys
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,vstack
import ev_experiment as ev
import ev_cost_screen as cs

OUT=pathlib.Path(__file__).resolve().parent/'results_ev_cost_joint'
VERSION='joint_loss_pwl_v1'
EPS=1e-7

def curvature(coef):
    a=2*coef[2];b=2*coef[2]+6*coef[3]+12*coef[4]
    x=np.divide(-coef[3],4*coef[4],out=np.zeros_like(coef[3]),where=abs(coef[4])>1e-30)
    y=2*coef[2]+6*coef[3]*x+12*coef[4]*x*x
    low=np.minimum(a,b);high=np.maximum(a,b)
    low=np.where((coef[4]>0)&(x>0)&(x<1),np.minimum(low,y),low)
    high=np.where((coef[4]<0)&(x>0)&(x<1),np.maximum(high,y),high)
    return low,high

def convexity_audit():
    rows=[]
    for key in ev.KEYS:
        for asset in ev.ASSETS:
            coef=np.load(ev.ROOT/'catalog'/f'loss_{key}_{asset}.npz')['coefficients']
            low,high=curvature(coef)
            rows.append(dict(network=key,asset=asset,hours=len(low),minimum_second_derivative_mw=float(low.min()),maximum_second_derivative_mw=float(high.max()),nonconvex_hours=int((low<0).sum()),full_interval_16segment_annual_gap_bound_mwh=float((high/(8*16**2)).sum())))
    df=pd.DataFrame(rows);df.to_csv(OUT/'convexity_audit.csv',index=False)
    assert int(df.nonconvex_hours.sum())==0
    return dict(network_asset_caches=len(df),hourly_polynomials=int(df.hours.sum()),nonconvex_hours=int(df.nonconvex_hours.sum()),minimum_second_derivative_mw=float(df.minimum_second_derivative_mw.min()))

def joint_lp(h,c,b,cap,pmax,modules,count,coef,segments=16):
    n=len(h);t=np.arange(n);nv=6*n
    avail,sid,_=ev.contract();Q=max(0,modules-1)*ev.UNIT_Q;E=max(0,modules-1)*ev.UNIT_E
    ix=np.where(avail)[0]
    rows=np.r_[t,t,t,n+t,n+t,n+t,n+t,2*n+sid[ix]]
    cols=np.r_[t,n+t,2*n+t,3*n+t,3*n+(t-1)%n,n+t,2*n+t,4*n+ix]
    vals=np.r_[c,np.full(n,-1/ev.ALPHA),np.ones(n),np.ones(n),np.full(n,-ev.RHO),np.full(n,-ev.ETA),np.full(n,1/ev.ETA),np.ones(len(ix))]
    eq=coo_matrix((vals,(rows,cols)),shape=(2*n+365,nv)).tocsr()
    rhs=np.r_[h,np.zeros(n),np.full(365,count*12.)]
    lower=b+np.maximum(0,(h-Q)/c)
    upper=np.minimum(cap,b+np.where(h>0,pmax,0)+avail*count*3.7)
    assert (lower<=upper+1e-6).all()
    low=lower/cap;up=upper/cap
    # Fixed connection intervals need a fixed loss, not 17 duplicate cuts.
    variable=(up-low)>1e-10;idx=np.where(variable)[0];j=len(idx)
    knots=np.linspace(0,1,segments+1)[:,None]
    points=low[idx][None,:]+knots*(up-low)[idx][None,:]
    f=np.polynomial.polynomial.polyval(points,coef[:,idx],tensor=False)*1000
    derivative=np.polynomial.polynomial.polyval(points,(np.arange(1,5)[:,None]*coef[1:,idx]),tensor=False)/(cap[idx][None,:]/1000)
    # slope*(hp+EV)-loss <= -f+slope*(P_knot-fixed)
    slope=derivative.reshape(-1);hours=np.tile(idx,segments+1);cuts=len(hours);rr=n+np.arange(cuts)
    ur=np.r_[t,t,rr,rr,rr]
    uc=np.r_[t,4*n+t,hours,4*n+hours,5*n+hours]
    uv=np.r_[np.ones(2*n),slope,slope,np.full(cuts,-1.)]
    ub=coo_matrix((uv,(ur,uc)),shape=(n+cuts,nv)).tocsr()
    limits=np.r_[cap-b,(-f+derivative*(points*cap[idx][None,:]-b[idx][None,:])).reshape(-1)]
    bounds=list(zip(np.zeros(n),np.where(h>0,pmax,0)))
    bounds+=list(zip(np.zeros(n),np.where(h>0,Q,0)))*2
    bounds += [(0,E)]*n
    bounds += list(zip(np.zeros(n),avail*count*3.7))
    fixedloss=np.polynomial.polynomial.polyval(low,coef,tensor=False)*1000
    bounds += [(None,None) if variable[i] else (float(fixedloss[i]),float(fixedloss[i])) for i in range(n)]
    objective=np.zeros(nv);objective[:n]=1.;objective[n:3*n]=EPS;objective[5*n:]=1.
    res=linprog(objective,A_ub=ub,b_ub=limits,A_eq=eq,b_eq=rhs,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not res.success:return dict(status='infeasible' if res.status==2 else 'solver_failed',message=res.message,variables=nv,inequalities=n+cuts)
    x=res.x;power=x[:n]+x[4*n:5*n]+b
    loss=np.polynomial.polynomial.polyval(power/cap,coef,tensor=False)
    _,maxcurv=curvature(coef)
    theoretical_gap=float((maxcurv*(up-low)**2/(8*segments**2)).sum())
    actual_gap=float((loss-x[5*n:]/1000).sum())
    assert actual_gap>=-1e-7 and actual_gap<theoretical_gap+1e-6,(actual_gap,theoretical_gap)
    return dict(status='optimal',count=count,witness_count=count,hp=x[:n],charge=x[n:2*n],discharge=x[2*n:3*n],soc=x[3*n:4*n],ev=x[4*n:5*n],loss_mw=loss,loss_proxy_kw=x[5*n:],objective_kwh=float(res.fun),regulariser_max_kwh=EPS*2*Q*int((h>0).sum()),actual_loss_gap_mwh=actual_gap,theoretical_loss_gap_mwh=theoretical_gap,variables=nv,inequalities=n+cuts)

def case(key,asset,year,modules,count,segments):
    start=time.perf_counter();row=dict(model_version=VERSION,network=key,asset=asset,year=year,installed_modules=modules,fixed_evs=count,segments=segments)
    previous=pd.read_csv(ev.OUT/'main.csv')
    rr=previous[(previous.network==key)&(previous.asset==asset)&(previous.year==year)&(previous.installed_modules==modules)&(previous['mode']=='managed')].iloc[0]
    if rr.status!='optimal' or count>float(rr['count'])+1e-7:
        return dict(row,status='infeasible_capacity_bound',hosting_upper_bound=float(rr['count']),seconds=time.perf_counter()-start)
    h,c,b,pmax,base=ev.model_inputs(key,year,modules)
    cap=np.load(ev.ROOT/'catalog'/f'capacity_{key}_{asset}.npz')['cap']*1000
    coef=np.load(ev.ROOT/'catalog'/f'loss_{key}_{asset}.npz')['coefficients']
    cvlo,cvhi=curvature(coef);assert cvlo.min()>0
    o=joint_lp(h,c,b,cap,pmax,modules,count,coef,segments)
    row.update({k:v for k,v in o.items() if np.isscalar(v)});row['seconds']=time.perf_counter()-start
    if o['status']!='optimal':return row
    row.update(ev.validate(o,h,c,b,cap,pmax,modules,'managed',12.,3.7,18,7))
    h0,c0,b0,_,_=ev.model_inputs(key,year,0)
    common_input=float((h0/c0+b0).sum()/1000+365*12*count/1000)
    background=float(np.load(ev.ROOT/'catalog'/f'loss_{key}_OU.npz')['coefficients'][0].sum())
    actual_total=float((o['hp']+o['ev']+b).sum()/1000+o['loss_mw'].sum())
    lp_total_lower=float((o['objective_kwh']-o['regulariser_max_kwh']+b.sum()+365*12*count)/1000)
    allowance=cs.LOSS_TOLERANCE_MW*ev.N
    lower=lp_total_lower-common_input-background-allowance
    upper=actual_total-common_input-background+allowance
    assert lower<=upper+1e-7
    row.update(annual_incremental_operating_mwh=actual_total-common_input-background,lower_incremental_operating_mwh=lower,upper_incremental_operating_mwh=upper,loss_fit_allowance_mwh=allowance,annual_total_energy_mwh=actual_total,annual_network_loss_mwh=float(o['loss_mw'].sum()),minimum_curvature_mw=float(cvlo.min()),maximum_curvature_mw=float(cvhi.max()),base_scale=base)
    name=f'{key}_{asset}_{year}_{modules}_N{count}_s{segments}.npz'
    np.savez_compressed(OUT/'witnesses'/name,hp_kw=o['hp'],ev_kw=o['ev'],fixed_kw=b,soc_kwh=o['soc'],charge_kw=o['charge'],discharge_kw=o['discharge'],heat_kwh=h,cop=c,cap_kw=cap,loss_mw=o['loss_mw'],loss_proxy_kw=o['loss_proxy_kw'],count=count)
    row['witness']=name
    return row

def options_from(df):
    out=[]
    # Only the most refined result for each annual case participates in NPC.
    latest=df.sort_values('segments').drop_duplicates(['network','asset','year','installed_modules'],keep='last')
    latest.to_csv(OUT/'annual_latest.csv',index=False)
    for label,g in latest.groupby(['network','asset','installed_modules','fixed_evs']):
        key,asset,modules,count=label;row=dict(network=key,asset=asset,installed_modules=int(modules),fixed_evs=int(count),segments=int(g.segments.max()))
        feasible=len(g)==2 and set(g.year)=={2024,2025} and bool((g.status=='optimal').all())
        row.update(feasible=feasible,status='optimal' if feasible else '|'.join(sorted(set(g.status))))
        if feasible:
            en=float(g.annual_incremental_operating_mwh.mean());lo=float(g.lower_incremental_operating_mwh.mean());up=float(g.upper_incremental_operating_mwh.mean())
            row.update(annual_incremental_operating_mwh=en,lower_incremental_operating_mwh=lo,upper_incremental_operating_mwh=up)
            for prefix,gf,sf,price,r in [('',1.,1.,200.,.04),('adverse_',.65,1.5,300.,.02)]:
                row[prefix+'npc_eur']=cs.costs(asset,int(modules),key,en,gf,sf,price,r)['npc_eur']
                row[prefix+'npc_lower_eur']=cs.costs(asset,int(modules),key,lo,gf,sf,price,r)['npc_eur']
                row[prefix+'npc_upper_eur']=cs.costs(asset,int(modules),key,up,gf,sf,price,r)['npc_eur']
        out.append(row)
    options=pd.DataFrame(out);options.to_csv(OUT/'cost_options.csv',index=False)
    return options

def decisions_from(options):
    decisions=[]
    for key in ev.KEYS:
        f=options[(options.network==key)&options.feasible]
        if f.empty:continue
        row=dict(network=key,fixed_evs=int(f.fixed_evs.iloc[0]))
        for prefix in ['', 'adverse_']:
            wire=f[f.installed_modules==0];tes=f[f.installed_modules>0]
            if wire.empty or tes.empty:continue
            w=wire.sort_values(prefix+'npc_upper_eur').iloc[0];s=tes.sort_values(prefix+'npc_upper_eur').iloc[0];best=f.sort_values(prefix+'npc_upper_eur').iloc[0]
            wl=float(wire[prefix+'npc_lower_eur'].min());wu=float(w[prefix+'npc_upper_eur']);sl=float(tes[prefix+'npc_lower_eur'].min());su=float(s[prefix+'npc_upper_eur'])
            row.update({prefix+k:v for k,v in dict(wire_asset=w.asset,wire_npc_lower_eur=wl,wire_npc_upper_eur=wu,tes_asset=s.asset,tes_modules=int(s.installed_modules),tes_npc_lower_eur=sl,tes_npc_upper_eur=su,best_asset=best.asset,best_modules=int(best.installed_modules),best_npc_eur=float(best[prefix+'npc_eur']),tes_cheaper_than_all_wire=bool(su<wl),wire_cheaper_than_all_tes=bool(wu<sl),feasible_tes_saving_eur=float(w[prefix+'npc_eur']-s[prefix+'npc_eur']),guaranteed_tes_saving_eur=wl-su,guaranteed_wire_saving_eur=sl-wu).items()})
        decisions.append(row)
    result=pd.DataFrame(decisions);result.to_csv(OUT/'decisions.csv',index=False)
    return result

def write_summary(df,options,decisions,fleets):
    latest=df.sort_values('segments').drop_duplicates(['network','asset','year','installed_modules'],keep='last')
    summary=dict(model_version=VERSION,classification='Exploratory fixed-service finite-menu joint energy optimisation with convex-loss LP bounds; not a growth/deferral analysis.',fleet_selection=fleets,annual_cases=len(latest),annual_statuses=latest.status.value_counts().to_dict(),refinement_history_rows=len(df),segments_used=sorted(set(map(int,df.segments))),cost_options=len(options),feasible_options=int(options.feasible.sum()),nominal_tes_certificates=int(decisions.tes_cheaper_than_all_wire.sum()),nominal_wire_certificates=int(decisions.wire_cheaper_than_all_tes.sum()),adverse_tes_certificates=int(decisions.adverse_tes_cheaper_than_all_wire.sum()),adverse_wire_certificates=int(decisions.adverse_wire_cheaper_than_all_tes.sum()),decisions=decisions.to_dict('records'),maximum_pwl_energy_gap_mwh=float(latest.actual_loss_gap_mwh.max()),maximum_annual_energy_bound_width_mwh=float((latest.upper_incremental_operating_mwh-latest.lower_incremental_operating_mwh).max()),maximum_residuals={k:float(latest[k].max()) for k in ['heat_balance_kw','inventory_balance_kwh','departure_energy_kwh','network_excess_kw','simultaneous_cycling_kw']},compute_case_seconds=float(df.seconds.sum()),loss_fit_tolerance_mw=cs.LOSS_TOLERANCE_MW,convexity=convexity_audit(),notes=['Same post-capacity boundary fleet rule and costs as the diagnostic minimum-HP-energy screen; no fleet/cost changes.','Every year is independently periodic and uses perfect information.','All cached loss quartics are checked convex over u in [0,1]; tangent objective is a lower bound and polynomial-evaluated dispatch is feasible.','Bounds include the stated loss-surrogate error allowance and subtract the cycling regulariser worst case.','Certificates compare only the listed finite grid/storage menu under the model and loss-tolerance assumptions; they do not establish a pending utility investment or observed deferral.'])
    if (OUT/'ac_validation.csv').exists():
        ac=pd.read_csv(OUT/'ac_validation.csv')
        summary['selected_wire_ac_validation']=dict(records=len(ac),intervals=int(ac.intervals.sum()),violations=int(ac.violations.sum()),minimum_voltage_pu=float(ac.min_voltage.min()),maximum_voltage_pu=float(ac.max_voltage.max()),maximum_line_percent=float(ac.max_line.max()),maximum_transformer_percent=float(ac.max_trafo.max()),newton_voltage_error=float(ac.voltage_error.max()),newton_line_percentage_point_error=float(ac.line_loading_error.max()),newton_transformer_percentage_point_error=float(ac.trafo_loading_error.max()))
    if (OUT/'loss_validation.csv').exists():
        loss=pd.read_csv(OUT/'loss_validation.csv')
        summary['selected_wire_loss_validation']=dict(records=len(loss),maximum_hourly_error_mw=float(loss.maximum_hourly_error_mw.max()),maximum_annual_error_mwh=float(loss.absolute_annual_error_mwh.max()),all_within_assumed_allowance=bool((loss.maximum_hourly_error_mw<=cs.LOSS_TOLERANCE_MW).all()))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('SUMMARY '+json.dumps(summary),flush=True)

def run():
    OUT.mkdir(exist_ok=True);(OUT/'witnesses').mkdir(exist_ok=True);(OUT/'README.txt').write_text(__doc__,encoding='utf-8')
    fleets=cs.fleet_selection();f=OUT/'annual_history.csv';rows=pd.read_csv(f).to_dict('records') if f.exists() else []
    done={(r['network'],r['asset'],int(r['year']),int(r['installed_modules']),int(r['segments'])) for r in rows}
    def execute(key,asset,year,modules,count,segments):
        ident=(key,asset,year,modules,segments)
        if ident in done:return
        row=case(key,asset,year,modules,count,segments);rows.append(row);done.add(ident);pd.DataFrame(rows).to_csv(f,index=False)
        print(json.dumps({k:row.get(k) for k in ['network','asset','year','installed_modules','segments','status','seconds','annual_incremental_operating_mwh','actual_loss_gap_mwh']}),flush=True)
    for fleet in fleets:
        for asset in ev.ASSETS:
            for modules in [0,2,5]:
                for year in [2024,2025]:execute(fleet['network'],asset,year,modules,fleet['fixed_evs'],16)
    for nextsegments in [32,64]:
        options=options_from(pd.DataFrame(rows));decisions=decisions_from(options)
        refine=set()
        for _,decision in decisions.iterrows():
            fopts=options[(options.network==decision.network)&options.feasible]
            for prefix in ['', 'adverse_']:
                if bool(decision[prefix+'tes_cheaper_than_all_wire']) or bool(decision[prefix+'wire_cheaper_than_all_tes']):continue
                best_upper=float(fopts[prefix+'npc_upper_eur'].min())
                for _,r in fopts[fopts[prefix+'npc_lower_eur']<=best_upper].iterrows():refine.add((r.network,r.asset,int(r.installed_modules),int(r.fixed_evs)))
        if not refine:break
        for key,asset,modules,count in sorted(refine):
            for year in [2024,2025]:execute(key,asset,year,modules,count,nextsegments)
    df=pd.DataFrame(rows);options=options_from(df);decisions=decisions_from(options);write_summary(df,options,decisions,fleets)

def ac_validate_selected():
    """Annual AC for the actual selected nominal/adverse wire upper bounds."""
    if not (ev.LEGACY_ROOT/'catalog_options.py').exists():
        raise RuntimeError('--ac requires the original project network environment.')
    sys.path.insert(0,str(ev.LEGACY_ROOT));import catalog_options as co
    decisions=pd.read_csv(OUT/'decisions.csv');latest=pd.read_csv(OUT/'annual_latest.csv');rows=[]
    for key in ev.KEYS:
        decision=decisions[decisions.network==key].iloc[0]
        assets=sorted(set([decision.wire_asset,decision.adverse_wire_asset]))
        base=co.a.simbench.get_simbench_net(co.a.m.NETWORKS[key][0]);lp,lq,sg,_=co.a.m.simbench_profiles(base);bus=co.a.m.farthest_bus(base)
        for asset in assets:
            net,connection,_=co.option(base,bus,asset)
            for year in [2024,2025]:
                row=latest[(latest.network==key)&(latest.asset==asset)&(latest.installed_modules==0)&(latest.year==year)].iloc[0]
                z=np.load(OUT/'witnesses'/row.witness)
                power=(z['hp_kw']+z['ev_kw']+z['fixed_kw'])/1000
                stats,_=co.a.annual(net,lp,lq,sg,np.repeat(power,4),connection,factor=1.)
                result=dict(model_version=VERSION,network=key,asset=asset,year=year,fixed_evs=int(row.fixed_evs),modules=0,**stats)
                assert stats['violations']==0,result
                rows.append(result);pd.DataFrame(rows).to_csv(OUT/'ac_validation.csv',index=False)
                print('AC '+json.dumps(result),flush=True)
    write_summary(pd.read_csv(OUT/'annual_history.csv'),pd.read_csv(OUT/'cost_options.csv'),decisions,cs.fleet_selection())

def loss_validate_selected():
    """Check selected-wire loss fits against exact quarter-hour AC losses."""
    if not (ev.LEGACY_ROOT/'catalog_options.py').exists():raise RuntimeError('Loss validation needs original network environment.')
    sys.path.insert(0,str(ev.LEGACY_ROOT));import catalog_options as co
    decisions=pd.read_csv(OUT/'decisions.csv');latest=pd.read_csv(OUT/'annual_latest.csv');rows=[]
    for key in ev.KEYS:
        decision=decisions[decisions.network==key].iloc[0]
        assets=sorted(set([decision.wire_asset,decision.adverse_wire_asset]))
        base=co.a.simbench.get_simbench_net(co.a.m.NETWORKS[key][0]);lp,lq,sg,_=co.a.m.simbench_profiles(base);bus=co.a.m.farthest_bus(base)
        for asset in assets:
            net,connection,_=co.option(base,bus,asset)
            evaluator=co.make_loss_evaluator(net,lp,lq,sg,connection)
            for year in [2024,2025]:
                row=latest[(latest.network==key)&(latest.asset==asset)&(latest.installed_modules==0)&(latest.year==year)].iloc[0]
                z=np.load(OUT/'witnesses'/row.witness);power=(z['hp_kw']+z['ev_kw']+z['fixed_kw'])/1000
                feasible,loss=evaluator(np.repeat(power,4));assert feasible.all()
                hourly=loss.reshape(-1,4).mean(axis=1)
                error=float(abs(hourly-z['loss_mw']).max())
                result=dict(model_version=VERSION,network=key,asset=asset,year=year,maximum_hourly_error_mw=error,exact_annual_loss_mwh=float(hourly.sum()),cached_annual_loss_mwh=float(z['loss_mw'].sum()),absolute_annual_error_mwh=float(abs(hourly.sum()-z['loss_mw'].sum())),quarter_hour_intervals=len(loss))
                assert error<=cs.LOSS_TOLERANCE_MW,result
                rows.append(result);pd.DataFrame(rows).to_csv(OUT/'loss_validation.csv',index=False);print('LOSS '+json.dumps(result),flush=True)
    write_summary(pd.read_csv(OUT/'annual_history.csv'),pd.read_csv(OUT/'cost_options.csv'),decisions,cs.fleet_selection())

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--ac',action='store_true');p.add_argument('--losscheck',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'witnesses').mkdir(exist_ok=True)
    if args.pilot:
        row=case('rural1','OU',2025,2,31,16);print('PILOT '+json.dumps(row),flush=True)
        (OUT/'pilot.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    if args.run:run()
    if args.ac:ac_validate_selected()
    if args.losscheck:loss_validate_selected()

if __name__=='__main__':main()
