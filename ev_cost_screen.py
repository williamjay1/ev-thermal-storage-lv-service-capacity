"""Exploratory fixed-EV-service, finite-menu 20-year cost screen.

This analysis was specified AFTER the hosting results.  For every network,
N=floor(two-year OU/no-TES/managed-EV hosting)+1 is the same transparent
boundary-selection rule.  It is a deliberately informative capacity boundary,
not an independently selected validation fleet or a representative sample.

The four existing assets OU/AU/OP100/AP100 and 0/2/5 installed 1000 L modules
are compared at exactly this N, identical hourly heat and daily EV service,
in 2024 and 2025.  Capacity failures are retained.  Fixed-N annual minimum
heat-pump-electricity LPs are solved afresh; maximum-N energy is never reused.
All schedules have annual perfect information and independent periodic SOC.
Network losses are evaluated afterwards with the existing, AC-validated
degree-four hourly loss cache.  Hence this is a cost screen under a common
minimum-HP-energy dispatch convention, not a global co-optimisation of
network losses, operation and investment or a deployable controller.

All new assets are commissioned at year 0.  Demand/service stays constant
for 20 years.  Annual operating cost uses the mean of the two development
years.  There is no load growth, deferral year, or permanent-avoidance claim.
Real costs: cable EUR100000/km+5000/stage, life40/O&M0.5%; transformer
EUR22000*(rating/.63)^.7, life45/O&M0.5%; tanks EUR4000/module, life20/
O&M1%; TES controls EUR5000, life15/O&M1%, replaced in year15.  Nominal
r4%, electricity EUR200/MWh; linear terminal residual.  Common EV control
infrastructure cost is unestimated and cancels.  Adverse: grid .65, TES1.5,
r2%, electricity EUR300/MWh, comparing the saved nominal alternative
against a freshly selected grid-only alternative.  No costs are empirical
procurement quotations.  Original heat-investment results are untouched.

Run with the project Python: python ev_cost_screen.py .  Inputs default to
the original directory; PROJECT09_EV_DATA_ROOT can point to the portable
data root, additionally containing catalog/loss_<network>_<asset>.npz.
"""
from __future__ import annotations
import json, math, pathlib, time
import numpy as np
import pandas as pd
import ev_experiment as ev

OUT=pathlib.Path(__file__).resolve().parent/'results_ev_cost'
VERSION='fixed_ev_cost_v1'
YEARS=20
LOSS_TOLERANCE_MW=1e-6
_ROOT_CACHE={}

def asset_pv(cost,life,om,r=.04,horizon=20):
    """Initial year-0 capital; replacements; end-year O&M; straight residual."""
    purchase_years=list(range(0,horizon,life))
    capital=sum(cost/(1+r)**y for y in purchase_years)
    maintenance=sum(cost*om/(1+r)**(y+1) for y in range(horizon))
    last=purchase_years[-1]
    residual=cost*max(0,1-(horizon-last)/life)/(1+r)**horizon
    return capital+maintenance-residual

def fleet_selection():
    z=pd.read_csv(ev.OUT/'hosting_two_year.csv')
    rows=[]
    for key in ev.KEYS:
        row=z[(z.network==key)&(z.asset=='OU')&(z.installed_modules==0)&(z['mode']=='managed')].iloc[0]
        assert bool(row.two_year_feasible)
        rows.append(dict(network=key,reference_equivalent_evs=float(row.equivalent_evs),reference_feasible_equivalent_evs=float(row.feasible_equivalent_evs),fixed_evs=int(math.floor(row.feasible_equivalent_evs))+1,selection='Exploratory: first integer above the observed OU/no-TES managed-EV two-year bound'))
    return rows

def case(key,asset,year,modules,count):
    start=time.perf_counter()
    row=dict(model_version=VERSION,network=key,asset=asset,year=year,installed_modules=modules,fixed_evs=count)
    h,c,b,pmax,base=ev.model_inputs(key,year,modules)
    cap=np.load(ev.ROOT/'catalog'/f'capacity_{key}_{asset}.npz')['cap']*1000
    previous=pd.read_csv(ev.OUT/'main.csv')
    previous=previous[(previous.network==key)&(previous.asset==asset)&(previous.year==year)&(previous.installed_modules==modules)&(previous['mode']=='managed')].iloc[0]
    if previous.status!='optimal' or count>float(previous['count'])+1e-7:
        return dict(row,status='infeasible_capacity_bound',hosting_upper_bound=float(previous['count']),seconds=time.perf_counter()-start)
    # With 0 modules, ask the generic LP for zero E/Q using modules=1 while
    # retaining model_inputs(...,0), i.e. no hot-spare maintenance electricity.
    o=ev.with_storage(h,c,b,cap,pmax,modules if modules else 1,'managed',12.,3.7,18,7,fixed_count=count)
    row.update(status=o['status'],seconds=time.perf_counter()-start)
    if o['status']!='optimal':
        row['message']=o.get('message','');return row
    row.update(ev.validate(o,h,c,b,cap,pmax,modules,'managed',12.,3.7,18,7))
    assert abs(o['witness_count']-count)<1e-9
    power_kw=o['hp']+o['ev']+b
    fraction=power_kw/cap
    coef=np.load(ev.ROOT/'catalog'/f'loss_{key}_{asset}.npz')['coefficients']
    background=np.load(ev.ROOT/'catalog'/f'loss_{key}_OU.npz')['coefficients'][0]
    loss=np.polynomial.polynomial.polyval(fraction,coef,tensor=False)
    assert np.isfinite(loss).all() and fraction.max()<1.000001 and loss.min()>-1e-7
    h0,c0,b0,_,_=ev.model_inputs(key,year,0)
    common_direct_mwh=float((h0/c0+b0).sum()/1000+365*12*count/1000)
    total_input_mwh=float(power_kw.sum()/1000)
    total_loss_mwh=float(loss.sum())
    background_loss_mwh=float(background.sum())
    extra_input_mwh=total_input_mwh-common_direct_mwh
    extra_network_loss_mwh=total_loss_mwh-background_loss_mwh
    row.update(base_scale=base,hp_ev_fixed_mwh=total_input_mwh,common_direct_heat_ev_mwh=common_direct_mwh,extra_input_mwh=extra_input_mwh,network_loss_mwh=total_loss_mwh,common_background_loss_mwh=background_loss_mwh,extra_network_loss_mwh=extra_network_loss_mwh,incremental_operating_mwh=extra_input_mwh+extra_network_loss_mwh,full_energy_mwh=total_input_mwh+total_loss_mwh,maximum_connection_fraction=float(fraction.max()))
    name=f'{key}_{asset}_{year}_{modules}_N{count}.npz'
    np.savez_compressed(OUT/'witnesses'/name,hp_kw=o['hp'],ev_kw=o['ev'],fixed_kw=b,soc_kwh=o['soc'],charge_kw=o['charge'],discharge_kw=o['discharge'],heat_kwh=h,cop=c,cap_kw=cap,loss_mw=loss,count=count)
    row['witness']=name
    return row

def costs(asset,modules,key,energy_mwh,grid_factor=1.,storage_factor=1.,price=200.,discount=.04):
    z=np.load(ev.ROOT/'catalog'/f'capacity_{key}_{asset}.npz')
    trafo=float(z['trafo']);length=float(z['length'])
    transformer=grid_factor*22000*(trafo/.63)**.7 if asset[0]!='O' else 0.
    corridor=grid_factor*(100000*length+5000) if asset[1:]!='U' else 0.
    grid=asset_pv(transformer,45,.005,discount)+asset_pv(corridor,40,.005,discount)
    tank=asset_pv(storage_factor*modules*4000,20,.01,discount) if modules else 0.
    control=asset_pv(storage_factor*5000,15,.01,discount) if modules else 0.
    operating=price*energy_mwh*sum(1/(1+discount)**(y+1) for y in range(YEARS))
    return dict(grid_pv_eur=grid,tanks_pv_eur=tank,tes_controls_pv_eur=control,operating_pv_eur=operating,npc_eur=grid+tank+control+operating,transformer_capex_eur=transformer,corridor_capex_eur=corridor)

def polynomial_minimum(key,asset,lower,upper):
    """Exact quartic surrogate minimum: endpoints and all real cubic roots."""
    cachekey=(key,asset)
    coef=np.load(ev.ROOT/'catalog'/f'loss_{key}_{asset}.npz')['coefficients']
    if cachekey not in _ROOT_CACHE:
        roots=np.full((3,ev.N),np.nan)
        for t in range(ev.N):
            rr=np.roots((np.arange(1,5)*coef[1:,t])[::-1])
            real=rr.real[np.abs(rr.imag)<1e-8]
            roots[:len(real),t]=real
        _ROOT_CACHE[cachekey]=roots
    roots=_ROOT_CACHE[cachekey]
    result=np.minimum(np.polynomial.polynomial.polyval(lower,coef,tensor=False),np.polynomial.polynomial.polyval(upper,coef,tensor=False))
    for candidate in roots:
        valid=np.isfinite(candidate)&(candidate>=lower)&(candidate<=upper)
        x=np.where(valid,candidate,lower)
        value=np.polynomial.polynomial.polyval(x,coef,tensor=False)
        result=np.minimum(result,np.where(valid,value,np.inf))
    return result

def annual_bounds(row):
    """Optimistic operation lower bound and a feasible-schedule upper bound.

    Daily EV energy and inventory coupling are relaxed only for pointwise
    loss minimisation, which can lower (never raise) the achievable minimum.
    The HP objective lower bound corrects its 1e-5 cycling regulariser using
    F*=sumHP+eps*sumcycle minus eps*2Q*n_active.  Thus witness HP alone is
    never incorrectly asserted to be a rigorous lower bound.
    """
    key=row['network'];asset=row['asset'];modules=int(row['installed_modules']);year=int(row['year']);count=int(row['fixed_evs'])
    z=np.load(OUT/'witnesses'/row['witness'])
    h=z['heat_kwh'];c=z['cop'];b=z['fixed_kw'];cap=z['cap_kw']
    h0,c0,b0,pmax,_=ev.model_inputs(key,year,0)
    avail,_,_=ev.contract()
    Q=max(0,modules-1)*ev.UNIT_Q
    if modules:
        hp_lower=np.maximum(0,(h-Q)/c)
        lower=b+hp_lower
        upper=np.minimum(cap,b+np.where(h>0,pmax,0)+avail*count*3.7)
    else:
        lower=b+h/c
        upper=np.minimum(cap,lower+avail*count*3.7)
    assert (lower<=upper+1e-6).all()
    lowfrac=np.maximum(lower/cap,0);upfrac=np.minimum(upper/cap,1)
    minimum_loss=polynomial_minimum(key,asset,lowfrac,upfrac)
    n_active=int((h>0).sum());cycle_actual=float((z['charge_kw']+z['discharge_kw']).sum())
    # The solved regularised LP optimum supplies a valid lower bound on
    # unregularised HP energy after this worst-case correction.
    objective_kwh=float(z['hp_kw'].sum()+1e-5*cycle_actual)
    minimum_hp_kwh=objective_kwh-1e-5*2*Q*n_active
    common_input=float(row['common_direct_heat_ev_mwh'])
    minimum_input_mwh=(minimum_hp_kwh+b.sum()+365*12*count)/1000
    loss_allowance_mwh=LOSS_TOLERANCE_MW*ev.N
    lower_energy=minimum_input_mwh-common_input+float(minimum_loss.sum())-float(row['common_background_loss_mwh'])-loss_allowance_mwh
    upper_energy=float(row['incremental_operating_mwh'])+loss_allowance_mwh
    assert lower_energy<=upper_energy+1e-7
    return dict(bounds_version='pointwise_quartic_v1',lower_incremental_operating_mwh=lower_energy,upper_incremental_operating_mwh=upper_energy,pointwise_min_network_loss_mwh=float(minimum_loss.sum()),loss_fit_allowance_mwh=loss_allowance_mwh,hp_regularisation_correction_kwh=1e-5*2*Q*n_active,minimum_input_lower_bound_mwh=minimum_input_mwh)

def summarise(df,fleets):
    # Only derivative evaluations are added here; saved fixed-N LPs remain.
    for idx,row in df.iterrows():
        if row.status=='optimal':
            for k,v in annual_bounds(row).items():df.at[idx,k]=v
    df.to_csv(OUT/'annual_fixed_service.csv',index=False)
    groups=[]
    for label,g in df.groupby(['network','asset','installed_modules','fixed_evs']):
        key,asset,modules,count=label
        row=dict(network=key,asset=asset,installed_modules=int(modules),fixed_evs=int(count),years_evaluated=int(g.year.nunique()))
        feasible=len(g)==2 and set(g.year)=={2024,2025} and bool((g.status=='optimal').all())
        row.update(feasible=feasible,status='optimal' if feasible else '|'.join(sorted(set(g.status))))
        if feasible:
            energy=float(g.incremental_operating_mwh.mean());row['annual_incremental_operating_mwh']=energy
            row['annual_total_input_mwh']=float(g.hp_ev_fixed_mwh.mean());row['annual_total_network_loss_mwh']=float(g.network_loss_mwh.mean())
            row.update(costs(asset,int(modules),key,energy))
            row.update({f'adverse_{k}':v for k,v in costs(asset,int(modules),key,energy,.65,1.5,300.,.02).items()})
            lower=float(g.lower_incremental_operating_mwh.mean());upper=float(g.upper_incremental_operating_mwh.mean())
            row['npc_lower_bound_eur']=costs(asset,int(modules),key,lower)['npc_eur']
            row['npc_feasible_upper_bound_eur']=costs(asset,int(modules),key,upper)['npc_eur']
            row['adverse_npc_lower_bound_eur']=costs(asset,int(modules),key,lower,.65,1.5,300.,.02)['npc_eur']
            row['adverse_npc_feasible_upper_bound_eur']=costs(asset,int(modules),key,upper,.65,1.5,300.,.02)['npc_eur']
        groups.append(row)
    options=pd.DataFrame(groups);options.to_csv(OUT/'cost_options.csv',index=False)
    decisions=[]
    for key in ev.KEYS:
        feasible=options[(options.network==key)&options.feasible]
        grid=feasible[feasible.installed_modules==0].sort_values('npc_eur').iloc[0]
        tes=feasible[feasible.installed_modules>0].sort_values('npc_eur').iloc[0]
        best=feasible.sort_values('npc_eur').iloc[0]
        adverse_grid=feasible[feasible.installed_modules==0].sort_values('adverse_npc_eur').iloc[0]
        adverse_best=feasible.sort_values('adverse_npc_eur').iloc[0]
        saving=float(grid.npc_eur-best.npc_eur)
        wires=feasible[feasible.installed_modules==0];stores=feasible[feasible.installed_modules>0]
        wl=float(wires.npc_lower_bound_eur.min());wu=float(wires.npc_feasible_upper_bound_eur.min());sl=float(stores.npc_lower_bound_eur.min());su=float(stores.npc_feasible_upper_bound_eur.min())
        awl=float(wires.adverse_npc_lower_bound_eur.min());awu=float(wires.adverse_npc_feasible_upper_bound_eur.min());asl=float(stores.adverse_npc_lower_bound_eur.min());asu=float(stores.adverse_npc_feasible_upper_bound_eur.min())
        decisions.append(dict(network=key,fixed_evs=int(grid.fixed_evs),grid_only_asset=grid.asset,grid_only_npc_eur=float(grid.npc_eur),storage_only_best_asset=tes.asset,storage_only_best_modules=int(tes.installed_modules),storage_only_best_npc_eur=float(tes.npc_eur),best_asset=best.asset,best_modules=int(best.installed_modules),best_npc_eur=float(best.npc_eur),saving_eur=saving,saving_percent=100*saving/float(grid.npc_eur),changes_configuration=bool(best.installed_modules>0 and best.asset!=grid.asset),adverse_reoptimised_grid_asset=adverse_grid.asset,adverse_reoptimised_grid_npc_eur=float(adverse_grid.adverse_npc_eur),adverse_fixed_nominal_npc_eur=float(best.adverse_npc_eur),adverse_fixed_nominal_saving_eur=float(adverse_grid.adverse_npc_eur-best.adverse_npc_eur),adverse_reoptimised_best_asset=adverse_best.asset,adverse_reoptimised_best_modules=int(adverse_best.installed_modules),adverse_reoptimised_best_npc_eur=float(adverse_best.adverse_npc_eur),wire_lower_bound_eur=wl,wire_feasible_upper_bound_eur=wu,tes_lower_bound_eur=sl,tes_feasible_upper_bound_eur=su,certifies_tes_cheaper_than_all_wire=bool(su<wl),certifies_wire_cheaper_than_all_tes=bool(wu<sl),adverse_wire_lower_bound_eur=awl,adverse_wire_feasible_upper_bound_eur=awu,adverse_tes_lower_bound_eur=asl,adverse_tes_feasible_upper_bound_eur=asu,adverse_certifies_tes_cheaper_than_all_wire=bool(asu<awl),adverse_certifies_wire_cheaper_than_all_tes=bool(awu<asl)))
    pd.DataFrame(decisions).to_csv(OUT/'decisions.csv',index=False)
    summary=dict(model_version=VERSION,classification='Exploratory capacity-boundary fixed-service 20-year cost screen; not a growth/deferral analysis or independent validation.',fleet_rule=fleets,annual_cases=len(df),annual_statuses=df.status.value_counts().to_dict(),cost_options=len(options),feasible_cost_options=int(options.feasible.sum()),nominal_changes_configuration=sum(d['changes_configuration'] for d in decisions),nominal_certifies_tes_cheaper=sum(d['certifies_tes_cheaper_than_all_wire'] for d in decisions),nominal_certifies_wire_cheaper=sum(d['certifies_wire_cheaper_than_all_tes'] for d in decisions),adverse_certifies_tes_cheaper=sum(d['adverse_certifies_tes_cheaper_than_all_wire'] for d in decisions),adverse_certifies_wire_cheaper=sum(d['adverse_certifies_wire_cheaper_than_all_tes'] for d in decisions),decisions=decisions,discount_nominal=.04,discount_adverse=.02,price_nominal_eur_mwh=200,price_adverse_eur_mwh=300,grid_factor_adverse=.65,storage_factor_adverse=1.5,maximum_residuals={k:float(df[k].max()) for k in ['heat_balance_kw','inventory_balance_kwh','departure_energy_kwh','network_excess_kw','simultaneous_cycling_kw']},loss_fit_allowance_mw=LOSS_TOLERANCE_MW,bound_method='Pointwise quartic minima from endpoints and all real derivative roots, relaxed EV daily/SOC coupling; HP regularised-LP lower bound subtracts worst-case cycling penalty. Upper bounds are feasible schedules. Bounds assume the stated 1e-6 MW hourly loss-surrogate error allowance.',bounds=['Fixed fleet selected after capacity outcomes, using the same rule for all six networks.','Identical daily EV service and hourly heat. Independent annual periodic perfect-information operation.','Minimum HP electricity schedules; cached network losses evaluated afterwards, not jointly optimised in dispatch.','Only four grid configurations and three storage sizes; no global engineering least-cost claim.','Static year-0 investments and constant twenty-year service; no deferral timing and no permanent avoidance.','Indicative cost catalogue, no site quotations; common EV infrastructure cost unestimated and cancels.','Hot spare uses the corrected EV-layer eta-inclusive maintenance convention.','A schedule ranking is not an optimal-operation cost ranking; only nonoverlapping bounds support all-dispatch comparisons inside the finite menu and surrogate assumptions.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('SUMMARY '+json.dumps(summary),flush=True)

def main():
    OUT.mkdir(exist_ok=True);(OUT/'witnesses').mkdir(exist_ok=True)
    (OUT/'README.txt').write_text(__doc__,encoding='utf-8')
    fleets=fleet_selection();(OUT/'fleet_selection.json').write_text(json.dumps(fleets,indent=2),encoding='utf-8')
    f=OUT/'annual_fixed_service.csv';rows=pd.read_csv(f).to_dict('records') if f.exists() else []
    rows=[r for r in rows if r.get('model_version')==VERSION]
    done={(r['network'],r['asset'],int(r['year']),int(r['installed_modules'])) for r in rows}
    for fleet in fleets:
        for asset in ev.ASSETS:
            for modules in [0,2,5]:
                for year in [2024,2025]:
                    identity=(fleet['network'],asset,year,modules)
                    if identity in done:continue
                    row=case(fleet['network'],asset,year,modules,fleet['fixed_evs']);rows.append(row);done.add(identity)
                    pd.DataFrame(rows).to_csv(f,index=False)
                    print(json.dumps({k:row.get(k) for k in ['network','asset','year','installed_modules','fixed_evs','status','incremental_operating_mwh','seconds']}),flush=True)
    summarise(pd.DataFrame(rows),fleets)

if __name__=='__main__':main()
