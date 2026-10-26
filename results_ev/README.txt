EV hosting extension: synthetic service contracts and perfect-information bounds.

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
