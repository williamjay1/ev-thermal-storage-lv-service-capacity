Exploratory fixed-EV-service, finite-menu 20-year cost screen.

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
