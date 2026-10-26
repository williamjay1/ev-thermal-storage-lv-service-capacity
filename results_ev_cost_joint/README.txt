Exploratory joint HP/EV/network-loss optimisation at fixed EV service.

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
