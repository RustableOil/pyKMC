# Basins

Basin settings are defined in the `[Basin]` section of the input file.
The only required parameter is the energy threshold below which a state is considered part of the basin.
You must also enable basin mode in the `[Control]` section.

Example:

```INI
[Control]
...
basin = True
...

[Basin]
energy_thr = 0.1
```

*Note: currently only one basin-handling strategy is implemented.
The `[Basin]` section is intended for future extensions when multiple algorithms (e.g., FTPA, MRT, local basins, …) will be available.*

*With `[RateConstant] style = htst`, every rate inside a basin (internal hops and exits) uses the constant prefactor `k0`; steps outside basins use the per-event `nu0`. A startup warning repeats this.*

---

## General Idea

During a KMC step, if the selected event has both forward and backward barriers lower than `energy_thr`, a `Basin` object is created.
It explores the basin, computes the exit time, and determines the exit state.
Once finished, the selected event in the KMC loop is replaced with the basin event.

FIGURE

While exploring, the main objective is to build a connectivity table containing all information required to apply the exit algorithm.
This table is stored as a `pandas.DataFrame`, managed by a `Connectivity` object (merge, remove, search for connections, ...).

FIGURE

Column meaning:

* **state**: a transient state
* **state_connexion**: a state reachable from `state`
* **event_connexion**: ID of the event that takes `state → state_connexion`
* **central_atom**: atom index on which the event must be applied
* **sym**: symmetry index of the event
* **transient**: whether `state_connexion` is transient or not
* **dE_forward / k_forward**: barrier and rate from `state` to `state_connexion`
* **dE_backward / k_backward**: barrier and rate from `state_connexion` back to `state`

_Note: Pandas Dataframe were choosen for fast querying and sorting, and it can easily be converted to a graph for analysis and characterization._

The `Basin` object uses two additional components:

* **Explorer**: explores a given state by creating its connectivity table
* **Selector**: use the connectivity table to compute the exit state and exit time

*This structure is designed to support multiple future exploration/selection algorithms.*


## Algorithm

1. **Initialization**: state to explore = current state
2. **While states remain to explore:**

   1. Find the connection (event) between a known state and this state
   2. If the state is absorbing (an exit), stop: exit states are not built
   3. Apply the event (reconstruction using a generic event); if the generic event does not match (PSR), drop the transition; if the state cannot be built, mark it as absorbing → stop
   4. If the state is already known (distance-based check), stop
   5. If the state has an unknown atomic environment → mark as absorbing → stop
   6. Send the state to the Explorer:

      * detect applicable events
      * build its connectivity table
   7. Merge with the global connectivity table
3. **Refine** the transient → absorbing transitions that matter (update dE and k)
4. **Selector step**:

   * build matrix ( M ), solve ( P = e^{-Mt} P_0 )
   * use bisection to find ( t_{\text{exit}} ) and the exit state
5. Build the result and return it to the KMC loop: the expected final positions of the exit are those of its generic event, mapped onto the from state
6. Replace the initially chosen KMC event with the basin event: the KMC loop reconstructs the exit from its refined saddle

Only transient states are built during the exploration.
An exit state is only needed when its exit is selected, and building every exit from the unrefined catalogue saddle was most of the cost of a basin; it could also fail on exits that are never selected (e.g. a 2.75 eV exit with a 0.127 A deviation) and abandon the whole basin.


The basin process may fail during PSR, refinement, reconstruction, or exit-time calculation.
If a failure occurs, the basin returns an `Err`, and the originally selected KMC event is applied instead.

A selected exit that cannot be reconstructed (or whose generic event no longer maps onto its from state) does not abandon the basin: the exit is dropped, its rate is removed from the escape rate, and the exit time and exit are drawn again from the remaining exits.
The log reports each dropped exit and the share of the exit rate the dropped exits carried.
A transient state that cannot be built from its catalogue event becomes an exit the same way, instead of abandoning the basin: if it is selected, the KMC loop reconstructs it from its refined saddle.

A failed PSR match (applying a generic event to a basin state, or refining a transient → absorbing
transition) is the exception, within a bound: the transition is dropped and the basin goes on, as long
as the dropped transitions carry at most `psr_skip_rate_thr` (default 1%) of their source state's total
outgoing rate, computed with the generic rates of the exploration and accumulated over the basin.
Dropping a transition is an approximation: a missing exit lowers the escape rate (longer exit time) and
redistributes its probability over the remaining exits, a missing internal transition changes the
residence in the basin states. Beyond the threshold the basin fails as above. `psr_skip_rate_thr = 0`
restores the strict behaviour: any failed PSR match aborts the basin. The log reports the number of
skipped transitions and the largest dropped fraction.

### Negligible exits

A basin state typically has a few low-barrier exits and many high-barrier ones (e.g. a Si split
dumbbell: ~125 exits per basin, of which 5 to 7 carry 99.99% of the escape rate). Refining the saddle
of every exit (pARTn) dominates the cost of the basin once exit states are no longer built.

For each basin state, its exits are ranked by rate (generic barrier, constant prefactor `k0`). The
exits covering `exit_refine_thr` (default 0.9999) of the state's exit rate are refined. The remaining
exits are *deferred*: they keep their generic barrier and enter the selector with that rate, so the
exit time and the exit probabilities still account for them, but they are only refined if the
selector picks one. `exit_refine_thr = 1` refines every exit.
After 10 dropped exits, or when no exit is left, the basin fails with `BASIN_EXIT_FAILED` and the original event is executed.

### Choosing `energy_thr`

A transition is inside the basin when both its forward and backward barriers are below `energy_thr`, and the basin only ends at exits: transitions with a barrier above the threshold, or states with an atomic environment not yet in the catalogue.
Set `energy_thr` below the lowest barrier that should be simulated explicitly, e.g. a migration barrier.
If migration is inside the basin, every hop leads to an equivalent, known state and the basin spreads over the whole cell (a Si self-interstitial with 0.4-0.55 eV hops and `energy_thr = 0.8` built more than 20000 states in 24 hours).

`max_states` (default 1000) bounds the exploration: once the basin holds that many (transient) states, it fails with `BASIN_TOO_MANY_STATES` and the original event is executed.
The log reports the size of the basin every 100 states, and its number of states and exits when the exploration ends.

`max_radius` (in A, default 0: no limit) bounds the exploration geometrically instead.
The basin centre is the central atom of the event that triggered the basin, in the state where the basin was entered.
Each state built during the exploration is compared with that state: if an atom moved by more than `[Control] disp_thr` lies farther than `max_radius` from the centre, the basin fails with `BASIN_OUTSIDE_RADIUS` and the original event is executed.
The end-of-exploration log line reports the basin's extent (the largest such distance), which helps choose the radius: a basin confined to a defect's neighbourhood extends a few A, a migrating defect keeps growing it.
Both bounds can be used together.

