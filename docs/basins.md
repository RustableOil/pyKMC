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

   1. If the state is already known (distance-based check), stop
   2. If the state has an unknown atomic environment → mark as absorbing → stop
   3. If the state is absorbing, stop
   4. Find the connection (event) between a known state and this state
   5. Apply the event (reconstruction using a generic event)
   6. Send the state to the Explorer:

      * detect applicable events
      * build its connectivity table
   7. Merge with the global connectivity table
3. **Refine** the transient → absorbing transitions that matter (update dE and k)
4. **Selector step**:

   * build matrix ( M ), solve ( P = e^{-Mt} P_0 )
   * use bisection to find ( t_{\text{exit}} ) and the exit state
5. Build the result and return it to the KMC loop. If the selected exit was deferred, build and refine it first

### Negligible exits

A basin state typically has a few low-barrier exits and many high-barrier ones (e.g. a Si split
dumbbell: ~125 exits per basin, of which 5 to 7 carry 99.99% of the escape rate). Building an exit
state (a minimization) and refining its saddle (pARTn) for every exit dominates the cost of the basin.

For each basin state, its exits are ranked by rate (generic barrier, constant prefactor `k0`). The
exits covering `exit_refine_thr` (default 0.9999) of the state's exit rate are built and refined as
before. The remaining exits are *deferred*: they keep their generic barrier and enter the selector
with that rate, so the exit time and the exit probabilities still account for them, but they are only
built and refined if the selector picks one. Deferred exits are not checked for duplicate states (an
exit whose final state is already known stays a separate exit), which does not change the exit time.
`exit_refine_thr = 1` builds and refines every exit.
6. Replace the initially chosen KMC event with the basin event


The basin process may fail during PSR, refinement, reconstruction, or exit-time calculation.
If a failure occurs, the basin returns an `Err`, and the originally selected KMC event is applied instead.

A failed PSR match (applying a generic event to a basin state, or refining a transient → absorbing
transition) is the exception, within a bound: the transition is dropped and the basin goes on, as long
as the dropped transitions carry at most `psr_skip_rate_thr` (default 1%) of their source state's total
outgoing rate, computed with the generic rates of the exploration and accumulated over the basin.
Dropping a transition is an approximation: a missing exit lowers the escape rate (longer exit time) and
redistributes its probability over the remaining exits, a missing internal transition changes the
residence in the basin states. Beyond the threshold the basin fails as above. `psr_skip_rate_thr = 0`
restores the strict behaviour: any failed PSR match aborts the basin. The log reports the number of
skipped transitions and the largest dropped fraction.

