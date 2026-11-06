"""A failed reconstruction removes its reference event (and the backward event)
from the catalogue, unless the executed event uses one of them; detection and
the next step must not fail on a reference event that was removed.

Si_sw_tet, step 401: reference event 8 failed to reconstruct for atom 2753,
reference event 8 was executed for atom 2788, then removed from the catalogue
(with its backward event 9), and basin detection of the executed event raised.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc.basins.detection import DetectorThreshold
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.kmc import KMC


def reference_table():
    # 8: A -> B, 9: B -> A (backward of 8), 20: an unrelated event and 21 its backward
    table = pd.DataFrame(
        {
            "idx_ref": [8, 9, 20, 21],
            "idx_backward": [9, 8, 21, 20],
            "event_id": ["A", "B", "C", "D"],
            "id_final": ["B", "A", "D", "C"],
            "energy_barrier": [0.019, 0.020, 1.0, 1.2],
        }
    )
    return SimpleNamespace(
        table=table,
        with_backward_events=lambda refs: ReferenceEventTable.with_backward_events(
            SimpleNamespace(table=table), refs
        ),
    )


def active_table(refs):
    return SimpleNamespace(table=pd.DataFrame({"num_reference_event": refs}))


def keep(err_reference, err_ae, executed_ref):
    kmc = SimpleNamespace(reference_table=reference_table(), loggers=Mock())
    kept = KMC._keep_executed_reference(
        kmc, err_reference, err_ae, active_table([executed_ref]), 0
    )
    return kept, [c.args[1] for c in kmc.loggers.info.call_args_list]


def test_executed_reference_event_is_kept():
    (refs, aes), logged = keep([8], ["A"], executed_ref=8)

    assert refs == [] and aes == []
    assert "Keeping reference event 8" in logged[0]


def test_reference_event_whose_backward_was_executed_is_kept():
    (refs, _), _ = keep([9], ["B"], executed_ref=8)

    assert refs == []


def test_other_reference_events_are_removed():
    (refs, aes), logged = keep([8, 20], ["A", "C"], executed_ref=8)

    assert refs == [20] and aes == ["C"]
    assert len(logged) == 1


def test_with_backward_events():
    table = reference_table()
    assert table.with_backward_events([8]) == {8, 9}
    assert table.with_backward_events([20, 8]) == {8, 9, 20, 21}


def test_active_events_of_removed_references_are_dropped():
    table = ActiveEventTable.__new__(ActiveEventTable)
    table.table = pd.DataFrame({"num_reference_event": [8, 20, 9, 20]})
    references = SimpleNamespace(table=pd.DataFrame({"idx_ref": [20, 21]}))

    table.remove_missing_references(references)

    assert table.table["num_reference_event"].tolist() == [20, 20]
    assert table.table.index.tolist() == [0, 1]


def selected(ref, dE=0.019):
    return pd.Series({"energy_barrier": dE, "num_reference_event": ref})


def test_detection_without_reference_event_is_not_a_basin():
    table = reference_table().table
    table = table[~table["idx_ref"].isin([8, 9])]

    assert not DetectorThreshold().detect(selected(8), table, 0.38, True)


def test_detection_without_backward_event_is_not_a_basin():
    table = reference_table().table
    table = table[table["idx_ref"] != 9]

    assert not DetectorThreshold().detect(selected(8), table, 0.38, True)


def test_detection_with_both_events():
    assert DetectorThreshold().detect(selected(8), reference_table().table, 0.38, True)
    assert not DetectorThreshold().detect(
        selected(20, 1.0), reference_table().table, 0.38, True
    )
