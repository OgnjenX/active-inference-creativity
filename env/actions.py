"""Action definitions for the Phase 1 environment."""

DO_NOTHING = 0
CLIMB = 1
CLIMB_OBJECT_1 = 1
CLIMB_OBJECT_2 = 2

ACTION_NAMES = {
    DO_NOTHING: "do_nothing",
    CLIMB: "climb",
}

ALL_ACTIONS = (DO_NOTHING, CLIMB)

PHASE1_5_ACTION_NAMES = {
    DO_NOTHING: "do_nothing",
    CLIMB_OBJECT_1: "climb_object_1",
    CLIMB_OBJECT_2: "climb_object_2",
}

PHASE1_5_ACTIONS = (DO_NOTHING, CLIMB_OBJECT_1, CLIMB_OBJECT_2)
