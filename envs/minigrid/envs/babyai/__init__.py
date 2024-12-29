from __future__ import annotations

from envs.minigrid.envs.babyai.goto import (
    GoTo,
    GoToDoor,
    GoToImpUnlock,
    GoToLocal,
    GoToObj,
    GoToObjDoor,
    GoToRedBall,
    GoToRedBallGrey,
    GoToRedBallNoDists,
    GoToRedBlueBall,
    GoToSeq,
)
from envs.minigrid.envs.babyai.open import (
    Open,
    OpenDoor,
    OpenDoorsOrder,
    OpenRedDoor,
    OpenTwoDoors,
)
from envs.minigrid.envs.babyai.other import (
    ActionObjDoor,
    FindObjS5,
    KeyCorridor,
    MoveTwoAcross,
    OneRoomS8,
)
from envs.minigrid.envs.babyai.pickup import (
    Pickup,
    PickupAbove,
    PickupDist,
    PickupLoc,
    UnblockPickup,
)
from envs.minigrid.envs.babyai.putnext import PutNext, PutNextLocal
from envs.minigrid.envs.babyai.synth import (
    BossLevel,
    BossLevelNoUnlock,
    MiniBossLevel,
    Synth,
    SynthLoc,
    SynthSeq,
)
from envs.minigrid.envs.babyai.unlock import (
    BlockedUnlockPickup,
    KeyInBox,
    Unlock,
    UnlockLocal,
    UnlockPickup,
    UnlockToUnlock,
)
