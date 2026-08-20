# Avatar rendering: primitive skeleton vs. rigged VRM

`frontend/js/avatarScene.js` now supports **two** avatar rendering paths
behind the same interface, per the PRD's "same input, same output,
one-line swap" design goal (section 6.2):

1. **Rigged VRM + Kalidokit** (`frontend/js/riggedAvatar.js`) — this is
   the PRD's real Week 6-8 retargeting milestone, and it's wired in by
   default. `frontend/avatars/sample.vrm` (a real, valid rigged humanoid,
   MIT-licensed, taken directly from `@pixiv/three-vrm`'s own official
   example assets) loads automatically when you open the app, and
   Kalidokit drives its bone rotations from your live MediaPipe pose each
   frame.
2. **Primitive skeleton** (spheres + cylinders positioned straight from
   MediaPipe world landmarks, in `avatarScene.js` itself) — the
   guaranteed-working fallback. `AvatarScene` automatically falls back to
   this if the VRM fails to load for any reason (missing file, blocked
   network, WebGL issue). You'll see a console message either way telling
   you which path is active.

## Using your own character instead of the included sample

The included `sample.vrm` is a generic humanoid meant to prove the
pipeline works, not a finished character. To swap in your own:

**Easiest — VRoid Studio** (free): design a character and export directly
to `.vrm`. Drop the file in `frontend/avatars/` and point
`DEFAULT_VRM_URL` in `frontend/js/riggedAvatar.js` at it.

**From Mixamo/ReadyPlayerMe instead:** these export FBX or glTF, not VRM,
so they don't have the standard VRM humanoid bone-name metadata Kalidokit
and `@pixiv/three-vrm` rely on (see "Why VRM" below). You'd need to either
convert FBX→VRM with a tool like [UniVRM](https://github.com/vrm-c/UniVRM)
(Unity-based), or write your own bone-name mapping table from Kalidokit's
output names to that specific model's skeleton — more work, only worth it
if you specifically need a Mixamo/RPM character.

## Why VRM specifically

Kalidokit's `Pose.solve()` output uses VRM's *standard* humanoid bone
names directly (`Hips`, `Spine`, `LeftUpperArm`, `RightUpperLeg`, ...).
`@pixiv/three-vrm` resolves those names against *any* VRM model's actual
internal skeleton via `vrm.humanoid.getNormalizedBoneNode(name)` — so
Kalidokit's output works on literally any valid VRM file with zero
per-model configuration. That's what makes VRM the practical choice here,
rather than a bare Mixamo-rigged glTF, which would need a hand-written
bone-mapping table per downloaded character.

## Keeping the same interface

Both avatar paths are driven through `AvatarScene.update()`:
```js
scene.update(worldLandmarks, deviationScores, referenceLandmarks, riggedPose)
```
`riggedPose` (Kalidokit's solved output, computed once per frame only if
the VRM finished loading — see `main.js`) drives the VRM path;
`worldLandmarks` + `deviationScores` alone drive the primitive-skeleton
fallback. Swapping which one is "primary" is internal to `AvatarScene` —
nothing in `main.js` needs to change based on which one is active.

## Deviation coloring: one difference worth knowing

The primitive skeleton colors each joint sphere/bone individually
(true per-joint HSL green→red lerp, PRD 6.2). The VRM path tints the
*whole figure* based on the single worst active deviation score instead
(see `riggedAvatar.js`'s `tintVRM`) — VRM meshes don't have the clean
one-material-per-joint split a primitive skeleton does, so per-joint
coloring isn't a natural fit there. The live deviation banner and the
session-stats panel still carry the precise per-joint detail either way,
so no information is actually lost — just displayed differently.

## Custom CCD-IK (PRD Week 8 stretch)

If you want to attempt your own CCD (Cyclic Coordinate Descent) IK solver
per the PRD's original Week 8 milestone instead of (or in addition to)
Kalidokit, build it as a third implementation behind the same
`scene.update(...)` interface, then run the PRD's own decision checkpoint:
compare tracking accuracy and visual smoothness against the Kalidokit/VRM
path using your own body as the test case, and keep whichever wins.
