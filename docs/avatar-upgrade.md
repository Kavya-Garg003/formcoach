# Upgrading the avatar: primitive skeleton → rigged Kalidokit avatar

`frontend/js/avatarScene.js` ships with a **primitive skeleton avatar**
(spheres + cylinders positioned directly from MediaPipe world landmarks).
This is deliberate: it's the guaranteed-working baseline the PRD asks for
(section 6.2/7), it needs zero downloaded assets, and it already covers
dual-avatar comparison, deviation coloring, lighting, and both camera modes.

When you're ready to build the PRD's real retargeting milestone (Weeks 6-8):

## 1. Get a rigged humanoid
- Easiest: [Mixamo](https://www.mixamo.com) → pick any character → download
  as FBX → convert to glTF with a tool like
  [FBX2glTF](https://github.com/facebookincubator/FBX2glTF) or Blender's
  glTF exporter.
- Alternative: [ReadyPlayerMe](https://readyplayer.me) → export a VRM
  avatar directly (no conversion needed).
- Drop the resulting file in `frontend/avatars/`.

## 2. Load it with Three.js's GLTFLoader
```js
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
const loader = new GLTFLoader();
const gltf = await loader.loadAsync("./avatars/your-model.glb");
scene.add(gltf.scene);
```

## 3. Add Kalidokit for landmark → bone-rotation retargeting
```html
<script type="importmap">
{ "imports": { "kalidokit": "https://cdn.jsdelivr.net/npm/kalidokit@1.1.4/dist/kalidokit.module.js" } }
</script>
```
```js
import * as Kalidokit from "kalidokit";
const rig = Kalidokit.Pose.solve(poseWorldLandmarks3D, poseLandmarks2D, { runtime: "mediapipe" });
// `rig` gives you rotations per bone (Hips, Spine, LeftUpperLeg, etc.) --
// apply them to the corresponding bones in gltf.scene's skeleton each frame.
```

## 4. Keep the same interface
Both avatar implementations should expose:
```js
scene.update(worldLandmarks, deviationScores, referenceLandmarks)
```
so swapping one for the other is a one-line change in `main.js`, per the
PRD's own "same input, same output, one-line swap" design goal (section
6.2).

## 5. Decision checkpoint (PRD: end of Week 8)
Compare tracking accuracy and visual smoothness of the primitive-skeleton
baseline vs. your custom CCD-IK or Kalidokit implementation using your own
body as the test case, and keep whichever wins. If neither custom option
is stable by then, shipping the primitive skeleton is not a failure --
it's the documented fallback.
