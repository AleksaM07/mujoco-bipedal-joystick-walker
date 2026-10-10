# Motion-control method review

This review compares the local task controller with the reference
implementations checked into `.tmp_mimickit` and `.tmp_deepmimic`.

## Immediate findings

1. The task PPO horizon was too short for fall avoidance. The project default
   was `discounting=0.97` at a 20 ms control step. The forward bootstrap now
   uses `0.995` so delayed falls influence the update.
2. The generated fast-training XML enables collision only for the two foot
   geoms. All other body geoms use `contype=0` and `conaffinity=0`. MimicKit's
   DeepMimic environment instead treats ground contact by any body outside its
   allowed foot-contact list as a fall. Our policy can therefore lean/fall
   through the floor until the root-height or torso-angle test fires.
3. The current BVH clip is not yet a valid tracking target. Its kinematic
   invariants pass after the velocity-scale fix, but the free-root PD tracker
   falls and inverse dynamics exceeds actuator capacity by roughly 14.5x.
4. The most stable damped retarget is not a gait: its post-IK stability blend
   leaves both feet contact-ready for the whole cycle. Moving the stability
   prior before IK restores swing, but pushes the peak actuator ratio to about
   69x and leaves the capture point outside support for most of the cycle.
5. The reference-free controller inherited a `52.9 cm` neutral stance from
   `left/right_hip_x=+/-0.18 rad`. This makes standing easy but asks for roughly
   `26 cm` of lateral COM travel before single support. A zero-abduction neutral
   pose reduces the stance to about `21.3 cm`; it needs its own long balance
   bootstrap because the wide-stance policy is not compatible with the changed
   action center.
6. Procedural foot lift is not the missing primitive by itself. A calibrated
   hip/knee keypose can produce large swing clearance, but forcing it makes the
   free-root character fall. The next curriculum must teach phase-conditioned
   COM transfer first, then introduce foot lift.
7. `action_noise_std` and `episode_bias_std` are currently configuration-only
   fields: the runtime action path does not consume them. They must not be
   credited as exploration or robustness mechanisms in experiment reports.

## DeepMimic

MimicKit DeepMimic uses randomized reference-state initialization, future
target observations, pose/velocity/root/key-body imitation terms, early fall
termination, and a physically coherent motion clip. Its PPO configuration uses
`discount=0.99`, fixed action standard deviation `0.05`, and actor/critic
learning rate `1e-4`.

The project already implements most of the reward decomposition. The missing
prerequisite is not another reward coefficient: it is a dynamically executable
retargeted walk. Training exact tracking against the current BVH asks the policy
to reproduce forces the model cannot generate.

The original `.tmp_deepmimic/data/motions/humanoid3d_walk.txt` is a clean
`wrap` locomotion clip designed for physics-based imitation. It is a stronger
retargeting candidate than a collection dominated by short BVH fragments, but
it still needs an explicit joint/morphology conversion and the same PD and
inverse-dynamics gates before PPO.

## AMP

AMP is the best conceptual fit for joystick locomotion. It scores short windows
of state transitions with a discriminator instead of requiring the policy to
match one exact phase and frame. MimicKit's task AMP configuration uses ten
observation steps, no target-frame observation, no pose termination, and mixes
task and discriminator rewards at `0.5 / 0.5`.

This removes the conflict between arbitrary joystick commands and a fixed
world-space trajectory. It does not remove the need for good demonstrations:
the discriminator still needs plausible motion expressed in the target
morphology. Porting AMP also requires a discriminator, demonstration sampler,
replay buffer, gradient penalty, and joint PPO/discriminator update loop; the
current Brax PPO wrapper does not provide these pieces.

## Other MimicKit methods

- **AWR** changes the policy optimizer for imitation. It does not repair an
  infeasible reference or the missing motion prior, so it is not the next step.
- **ASE** learns a reusable latent skill space. It is valuable for many motion
  families, but unnecessarily complex before one walk works.
- **LCP** adds an actor smoothness constraint. It can improve a working gait,
  but cannot create one from a bad objective or reference.
- **ADD** is another adversarial imitation method and has similar integration
  cost to AMP. AMP is the simpler first adversarial baseline.
- **SMP** uses a trained diffusion/score prior and optional generative state
  initialization. It is the heaviest option and pretrained priors do not match
  this custom skeleton, so it should not be the first rescue attempt.

## Recommended order

1. Finish the narrow-stance standing A/B at the same 10M budget as the original
   successful wide-stance baseline.
2. On the stable narrow policy, learn phase-conditioned lateral COM transfer at
   zero command without asking for swing clearance. Only then add foot lift and
   finally gradual forward commands. Keep survival, COM error, clearance, and
   displacement as separate gates.
3. Enable realistic fall semantics (non-foot ground contact or an equivalent
   body-height/contact test) before judging locomotion quality.
4. Retarget the original cyclic DeepMimic walk or another clean cycle and make
   it pass PD tracking plus actuator-limit audits.
5. Train a one-clip DeepMimic tracker only after step 4 passes.
6. Add AMP for joystick-conditioned style after the demonstration representation
   is trustworthy. Keep task and prior rewards separately logged and gated.
