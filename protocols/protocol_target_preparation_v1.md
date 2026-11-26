# Target updating preparation specification

15 October 2026. This amendment is made after source validation but before generating predictions on any target record. The original source specification and all source results remain unchanged. No primary method, endpoint, feature, budget or source classifier has been switched based on source-validation performance.

## Purpose and scope

Prepare the primary compact-logistic updating experiment using only the saved target-updating partition. Do not predict or evaluate the final target test. Restrict this stage to support counts, numerical fitting outcomes and saved artifacts. Training-pool fit measures cannot select strategies or change the protocol.

## Frozen choices

Use 30 outcome-blind permutations of the 2178-person updating pool with fixed seed 20261016. Budgets 100, 250, 500 and 1000 are nested prefixes of each permutation. The seed is an arbitrary reproducible identifier, not a claim about a future execution date. Save every participant assignment before evaluating updating support or fitting coefficients. Never replace a sample with too few events.

Methods are no update, intercept-only, unpenalized intercept-and-slope and full compact-logistic refitting. The source-selected C remains 0.1. Full refitting learns fresh median imputation, numeric scaling and categorical encoding from the particular updating sample, with the exact compact variables and source-selected regularization. It is not a refit using all target data and receives no hyperparameter search. The fitted classifier uses sample weights normalized to mean one. Do not train boosted target models in the minimum primary experiment.

Each fitted updating/refitting method requires ten observed positive and ten observed negative participants with positive weight. The no-update reference always remains available. Do not redraw samples or use a substitute strategy for failed fits. Report method-specific failure rates and paired eligible denominators. The primary 500-label intercept-versus-static comparison uses the prespecified pairs with successful intercept fits; report how many of the 30 draws contribute.

Normalize the updating pool's partition-adjusted MEC weights within each fit. A common simple-random budget inclusion factor cancels in normalization. Use source predicted probabilities clipped at 1e-6 only when converting to logits; retain original probabilities for the no-update reference.

## Numerical failure rules

Intercept-only fitting solves the weighted residual root with a fixed bracket from minus 40 to 40. Intercept-and-slope fitting uses analytic-gradient BFGS initialized at intercept zero and slope one, a maximum of 1000 iterations and the fixed convergence checks recorded in config/updating_v1.json. Reject nonfinite fits, maximum absolute coefficient at least 40, weighted gradient infinity norm at least 1e-6, minimum observed Hessian eigenvalue at most 1e-8 or Hessian condition at least 1e10. These are numerical guardrails, not clinical thresholds. Record negative slopes rather than changing their sign. A constant source score cannot identify a calibration slope.

Refitting treats a classifier convergence warning or exception as a failed fit. No fallback penalty or retuning is allowed. Zero-multiplier records are excluded when checking bootstrap support; a failed bootstrap fit stays failed and contributes to a reported failure fraction.

## Inference preparation and next gate

Synthetic checks recover known intercept and slope shifts, preserve identity probabilities for no update, confirm weight-scale invariance, exercise separation/support failure and verify outcome-blind nested sampling. The shared survey multiplier routine gives the same PSU multiplier to updating and evaluation records, regardless of partition. This is a component check, not a completed end-to-end target-bootstrap implementation.

Before final target evaluation, implement and verify the complete joint bootstrap, paired-failure accounting, all frozen prediction paths and the result-export contract using synthetic or source data. Retain the original 3277-person final target partition. No statement about target-test benefit is possible from the preparation stage.

The local preparation freeze hashes this specification, settings, executable code, source model and participant manifest before target updating fits. It is not external preregistration or final novelty clearance. Close methodological precedent remains documented.
