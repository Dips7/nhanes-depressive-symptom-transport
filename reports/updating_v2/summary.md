# Target updating preparation

Prepared 120 fixed updating samples from 2178 participants; four strategies yield 480 fitting records. No final target-test probabilities or metrics were computed.

| Budget | Method | Status | Draws |
|---:|---|---|---:|
| 100 | intercept | insufficient_events | 7 |
| 100 | intercept | success | 23 |
| 100 | intercept_slope | insufficient_events | 7 |
| 100 | intercept_slope | success | 23 |
| 100 | no_update | success | 30 |
| 100 | refit | insufficient_events | 7 |
| 100 | refit | success | 23 |
| 250 | intercept | success | 30 |
| 250 | intercept_slope | success | 30 |
| 250 | no_update | success | 30 |
| 250 | refit | success | 30 |
| 500 | intercept | success | 30 |
| 500 | intercept_slope | success | 30 |
| 500 | no_update | success | 30 |
| 500 | refit | success | 30 |
| 1000 | intercept | success | 30 |
| 1000 | intercept_slope | success | 30 |
| 1000 | no_update | success | 30 |
| 1000 | refit | success | 30 |

Failures are preserved without redraws. These are fitting/support results, not evidence that a method improves final-test predictions. The next step is end-to-end inference verification before the locked evaluation.