# Use case 3 — siamese method (Dong Liu)

Phase identification (which phase feeds each single-phase customer) with the method of D. Liu, J. S. Giraldo, P. Palensky, P. P. Vergara, *A siamese neural network model for phase identification in distribution networks*, International Journal of Electrical Power & Energy Systems 169 (2025) 110718 ([open access PDF](https://publications.tno.nl/publication/34644311/xp14vqSd/liu-2025-siamese.pdf)). Our implementation from the paper text.

**Result:** 98.5% of customers on the right phase, 72 cases on SimBench (personalized), against 84.7% documented.

## Minimum requirements

- Active power of every customer, smart meter, 1% error.
- Active power of each phase at the LV side of the transformer, 1% error (1–3% unmodelled losses in the tests).
- One month of data at 15 min (the paper reports good results from 10–20 days).

No voltage, no documentation, no impedances.

## Method

1. **Self-taught training.** Customers are shuffled into three balanced groups whose sums act as pseudo transformer phases; pairs (phase window, customer window) are labelled 1 for the customer's own group and 0 otherwise; positives duplicated, Gaussian noise added.
2. **Features.** One-day windows (96 × 15 min), normalised; Haar wavelet, two levels; the components L2, H2, H1 reconstructed.
3. **Siamese network** with shared weights: recurrent layer (input 96, hidden 500) over the three components, fully connected layer of 500 with ELU; Euclidean distance; contrastive loss with margin 1; SGD, learning rate 0.005, 100 epochs.
4. **Estimation** with the real transformer phases: distances summed over the windows, normalised; the closest phase is chosen.
5. **Kendall label modification:** low-confidence customers are moved to the phase that most increases the Kendall tau between the first differences of the measured and the identified phase profiles.
6. Of 5 independent runs, the one with the highest Kendall tau is kept (the paper averages the 5 best of 20).

Not given in the paper, our choices: vanilla RNN, training noise σ 0.05, momentum 0.9, mini-batches of 1/8 of the customers, 5 runs.

## Data

SimBench LV networks (6 feeders) with personalized customer profiles, as described in [`../combined_method/PERSONALIZED_PROFILES.md`](../combined_method/PERSONALIZED_PROFILES.md); each case draws a random phase for every customer (12 draws per feeder, 72 cases). With SimBench's standard profiles, where many customers share the same profile, a method that uses only active power cannot tell those customers apart.

## Files

| File | Content |
|---|---|
| `run.py` | reproduces the result (about 2 h on 2 cores) |
| `siamese.py` | the method, one feeder per call |
| `phase_cases.py` | the 72 phase cases (random phases, measurement noise, documentation) |
| `phase_sim.py` | three-phase LV power flow (Z0 = 4 Z1) |
| `results/phase_siamese_personalized/` | per-case results |

Uses `../combined_method/simbench/sb_sim.py` (with `INDIV=1`) for the power flow and `../variance_distance_spanning_tree` for the network loaders.

```bash
pip install -r ../combined_method/requirements.txt -r requirements.txt
python run.py --workers 2
```
