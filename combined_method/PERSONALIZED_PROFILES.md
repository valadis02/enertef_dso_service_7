# SimBench with personalized customer profiles

All SimBench results use networks whose customers have personalized load profiles (MV loads with lower variability, noise σ 0.1).

## Why

SimBench gives its LV customers only 5 to 11 standard load profiles per network, scaled per customer. In 15–26% of customer pairs the active power is practically the same (correlation above 0.99). Methods that rely on power measurements cannot tell such customers apart: moving either of two identical customers to another feeder or phase changes the balance in the same way. Real smart-meter data differ per customer.

## Personalized profiles

`simbench/indiv_profiles.py` turns each customer's SimBench profile into a personalized one, keeping its mean consumption and daily and seasonal shape:

1. each day is taken from a random day of the same weekday within ±3 weeks, independently per customer;
2. a fixed time shift per customer (σ 45 min) plus a daily jitter (σ 30 min);
3. 15-minute variability: multiplicative log-normal AR(1) noise (φ 0.7, σ 0.3);
4. households: about 3 appliance events per day, 1–2.5 kW for 15–60 min, mostly morning and evening (smaller for customers using under 1 kW on average);
5. scaled back to the original mean consumption;
6. electric vehicles for 10% of households: 3.7 kW, arriving 17:00–21:00, 5–15 kWh, on 60% of days.

Reactive power keeps each load's original Q/P ratio; PV is unchanged. After this no pair of customers has a correlation above 0.99. Everything else is as in the main protocol (6 networks, 72 cases per level, one month at 15 min).

## Use case 2 (95.2%\*)

The combined method with one difference: the power-balance refinement (move a customer to another feeder when that lowers the imbalance between the feeder-head measurement and the sum of its customers, best move first, repeated) is applied to every customer, not only to those the random forest is unsure about. Results in `results/simbench/final_feeder_month_personalized.csv`.

## Use case 3 (siamese, 98.5%)

The siamese method of Dong Liu, in [`../uc3_siamese_method`](../uc3_siamese_method). Results in `../uc3_siamese_method/results/phase_siamese_personalized/`.

## Limits

- The profiles are synthetic, derived from SimBench, not field data. Real smart-meter data (e.g. Liander) are needed to confirm the figures.
- For use case 2, refinement on every customer was chosen after seeing the results; 95.2% should be confirmed on data not used for that choice.

## Reproduce

```bash
python run_all.py --stage personalized --workers 2        # use case 2*, about 15 min on 2 cores
python ../uc3_siamese_method/run.py --workers 2           # use case 3, about 2 h on 2 cores
```

Both write to their own work folders, separate from the main results.
