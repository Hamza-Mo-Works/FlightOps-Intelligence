# Semantic KPI definitions

The notebook declares three domains and ten metrics. SUM is used for counts;
other metrics use a ratio of additive numerator/denominator components.

| ID | Metric | Numerator | Denominator | Unit |
| --- | --- | --- | --- | --- |
| M001 | Total flights | total_flights | none | flights |
| M002 | Cancellation rate | cancelled_flights | total_flights | percent |
| M003 | Network on-time arrival rate | on_time_arrivals | completed_flights | percent |
| M004 | Average arrival delay | total_arrival_delay_minutes | arrival_delay_observations | minutes |
| M005 | Airport departures | total_departures | none | flights |
| M006 | Average departure delay | total_departure_delay_minutes | departure_delay_observations | minutes |
| M007 | Average taxi-out | total_taxi_out_minutes | taxi_out_observations | minutes |
| M008 | Airport departure delay rate | delayed_departures_15 | departure_delay_observations | percent |
| M009 | Carrier cancellation rate | cancelled_flights | total_flights | percent |
| M010 | Carrier on-time arrival rate | on_time_arrivals | arrival_delay_observations | percent |

Roll up sums before division. For taxi-out groups with totals/counts 90/9 and
100/1, the combined mean is 190/10 = 19 minutes, not the unweighted average of
10 and 100. A zero denominator returns null. Percent metrics multiply by 100.

M003 and M010 deliberately use different declared denominators in this version.
A completed flight without an arrival observation affects those denominators
differently. Keep that distinction visible in cross-domain comparisons.

Airport/carrier Gold grains are `(flight_date, origin_airport)` and
`(flight_date, airline)`. Network Gold has one row per date.
