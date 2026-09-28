# Retrieval evaluation questions (iLab Project #08)

23 natural-language questions a user might ask the system, each with the video it should ideally be answered from.

**How they were built:** every question is based on an activity confirmed in the audited MEVA ground truth (`bronze.ground_truth`), not on what our model generated. So each question has an objectively correct source video. Coverage: all 10 videos currently used for evaluation (2018-03-05, 13:15 to 13:25).

**How to read the table:** "Expected video" is the `video_name` in `bronze.videos`. The full `video_id` for each one is in the video key at the bottom. "Ground truth basis" is the MEVA activity (and start frame, when there is a single clear instance) that makes the question answerable.

## Questions

| # | Question | Expected video | Ground truth basis |
|---|---|---|---|
| 1 | Did anyone open a facility door in the school camera near the start of the clip? | 2018-03-05.13-15-01.13-20-01.school.G328 | `person_opens_facility_door` @ frame 136 |
| 2 | Was there a vehicle that turned right in this school parking lot? | 2018-03-05.13-15-01.13-20-01.school.G328 | `vehicle_turns_right` @ frames 1513, 2146 |
| 3 | Did two people have a conversation near the end of this clip? | 2018-03-05.13-15-01.13-20-01.school.G328 | `person_talks_to_person` @ frame 8459 |
| 4 | Did a vehicle drop off a person in this scene? | 2018-03-05.13-20-00.13-25-00.school.G336 | `vehicle_drops_off_person` @ frame 392 |
| 5 | Was there a vehicle that made a U-turn in this clip? | 2018-03-05.13-20-00.13-25-00.school.G336 | `vehicle_makes_u_turn` @ frame 188 |
| 6 | Did someone carry a heavy object in this scene? | 2018-03-05.13-15-00.13-20-00.bus.G340 | `person_carries_heavy_object` |
| 7 | Was an object transferred between two people? | 2018-03-05.13-15-00.13-20-00.bus.G340 | `person_transfers_object` |
| 8 | Did a vehicle reverse at any point in this clip? | 2018-03-05.13-15-00.13-20-00.bus.G340 | `vehicle_reverses` |
| 9 | Did any vehicle make a U-turn later in this video? | 2018-03-05.13-20-00.13-25-00.bus.G340 | `vehicle_makes_u_turn` @ frame 4243 |
| 10 | How many vehicles started moving in this clip? | 2018-03-05.13-20-00.13-25-00.bus.G340 | `vehicle_starts` (4 instances) |
| 11 | Did anyone carry a heavy object outside the hospital? | 2018-03-05.13-15-00.13-20-00.hospital.G341 | `person_carries_heavy_object` |
| 12 | Was there a vehicle that made a U-turn near the hospital entrance? | 2018-03-05.13-15-00.13-20-00.hospital.G341 | `vehicle_makes_u_turn` @ frame 4745 |
| 13 | Did someone open a facility door at the hospital? | 2018-03-05.13-15-00.13-20-00.hospital.G341 | `person_opens_facility_door` @ frame 3920 |
| 14 | Did a person enter the scene through a structure in this clip? | 2018-03-05.13-15-00.13-20-00.school.G423 | `person_enters_scene_through_structure` @ frame 993 |
| 15 | Did anyone exit the scene through a building entrance? | 2018-03-05.13-20-00.13-25-00.school.G423 | `person_exits_scene_through_structure` @ frames 2995, 5126 |
| 16 | Was there more than one person entering the scene in this clip? | 2018-03-05.13-20-00.13-25-00.school.G423 | `person_enters_scene_through_structure` (2 instances) |
| 17 | Did anyone open a car trunk in the school parking lot? | 2018-03-05.13-15-00.13-20-00.school.G424 | `person_opens_trunk` @ frame 2713 |
| 18 | Was there a vehicle that made a U-turn in this clip? | 2018-03-05.13-15-00.13-20-00.school.G424 | `vehicle_makes_u_turn` (2 instances) |
| 19 | Did two people talk to each other near the parked vehicles? | 2018-03-05.13-15-00.13-20-00.school.G424 | `person_talks_to_person` @ frame 3022 |
| 20 | Did any vehicle turn right in this clip? | 2018-03-05.13-15-00.13-20-00.bus.G505 | `vehicle_turns_right` (4 instances) |
| 21 | Did someone transfer an object to another person near the vehicles? | 2018-03-05.13-15-00.13-20-00.bus.G506 | `person_transfers_object` |
| 22 | Was there a vehicle that made a U-turn late in this clip? | 2018-03-05.13-15-00.13-20-00.bus.G506 | `vehicle_makes_u_turn` @ frame 8657 |
| 23 | Did anyone open and then close a trunk in this scene? | 2018-03-05.13-15-00.13-20-00.bus.G506 | `person_opens_trunk` + `person_closes_trunk` |

## Notes on two useful test cases

- **Question 14 (G423, 13:15):** the annotation pipeline generated zero events for this video (YOLO detected nothing in the full 5 minutes), even though there is real activity. Retrieval will not be able to find this from the annotations. Good test of what happens when the source data is missing.
- **Question 20 (G505):** this camera is where the model over-generates (17 model events vs 4 real MEVA activities). Good stress test for retrieval precision on a noisy video.

## Video key

| video_name | camera | scene | start | video_id |
|---|---|---|---|---|
| 2018-03-05.13-15-01.13-20-01.school.G328 | G328 | school | 13:15:01 | `0126e91e668c2fdaafb279eee37ff57215ba30bbc4b7706b6ba56d89f6c9b256` |
| 2018-03-05.13-20-00.13-25-00.school.G336 | G336 | school | 13:20:00 | `20c38d5fcf120f6be806def062eedaebffc439ebbae279642efedb7eec9ee9cd` |
| 2018-03-05.13-15-00.13-20-00.bus.G340 | G340 | bus | 13:15:00 | `4b638634d856bbec2f837538aadeef51cdf7ee0455381af39aab6703f40443a5` |
| 2018-03-05.13-20-00.13-25-00.bus.G340 | G340 | bus | 13:20:00 | `755a77cf84ccfdc158fe8d1111494a7c53639af3d873c13117e7b7ab647bde88` |
| 2018-03-05.13-15-00.13-20-00.hospital.G341 | G341 | hospital | 13:15:00 | `da404ba17e2c2b8e2442c186cd8eb48a63e0bf73dff8bcde519643fdc97c2e9b` |
| 2018-03-05.13-15-00.13-20-00.school.G423 | G423 | school | 13:15:00 | `f749d8d43789a3ce1dcfd1ece79ad7229fed1e6c653342eee389fd4ab58321df` |
| 2018-03-05.13-20-00.13-25-00.school.G423 | G423 | school | 13:20:00 | `1551011f96e286a8bd4cc74bba820fc005e46547707a9af48ba5e1dfcef3f6ae` |
| 2018-03-05.13-15-00.13-20-00.school.G424 | G424 | school | 13:15:00 | `c53bcdd01c24fbb62e3041276b6f2dac1c1675e5d5a7cb49527b92ea22c7c307` |
| 2018-03-05.13-15-00.13-20-00.bus.G505 | G505 | bus | 13:15:00 | `a4157aeed1a95a2b7289d088010034301c25e5be71269c718242d2b9dfafd6ce` |
| 2018-03-05.13-15-00.13-20-00.bus.G506 | G506 | bus | 13:15:00 | `deef5d344553945fe18f3c5b7cc3f41b6c96817713b953a8f098266cb420c2ab` |

Video IDs were pulled from `bronze.videos` at the time of writing. If the pipeline reloads videos, re-check them by `video_name`, which stays stable.
