# Retrieval evaluation questions v2 (iLab Project #08)

25 natural-language questions, now covering the current 72-video dataset (`bronze.videos`), not just the original 10. Same method as v1: every question is based on an activity in the audited MEVA ground truth (`bronze.ground_truth`), so each has an objectively correct source video.

Note: `bronze.ground_truth` is still getting reloaded, so verify each `video_name` and frame still exist before using this with the actual pipeline.

| # | Question | Expected video | Ground truth basis |
|---|---|---|---|
| 1 | Did anyone make a purchase in this clip? | 2018-03-07.16-50-00.16-55-00.school.G421 | `person_purchases` (8 instances) |
| 2 | Was anyone reading a document on the bus? | 2018-03-07.11-05-00.11-10-00.bus.G331 | `person_reads_document` (9 instances) |
| 3 | Did two people embrace each other in this bus scene? | 2018-03-07.11-00-01.11-05-01.bus.G506 | `person_embraces_person` @ frame 3904 |
| 4 | Did someone close a facility door in the admin building? | 2018-03-07.11-05-00.11-10-00.admin.G326 | `person_closes_facility_door` @ frame 7860 |
| 5 | Did anyone close a facility door near the end of this school clip? | 2018-03-07.11-05-01.11-10-01.school.G419 | `person_closes_facility_door` @ frame 5925 |
| 6 | Was someone talking on the phone in this school scene? | 2018-03-07.11-00-00.11-05-00.school.G420 | `person_talks_on_phone` @ frame 2250 |
| 7 | Did a vehicle make a U-turn near the hospital entrance? | 2018-03-07.11-00-04.11-05-04.hospital.G341 | `vehicle_makes_u_turn` @ frame 810 |
| 8 | Did several vehicles turn left in this school parking lot? | 2018-03-07.11-00-07.11-05-07.school.G339 | `vehicle_turns_left` (12 instances) |
| 9 | Did someone stand up in this clip? | 2018-03-05.14-10-00.14-15-00.school.G421 | `person_stands_up` (4 instances) |
| 10 | Was anyone texting on their phone in this clip? | 2018-03-05.14-10-00.14-15-00.school.G330 | `person_texts_on_phone` (2 instances) |
| 11 | Did two people hug in this school scene? | 2018-03-07.11-10-01.11-15-01.school.G421 | `person_embraces_person` (11 instances) |
| 12 | Did someone carry a heavy object repeatedly in this clip? | 2018-03-07.16-50-00.16-55-00.bus.G506 | `person_carries_heavy_object` (16 instances) |
| 13 | Did anyone open a vehicle door at the hospital? | 2018-03-07.11-10-00.11-15-00.hospital.G436 | `person_opens_vehicle_door` @ frame 6419 |
| 14 | Did two people talk to each other in the admin building? | 2018-03-07.11-00-00.11-05-01.admin.G329 | `person_talks_to_person` @ frame 3821 |
| 15 | Did someone enter through a structure near the end of this clip? | 2018-03-07.11-00-01.11-05-01.bus.G509 | `person_enters_scene_through_structure` @ frame 4102 |
| 16 | Did anyone close a facility door in this school clip? | 2018-03-07.16-50-00.16-55-00.school.G638 | `person_closes_facility_door` (2 instances) |
| 17 | Did someone open a facility door at the hospital? | 2018-03-07.16-50-07.16-55-07.hospital.G301 | `person_opens_facility_door` @ frame 2869 |
| 18 | Was anyone talking on the phone in this clip? | 2018-03-07.11-10-00.11-15-00.school.G299 | `person_talks_on_phone` @ frame 4047 |
| 19 | Did a vehicle reverse in this bus scene? | 2018-03-07.16-50-00.16-55-00.bus.G340 | `vehicle_reverses` (3 instances) |
| 20 | Did someone make a purchase in this scene? | 2018-03-07.11-00-01.11-05-01.school.G421 | `person_purchases` (4 instances) |
| 21 | Did several vehicles turn right in this clip? | 2018-03-07.16-50-01.16-55-01.bus.G505 | `vehicle_turns_right` (7 instances) |
| 22 | Did someone enter the scene through a structure here? | 2018-03-07.16-50-01.16-55-01.school.G423 | `person_enters_scene_through_structure` (3 instances) |
| 23 | Did anyone enter the admin building in this clip? | 2018-03-07.16-50-01.16-55-01.admin.G326 | `person_enters_scene_through_structure` (3 instances) |
| 24 | Did a person read a document in this school scene? | 2018-03-07.11-10-01.11-15-01.school.G421 | `person_reads_document` (7 instances) |
| 25 | Did someone sit down and later stand up in this scene? | 2018-03-07.11-05-00.11-10-00.bus.G331 | `person_sits_down` (20) + `person_stands_up` (18) |

## Two useful test cases carried over from v1

- **G423 (2018-03-05, 13:15-13:20):** still the video with almost no model events despite real activity. Good test of retrieval when the source annotation is missing.
- **G505 (2018-03-05, 13:15-13:20):** still the camera where the model over-generates. Good stress test for retrieval precision on a noisy video.
