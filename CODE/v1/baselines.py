# baselines.py
#!/usr/bin/env python3

import numpy as np


class HospitalRouters:
    def __init__(self, hospitals):
        self.hospitals = hospitals

    def nearest_strategy(self, pat_pos):
        distances = [
            np.hypot(pat_pos[0] - h["sumo_x"], pat_pos[1] - h["sumo_y"])
            for h in self.hospitals
        ]
        return int(np.argmin(distances))

    def rule_based_strategy(self, severity):
        candidates = []
        for i, hospital in enumerate(self.hospitals):
            if hospital["occupancy"] >= 0.9:
                continue

            hospital_type = str(hospital.get("type", "일반"))
            if severity >= 3 and ("권역" in hospital_type or "지역" in hospital_type):
                candidates.append(i)
            elif severity < 3 and "일반" in hospital_type:
                candidates.append(i)

        if candidates:
            return int(min(candidates, key=lambda i: self.hospitals[i]["occupancy"]))

        return int(np.argmin([h["occupancy"] for h in self.hospitals]))

    def heuristic_strategy(self, pat_pos, severity):
        best_idx = 0
        min_score = float("inf")

        for i, hospital in enumerate(self.hospitals):
            dist = np.hypot(
                pat_pos[0] - hospital["sumo_x"],
                pat_pos[1] - hospital["sumo_y"],
            )
            occupancy = hospital["occupancy"]
            hospital_type = str(hospital.get("type", "일반"))

            mismatch_penalty = 1000.0 if severity >= 3 and "일반" in hospital_type else 0.0
            score = (dist * 0.3) + (occupancy * 1000.0 * 0.7) + mismatch_penalty

            if score < min_score:
                min_score = score
                best_idx = i

        return int(best_idx)
