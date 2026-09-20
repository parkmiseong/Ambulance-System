#baselines.py
#!/usr/bin/env python3
import numpy as np

class HospitalRouters:
    def __init__(self, hospitals):
        self.hospitals = hospitals

    def nearest_strategy(self, pat_pos):
        dists = [np.sqrt((pat_pos[0] - h['sumo_x'])**2 + (pat_pos[1] - h['sumo_y'])**2) for h in self.hospitals]
        return np.argmin(dists)

    def rule_based_strategy(self, severity):
        candidates = []
        for i, h in enumerate(self.hospitals):
            if h['occupancy'] >= 0.9:
                continue
            if severity >= 3 and ('권역' in h['type'] or '지역' in h['type']):
                candidates.append(i)
            elif severity < 3 and '일반' in h['type']:
                candidates.append(i)
        
        if not candidates:
            return int(np.argmin([h['occupancy'] for h in self.hospitals]))
        return candidates[0]

    def heuristic_strategy(self, pat_pos, severity):
        best_idx = 0
        min_score = float('inf')
        
        for i, h in enumerate(self.hospitals):
            dist = np.sqrt((pat_pos[0] - h['sumo_x'])**2 + (pat_pos[1] - h['sumo_y'])**2)
            occupancy = h['occupancy']
            mismatch_penalty = 1000.0 if (severity >= 3 and '일반' in h['type']) else 0.0
            
            score = (dist * 0.3) + (occupancy * 1000.0 * 0.7) + mismatch_penalty
            if score < min_score:
                min_score = score
                best_idx = i
        return best_idx