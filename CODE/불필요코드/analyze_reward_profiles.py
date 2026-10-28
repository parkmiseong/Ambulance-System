# -*- coding: utf-8 -*-
"""SUMO 없이 보상 민감도 비교. 연구 가정값이며 임상 근거가 아니다."""
import csv
import os
from pathlib import Path

from reward import HOSPITAL_TYPES, SUITABILITY, REWARD_PROFILES, reward_components

def main():
    out = Path("results") / "reward_profile_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, weights in REWARD_PROFILES.items():
        for ktas in range(1, 6):
            for hospital_type in HOSPITAL_TYPES:
                h = {"type": hospital_type}
                c = reward_components(180, ktas, h, occupancy_at_selection=.4,
                                      uncertainty=0, weights=weights)
                rows.append({
                    "profile": name, "pre_ktas": ktas,
                    "hospital_type": hospital_type,
                    "suitability": SUITABILITY[ktas][HOSPITAL_TYPES.index(hospital_type)],
                    **c, "total_reward": sum(c.values())
                })
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[보상 민감도] {out} ({len(rows)}개 조합)")
    for name in REWARD_PROFILES:
        example = [r for r in rows if r["profile"] == name and r["pre_ktas"] == 5]
        print(f"  {name} / Pre-KTAS 5: " +
              ", ".join(f"{r['hospital_type']}={r['total_reward']:.1f}" for r in example))

if __name__ == "__main__":
    main()
