# -*- coding: utf-8 -*-
"""Step 15: SUMO hospital coordinate, edge and route diagnostics. No project files changed."""
import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT / 'CODE' / 'gpt'))

import sumolib
import traci
from environment import SumoMedicalEnvironment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--edges-per-hospital', type=int, default=8)
    p.add_argument('--radius', type=float, default=5000)
    p.add_argument('--seed', type=int, default=20261007)
    args = p.parse_args()
    if args.samples < 1 or args.edges_per_hospital < 1:
        p.error('samples and edges-per-hospital must be positive')
    env = SumoMedicalEnvironment()
    binary = sumolib.checkBinary('sumo')
    cmd = [binary, '-c', str(env.cfg_file), '--no-warnings', 'true', '--no-step-log', 'true',
           '--duration-log.disable', 'true', '--seed', str(args.seed), '--start']
    rows = []
    traci.start(cmd)
    try:
        try:
            traci.vehicletype.copy('DEFAULT_VEHTYPE', 'ambulance_custom')
            traci.vehicletype.setVehicleClass('ambulance_custom', 'emergency')
            traci.vehicletype.setShapeClass('ambulance_custom', 'emergency')
        except Exception as exc:
            raise RuntimeError('ambulance_custom vehicle type setup failed') from exc
        env.update_hospital_coordinates()
        from train_agent import build_ambulance_edges
        origins = build_ambulance_edges(env, max_check=max(args.samples, 30))
        rng = random.Random(args.seed)
        rng.shuffle(origins)
        origins = origins[:args.samples]
        bbox = env.net.getBoundary()  # xmin, ymin, xmax, ymax
        print(f'[Step15] origins={len(origins)}, network_bbox={bbox}')
        for idx, h in enumerate(env.get_hospitals()):
            x, y = h.get('sumo_x'), h.get('sumo_y')
            row = {'index': idx, 'hospital_id': h.get('id'), 'name': h.get('name'),
                   'lat': h.get('lat'), 'lon': h.get('lon'), 'sumo_x': x, 'sumo_y': y,
                   'in_network_bbox': '', 'nearest_edge': '', 'nearest_edge_distance_m': '',
                   'neighbor_edges_5km': 0, 'candidate_edges_tested': 0,
                   'successful_routes': 0, 'failed_routes': 0, 'route_exceptions': 0,
                   'successful_origins': 0, 'test_pairs': 0, 'min_travel_sec': '',
                   'example_exception': '', 'diagnosis_hint': ''}
            if x is None or y is None or not all(math.isfinite(float(v)) for v in (x, y)):
                row['diagnosis_hint'] = 'coordinate_conversion_failed'
                rows.append(row)
                continue
            x, y = float(x), float(y)
            row['in_network_bbox'] = bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
            nearby = sorted(env.net.getNeighboringEdges(x, y, args.radius), key=lambda z: z[1])
            nearby = [(edge, dist) for edge, dist in nearby if edge.getID() and not edge.getID().startswith(':')]
            row['neighbor_edges_5km'] = len(nearby)
            if nearby:
                row['nearest_edge'] = nearby[0][0].getID()
                row['nearest_edge_distance_m'] = round(float(nearby[0][1]), 2)
            targets = nearby[:args.edges_per_hospital]
            row['candidate_edges_tested'] = len(targets)
            success_origins = 0
            min_travel = None
            for origin in origins:
                found = False
                for edge, _ in targets:
                    row['test_pairs'] += 1
                    try:
                        route = traci.simulation.findRoute(origin, edge.getID(), vType='ambulance_custom', depart=-1)
                        if route is not None and len(route.edges) > 0:
                            row['successful_routes'] += 1
                            found = True
                            tt = float(getattr(route, 'travelTime', float('nan')))
                            if math.isfinite(tt):
                                min_travel = tt if min_travel is None else min(min_travel, tt)
                            break
                        row['failed_routes'] += 1
                    except Exception as exc:
                        row['route_exceptions'] += 1
                        if not row['example_exception']:
                            row['example_exception'] = str(exc)[:180]
                if found:
                    success_origins += 1
            row['successful_origins'] = success_origins
            row['min_travel_sec'] = round(min_travel, 2) if min_travel is not None else ''
            if not row['in_network_bbox']:
                row['diagnosis_hint'] = 'outside_network_bbox'
            elif not targets:
                row['diagnosis_hint'] = 'no_nearby_edges_within_radius'
            elif success_origins == 0:
                row['diagnosis_hint'] = 'no_routes_from_sampled_origins_check_direction_permissions_connectivity'
            else:
                row['diagnosis_hint'] = 'some_routes_found'
            rows.append(row)
            print(f"[병원 {idx}] {h.get('id')} bbox={row['in_network_bbox']} nearest_m={row['nearest_edge_distance_m']} nearby={len(nearby)} reachable_origins={success_origins}/{len(origins)} hint={row['diagnosis_hint']}")
    finally:
        traci.close()
    outdir = PROJECT / 'results' / 'diagnostics'
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / 'step15_hospital_edges.csv'
    with outfile.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[저장] {outfile}')
    print('[주의] sampled origins / nearby edges diagnostic only; zero routes does not prove global disconnection.')

if __name__ == '__main__':
    main()
