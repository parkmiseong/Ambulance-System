#!/usr/init/env python3
import numpy as np

class BaselineRouters:
    def __init__(self, hospitals=None):
        # 만약 병원 데이터가 없다면 기본 서울시 응급센터 성격에 맞춰 가상의 주요 거점 위치 활용 가능
        # 여기서는 구급차와 환자의 유클리드 거리를 기반으로 작동하거나 환자 상태를 반영합니다.
        pass

    def nearest_strategy(self, amb_pos, patient_pos):
        """단순 최단 거리 기준: 환자와 가장 가까운 구급차 혹은 목적지 선택"""
        # 환자 위치 좌표들과의 거리 계산용
        dists = [np.sqrt((amb_pos[0] - p['pos'][0])**2 + (amb_pos[1] - p['pos'][1])**2) for p in patient_pos]
        return np.argmin(dists)

    def rule_based_strategy(self, amb_pos, patient_pos, urgencies):
        """규칙 기반 알고리즘: 응급도가 높은(Urgency가 높은) 환자를 우선적으로 최단 거리 순 배차"""
        # 응급도가 0.7 이상인 환자가 있으면 최우선 선택, 없으면 최단 거리 선택
        scores = []
        for i, p in enumerate(patient_pos):
            dist = np.sqrt((amb_pos[0] - p['pos'][0])**2 + (amb_pos[1] - p['pos'][1])**2)
            urgency = urgencies[i]
            # 규칙: 응급도가 높을수록 우선순위 점수 부여 (거리가 가깝고 응급도가 높을수록 낮거나 높은 점수)
            score = dist - (urgency * 500.0) 
            scores.append(score)
        return np.argmin(scores)

    def heuristic_strategy(self, amb_pos, patient_pos, urgencies):
        """휴리스틱 알고리즘: 거리 가중치와 환자 긴급도를 복합적으로 고려한 점수제"""
        best_idx = 0
        min_score = float('inf')
        w1, w2 = 0.6, 0.4  # 거리 가중치, 응급도 가중치
        
        for i, p in enumerate(patient_pos):
            dist = np.sqrt((amb_pos[0] - p['pos'][0])**2 + (amb_pos[1] - p['pos'][1])**2)
            urgency = urgencies[i]
            # 휴리스틱 점수 계산 (거리가 멀고 응급도가 낮을수록 점수가 커짐)
            score = (dist * w1) + ((1.0 - urgency) * 1000.0 * w2)
            if score < min_score:
                min_score = score
                best_idx = i
        return best_idx