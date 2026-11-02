"""Research bed accounting: static background load, reservations and timed stays."""
import heapq


class HospitalCapacity:
    def __init__(self, hospitals, config):
        self.hospitals = hospitals
        self.config = config
        self.background = [float(h["occupancy"]) for h in hospitals]
        self.reservations = {}
        self.admitted = [0] * len(hospitals)
        self.releases = []
        self.serial = 0
        self.now = 0.
        self.peak_occupancy = list(self.background)
        self.refresh(0.)

    def _reserved(self, action):
        return sum(i == action for i in self.reservations.values())

    def occupancy(self, action):
        return self.background[action] + self.config.bed_fraction * (
            self.admitted[action] + self._reserved(action)
        )

    def can_reserve(self, action):
        return (self.occupancy(action) + self.config.bed_fraction
                <= self.config.occupancy_limit + 1e-9)

    def reserve(self, action, token):
        if token in self.reservations:
            raise ValueError("Duplicate bed reservation.")
        if not self.can_reserve(action):
            raise ValueError("No bed available for reservation.")
        self.reservations[token] = action
        self.refresh(self.now)

    def complete(self, token, arrived, now):
        self.refresh(now)
        action = self.reservations.pop(token)
        if arrived:
            self.admitted[action] += 1
            self.serial += 1
            heapq.heappush(self.releases, (
                now + self.config.treatment_seconds, self.serial, action,
            ))
        self.refresh(now)

    def refresh(self, now):
        if now < self.now:
            raise ValueError("Hospital clock cannot move backwards.")
        self.now = float(now)
        while self.releases and self.releases[0][0] <= now:
            _, _, action = heapq.heappop(self.releases)
            self.admitted[action] -= 1
        for i, hospital in enumerate(self.hospitals):
            occupancy = self.occupancy(i)
            self.peak_occupancy[i] = max(self.peak_occupancy[i], occupancy)
            release = min((event[0] for event in self.releases if event[2] == i), default=None)
            hospital.update(
                occupancy=occupancy,
                occupied_occupancy=self.background[i] + self.admitted[i] * self.config.bed_fraction,
                reserved_count=self._reserved(i), admitted_count=self.admitted[i],
                can_admit=self.can_reserve(i),
                next_release_seconds=(release - now if release is not None else 0.),
            )
