# Utils/Scheduler.py
class EpsilonGreedy:
    def __init__(self, start=1.0, end=0.1, decay=0.995, min_eps=0.01, warmup_steps=0):
        self.start, self.end, self.decay = start, end, decay
        self.min_eps, self.warmup = min_eps, warmup_steps
    def value(self, step: int) -> float:
        if step < self.warmup: return self.start
        eps = self.start * (self.decay ** (step - self.warmup))
        return max(self.end, max(self.min_eps, eps))
