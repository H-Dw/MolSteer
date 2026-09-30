"""Declared global reward weight, separate from the displacement budget."""
import math


class WeightedReward:
    def __init__(self, reward, weight):
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('reward_weight must be finite and positive')
        self.reward = reward
        self.weight = float(weight)
        self.spec = reward.spec
        # Expert control may normalize its scalar aggregation. Scale its final
        # declared direction explicitly, so this weight still affects injection.
        if hasattr(reward, 'control_gradient'):
            self.control_gradient = self._weighted_control

    def __getattr__(self, name):
        return getattr(self.reward, name)

    def evaluate(self, *args, **kwargs):
        value, detail = self.reward.evaluate(*args, **kwargs)
        return self.weight * value, dict(detail, reward_weight=self.weight,
                                        unweighted_reward=float(value.detach()))

    def _weighted_control(self, *args, **kwargs):
        gradient, value, detail = self.reward.control_gradient(*args, **kwargs)
        return self.weight * gradient, self.weight * value, dict(detail, reward_weight=self.weight)
