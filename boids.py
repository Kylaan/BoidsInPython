import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Boid(self, x, y, vx, vy, id):
    self.position = np.array([x, y], dtype=float)
    self.velocity = np.array([vx, vy], dtype=float)
    self.id = id  # 唯一标识符
    self.perception = 50  # 感知范围半径

    def seperate(self, boids):
        for boid in boids:
            if boid.id == self.id:
                continue
            dist = np.linalg.norm(self.position - boid.position)  # 计算距离

            if 0 < dist < self.perception:
                diff = (self.position - boid.position) / dist
                steer += diff
                total += 1
