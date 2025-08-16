import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as transforms


class Predator:
    """
    捕食者类。它以恒定速度移动，并在边界处反弹。
    """

    def __init__(self, x, y, vx, vy):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)

    def update(self, width, height):
        """更新捕食者的位置并处理边界反弹。"""
        self.position += self.velocity

        # 边界反弹逻辑
        margin = 2  # 设定一个小的边界缓冲，防止卡住
        if self.position[0] <= margin or self.position[0] >= width - margin:
            self.velocity[0] *= -1
        if self.position[1] <= margin or self.position[1] >= height - margin:
            self.velocity[1] *= -1


class Boid:
    def __init__(self, x, y, vx, vy, boid_id):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.id = boid_id
        self.max_speed = 10
        self.max_force = 0.3
        self.perception = 20  # 对同伴的感知范围半径
        self.predator_perception = 20  # 对捕食者的感知范围半径，通常更大

    def update(self, boids, predators, width, height):
        # 应用三个核心规则
        separation = self.separate(boids)
        alignment = self.align(boids)
        cohesion = self.cohere(boids)

        # 新增：应用躲避捕食者的规则
        avoidance = self.avoid(predators)

        # 组合所有影响并更新速度
        self.velocity += separation * 1.5
        self.velocity += alignment * 1.0
        self.velocity += cohesion * 1.0
        self.velocity += avoidance * 2.5

        # 限制速度
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        # 更新位置
        self.position += self.velocity

        # 修改：边界处理（反弹）
        margin = 5
        if self.position[0] <= margin or self.position[0] >= width - margin:
            self.velocity[0] *= -1
        if self.position[1] <= margin or self.position[1] >= height - margin:
            self.velocity[1] *= -1

    def avoid(self, predators):
        """计算一个用于躲避捕食者的转向力。"""
        steer = np.zeros(2)
        total = 0
        for predator in predators:
            dist = np.linalg.norm(self.position - predator.position)
            if dist < self.predator_perception:
                diff = self.position - predator.position
                diff /= dist**2
                steer += diff
                total += 1

        if total > 0:
            steer /= total
            if np.linalg.norm(steer) > 0:
                steer = (steer / np.linalg.norm(steer)) * self.max_speed
            steer -= self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = (steer / np.linalg.norm(steer)) * self.max_force
            return steer
        return np.zeros(2)

    def separate(self, boids):
        steer = np.zeros(2)
        total = 0
        for boid in boids:
            if boid.id == self.id:
                continue
            dist = np.linalg.norm(self.position - boid.position)
            if dist < self.perception and dist > 0:
                diff = self.position - boid.position
                diff = diff / dist
                steer += diff
                total += 1

        if total > 0:
            steer = steer / total
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * self.max_speed
                steer -= self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = steer / np.linalg.norm(steer) * self.max_force
        return steer

    def align(self, boids):
        avg_velocity = np.zeros(2)
        total = 0
        for boid in boids:
            if boid.id == self.id:
                continue
            dist = np.linalg.norm(self.position - boid.position)
            if dist < self.perception:
                avg_velocity += boid.velocity
                total += 1

        if total > 0:
            avg_velocity = avg_velocity / total
            if np.linalg.norm(avg_velocity) > 0:
                avg_velocity = (
                    avg_velocity / np.linalg.norm(avg_velocity) * self.max_speed
                )
            steer = avg_velocity - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = steer / np.linalg.norm(steer) * self.max_force
            return steer
        return np.zeros(2)

    def cohere(self, boids):
        center_of_mass = np.zeros(2)
        total = 0
        for boid in boids:
            if boid.id == self.id:
                continue
            dist = np.linalg.norm(self.position - boid.position)
            if dist < self.perception:
                center_of_mass += boid.position
                total += 1

        if total > 0:
            center_of_mass = center_of_mass / total
            direction = center_of_mass - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction) * self.max_speed
            steer = direction - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = steer / np.linalg.norm(steer) * self.max_force
            return steer
        return np.zeros(2)


class BoidsSimulation:
    def __init__(
        self,
        num_boids=50,
        num_predators=1,
        width=800,
        height=600,
        initial_speed=10,
        speed_variance=0.5,  # 只保留速度和速度变化的参数
    ):
        self.width = width
        self.height = height
        self.boids = []
        self.predators = []
        self.num_boids = num_boids

        # 设置图形
        self.fig, self.ax = plt.subplots(figsize=(10, 7.5))
        self.fig.patch.set_facecolor("#0f1c2e")
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_facecolor("#0f1c2e")
        self.ax.set_title("Boids Flocking Simulation", fontsize=16, color="white")
        self.ax.tick_params(axis="both", colors="white")

        # 添加边界框
        border = plt.Rectangle(
            (0, 0),
            width,
            height,
            fill=False,
            edgecolor="#3a506b",
            linewidth=2,
            linestyle="--",
        )
        self.ax.add_patch(border)

        # 添加说明文本
        text = "Boids Algorithm: Separation, Alignment, Cohesion"
        self.ax.text(10, height - 20, text, color="white", fontsize=10)

        # 初始化boids群体和捕食者
        self.initialize_boids(initial_speed, speed_variance)
        self.initialize_predators(num_predators)

        # 创建散点图 (使用简单的点标记)
        self.scatter = self.ax.scatter(
            [b.position[0] for b in self.boids],
            [b.position[1] for b in self.boids],
            s=25,
            c="#5bc0be",
            marker="o",  # 使用圆形作为Boid的标记
            alpha=0.8,
        )

        # 为捕食者创建散点图
        self.predator_scatter = self.ax.scatter(
            [p.position[0] for p in self.predators],
            [p.position[1] for p in self.predators],
            s=150,
            c="white",
            marker="X",  # 使用'X'作为捕食者的标记
        )

        # 添加速度图例
        self.add_speed_legend()

    def initialize_predators(self, num_predators):
        """初始化捕食者群体。"""
        for i in range(num_predators):
            x = np.random.uniform(self.width / 4, self.width * 3 / 4)
            y = np.random.uniform(self.height / 4, self.height * 3 / 4)
            vx = np.random.uniform(-4, 4)
            vy = np.random.uniform(-4, 4)
            self.predators.append(Predator(x, y, vx, vy))

    def initialize_boids(self, initial_speed, speed_variance):
        """初始化boids群体 - 完全随机的位置和方向"""
        for i in range(self.num_boids):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)

            # 生成随机方向（0到360度）
            angle = np.random.uniform(0, 2 * np.pi)
            # 生成随机速度大小
            speed = initial_speed + np.random.uniform(-speed_variance, speed_variance)

            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed
            self.boids.append(Boid(x, y, vx, vy, i))

    def add_speed_legend(self):
        """添加速度图例"""
        self.cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=self.cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=self.ax, label="Speed")
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

    def update(self, frame):
        # 更新捕食者
        for predator in self.predators:
            predator.update(self.width, self.height)

        # 更新所有boids状态
        for boid in self.boids:
            boid.update(self.boids, self.predators, self.width, self.height)

        # 准备Boids数据并更新图形
        positions = np.array([b.position for b in self.boids])
        velocities = np.array([b.velocity for b in self.boids])
        speeds = np.linalg.norm(velocities, axis=1)

        self.scatter.set_offsets(positions)

        max_speed = np.max(speeds) if len(speeds) > 0 and np.max(speeds) > 0 else 1
        colors = self.cmap(speeds / max_speed)
        self.scatter.set_color(colors)

        # 更新捕食者的图形
        pred_positions = np.array([p.position for p in self.predators])
        self.predator_scatter.set_offsets(pred_positions)

        return (self.scatter, self.predator_scatter)

    def run(self, frames=500, interval=5):
        """运行动画"""
        ani = FuncAnimation(
            self.fig, self.update, frames=frames, interval=interval, blit=True
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim_with_predator = BoidsSimulation(
        num_boids=200,
        num_predators=1,
        width=200,
        height=200,
        initial_speed=3.0,
        speed_variance=1.0,
    )
    sim_with_predator.run(frames=2000)
