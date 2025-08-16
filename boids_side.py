import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.transforms as transforms


class Boid:
    def __init__(self, x, y, vx, vy, boid_id):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.id = boid_id
        self.max_speed = 10
        self.max_force = 0.3
        self.perception = 50  # 感知范围半径

    def update(self, boids, width, height):
        # 应用三个核心规则
        separation = self.separate(boids)
        alignment = self.align(boids)
        cohesion = self.cohere(boids)

        # 应用边界规则（只对上下边界施加转向力）
        margin = 50
        turn_factor = 0.5
        border_force = np.zeros(2)

        # 只检查上下边界（y方向）
        if self.position[1] < margin:
            border_force[1] = turn_factor  # 底部边界，向上转向
        elif self.position[1] > height - margin:
            border_force[1] = -turn_factor  # 顶部边界，向下转向

        # 组合所有影响并更新速度
        self.velocity += separation * 1.5  # 加强分离效果
        self.velocity += alignment * 1.0
        self.velocity += cohesion * 1.0
        self.velocity += border_force

        # 限制速度
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        # 更新位置
        self.position += self.velocity

        # 边界处理（环绕）
        self.position[0] = self.position[0] % width
        self.position[1] = self.position[1] % height

    def separate(self, boids):
        steer = np.zeros(2)
        total = 0
        for boid in boids:
            if boid.id == self.id:
                continue
            dist = np.linalg.norm(self.position - boid.position)
            if dist < self.perception and dist > 0:
                diff = self.position - boid.position
                diff = diff / dist  # 归一化
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
        width=200,
        height=150,
        initial_direction=45,
        direction_variance=30,
        initial_speed=10,
        speed_variance=0.5,
        dynamic_direction=False,
        direction_change_rate=0.5,
    ):
        """
        参数说明:
        - num_boids: Boids数量
        - width, height: 画布尺寸
        - initial_direction: 初始方向角度(度)，0°表示正右方，90°表示正上方
        - direction_variance: 方向随机变化范围(±度)
        - initial_speed: 初始速度大小
        - speed_variance: 速度随机变化范围
        - dynamic_direction: 是否启用动态方向变化
        - direction_change_rate: 方向变化率(度/帧)
        """
        self.width = width
        self.height = height
        self.boids = []
        self.num_boids = num_boids

        # 方向相关参数
        self.initial_direction = initial_direction
        self.global_direction = initial_direction
        self.direction_change_rate = direction_change_rate
        self.dynamic_direction = dynamic_direction

        # 创建等腰三角形路径（尖头方向朝上）
        self.triangle_path = self.create_isosceles_triangle()

        # 设置图形
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.patch.set_facecolor("#0f1c2e")  # 深蓝背景
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_facecolor("#0f1c2e")
        self.ax.set_title("Boids Flocking Simulation", fontsize=16, color="white")
        self.ax.tick_params(axis="both", colors="white")  # 设置刻度颜色为白色

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
        self.ax.text(10, height - 8, text, color="white", fontsize=10)

        # 初始化boids群体
        self.initialize_boids(direction_variance, initial_speed, speed_variance)

        # 创建散点图（使用等腰三角形）
        self.scatter = self.ax.scatter(
            [b.position[0] for b in self.boids],
            [b.position[1] for b in self.boids],
            s=50,  # 大小
            c="#5bc0be",  # 默认颜色
            edgecolor="#e8f1f2",
            alpha=0.8,
        )

        # 添加方向指示器
        self.add_direction_indicator()

        # 添加速度图例
        self.add_speed_legend()

        # 预生成旋转路径（优化性能）
        self.precomputed_paths = self.precompute_rotated_paths()

    def create_isosceles_triangle(self):
        """创建等腰三角形路径（尖头方向朝上）"""
        # 等腰三角形顶点坐标（底边宽1，高度1.5）
        vertices = [
            (0, 0.6),  # 顶点（尖头）
            (-0.5, -0.6),  # 左下点
            (0.5, -0.6),  # 右下点
            (0, 0.6),  # 回到顶点
        ]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        return Path(vertices, codes)

    def precompute_rotated_paths(self, step=10):
        """预生成旋转路径（优化性能）"""
        paths = {}
        for angle in range(0, 360, step):
            transform = transforms.Affine2D()
            transform = transform.rotate_deg(angle)
            transform = transform.scale(1)  # 缩放至合适大小
            paths[angle] = self.triangle_path.transformed(transform)
        return paths

    def initialize_boids(self, direction_variance, initial_speed, speed_variance):
        """初始化boids群体"""
        # 将角度转换为弧度
        base_angle = np.radians(self.initial_direction)
        variance_rad = np.radians(direction_variance)

        for i in range(self.num_boids):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)

            # 计算带随机扰动的方向
            angle = base_angle + np.random.uniform(-variance_rad, variance_rad)

            # 计算带随机扰动的速度大小
            speed = initial_speed + np.random.uniform(-speed_variance, speed_variance)

            # 计算速度分量
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed

            self.boids.append(Boid(x, y, vx, vy, i))

    def add_direction_indicator(self):
        """添加方向指示箭头"""
        # 计算箭头位置（屏幕右上角）
        x = self.width * 0.85
        y = self.height * 0.85

        # 计算箭头方向
        angle = np.radians(self.global_direction)
        length = 8
        dx = np.cos(angle) * length
        dy = np.sin(angle) * length

        # 创建箭头
        self.direction_arrow = plt.Arrow(
            x, y, dx, dy, width=5, color="#f25f5c", alpha=0.7
        )
        self.ax.add_patch(self.direction_arrow)

        # 添加文本
        self.direction_text = self.ax.text(
            x,
            y - 5,
            f"Direction: {self.global_direction:.1f}°",
            color="white",
            fontsize=10,
            ha="center",
        )

    def add_speed_legend(self):
        """添加速度图例"""
        # 创建颜色映射
        self.cmap = plt.cm.viridis

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=self.cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=self.ax, label="Speed")
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

    def update_global_direction(self):
        """更新全局方向"""
        if self.dynamic_direction:
            self.global_direction += self.direction_change_rate
            self.global_direction %= 360

            # 更新方向指示器
            self.direction_arrow.remove()
            angle = np.radians(self.global_direction)
            length = 40
            dx = np.cos(angle) * length
            dy = np.sin(angle) * length

            x, y = self.direction_arrow.get_xy()
            self.direction_arrow = plt.Arrow(
                x, y, dx, dy, width=15, color="#f25f5c", alpha=0.7
            )
            self.ax.add_patch(self.direction_arrow)

            # 更新文本
            self.direction_text.set_text(f"Direction: {self.global_direction:.1f}°")

    def add_global_direction_force(self):
        """添加全局方向力"""
        if not self.dynamic_direction:
            return

        base_angle = np.radians(self.global_direction)
        target_velocity = np.array([np.cos(base_angle), np.sin(base_angle)]) * 2

        for boid in self.boids:
            # 计算转向力
            steer = target_velocity - boid.velocity
            # 限制转向力大小
            if np.linalg.norm(steer) > boid.max_force:
                steer = steer / np.linalg.norm(steer) * boid.max_force

            boid.velocity += steer * 0.1  # 应用部分转向力

    def update(self, frame):
        # 更新全局方向
        self.update_global_direction()

        # 添加全局方向力
        self.add_global_direction_force()

        # 更新所有boids状态
        for boid in self.boids:
            boid.update(self.boids, self.width, self.height)

        # 准备数据
        positions = np.array([b.position for b in self.boids])
        velocities = np.array([b.velocity for b in self.boids])

        # 计算速度大小
        speeds = np.linalg.norm(velocities, axis=1)

        # 计算方向角度（弧度转角度）
        angles = np.degrees(np.arctan2(velocities[:, 1], velocities[:, 0])) - 90

        # 更新位置
        self.scatter.set_offsets(positions)

        # 更新颜色（根据速度）
        colors = self.cmap(speeds / np.max(speeds))
        self.scatter.set_color(colors)

        # 更新方向（旋转等腰三角形）
        paths = []
        for angle in angles:
            # 四舍五入到最近的10度（使用预生成路径）
            rounded_angle = round(angle / 10) * 10
            path = self.precomputed_paths.get(
                rounded_angle % 360, self.precomputed_paths[0]
            )
            paths.append(path)

        self.scatter.set_paths(paths)

        return (self.scatter,)

    def run(self, frames=500, interval=50):
        """运行动画"""
        ani = FuncAnimation(
            self.fig, self.update, frames=frames, interval=interval, blit=True
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 示例1：静态方向（朝东北45°飞行）
    sim_static = BoidsSimulation(
        num_boids=100,
        initial_direction=30,
        direction_variance=0,  # 固定方向
        initial_speed=3.0,
        speed_variance=1.0,
        dynamic_direction=False,
    )
    sim_static.run(frames=1000)
