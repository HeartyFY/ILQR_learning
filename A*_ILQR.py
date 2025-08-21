import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.interpolate import CubicSpline

class AStarPlanner:
    """
    A* 路径生成
    """
    def __init__(self, grid):
        self.grid = grid
        self.height = grid.shape[0]
        self.width = grid.shape[1]
    

    class Node:
        def __init__(self, x, y, cost, parent_index, f_value=float('inf')):
            self.x = x
            self.y = y
            self.cost = cost  # g(n): 实际代价
            self.parent_index = parent_index
            self.f_value = f_value  # f(n) = g(n) + h(n): 总估计代价
        
        def __str__(self):
            return f"{self.x},{self.y},{self.cost},{self.parent_index},{self.f_value}"
        
        def __lt__(self, other):
            # 按照 f(n) 排序，这是 A* 算法的核心
            return self.f_value < other.f_value
        
    @staticmethod
    def heuristic(node, goal):
        return np.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)
    
    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(sx, sy, 0.0, -1)
        goal_node = self.Node(gx, gy, 0.0, -1)
        
        # 计算起点的 f_value
        start_node.f_value = start_node.cost + self.heuristic(start_node, goal_node)
        
        open_set = {}
        closed_set = {}
        priority_queue = []
        heapq.heappush(priority_queue, start_node)  # 现在直接存入 node
        open_set[self.calc_index(start_node)] = start_node
        
        while priority_queue:
            current_node = heapq.heappop(priority_queue)  # 直接取出 node
            current_index = self.calc_index(current_node)
            
            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                goal_node.parent_index = current_node.parent_index
                goal_node.cost = current_node.cost
                break
            
            if current_index in open_set:
                del open_set[current_index]
            closed_set[current_index] = current_node
            
            for move in self.get_motion_model():
                node = self.Node(current_node.x + move[0],
                                current_node.y + move[1],
                                current_node.cost + move[2],
                                current_index)
                
                node_index = self.calc_index(node)
                
                if not self.verify_node(node) or node_index in closed_set:
                    continue
                
                # 计算新节点的 f_value
                node.f_value = node.cost + self.heuristic(node, goal_node)
                
                if node_index not in open_set or open_set[node_index].cost > node.cost:
                    heapq.heappush(priority_queue, node)
                    open_set[node_index] = node
        
        path = self.calc_final_path(goal_node, closed_set)
        return np.array(path)
    
    def calc_index(self, node):
        return node.y * self.width + node.x
    
    def verify_node(self, node):
        if node.x < 0 or node.y < 0 or node.x >= self.width or node.y >= self.height:
            return False
        if self.grid[node.y][node.x] == 1:
            return False
        return True
    
    @staticmethod
    def get_motion_model():
        motion = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                  [-1, -1, np.sqrt(2)], [-1, 1, np.sqrt(2)],
                  [1, -1, np.sqrt(2)], [1, 1, np.sqrt(2)]]
        return motion
    
    def calc_final_path(self, goal_node, closed_set):
        path = [[goal_node.x, goal_node.y]]
        parent_index = goal_node.parent_index
        
        while parent_index != -1:
            node = closed_set[parent_index]
            path.append([node.x, node.y])
            parent_index = node.parent_index
        
        path.reverse()
        return path

def gradient_smooth(path, alpha=0.15, beta=0.3, iterations=100):
    """
    梯度下降平滑路径
    :param path: numpy数组 (N,2)
    :param alpha: 数据项权重
    :param beta: 平滑项权重
    :param iterations: 迭代次数
    :return: 平滑后的路径 (N,2)
    """
    smoothed = path.copy().astype(np.float32)
    
    for _ in range(iterations):
        gradient = np.zeros_like(smoothed)
        
        # 数据项梯度
        gradient += alpha * (smoothed - path)
        
        # 平滑项梯度
        for i in range(1, len(smoothed)-1):
            gradient[i] += beta * (2 * smoothed[i] - smoothed[i-1] - smoothed[i+1])
        
        smoothed -= gradient
    
    return smoothed

def spline_smooth(path, num_points=None):
    """
    样条插值平滑路径
    :param path: numpy数组 (N,2)
    :param num_points: 输出点数 (None则自动确定)
    :return: 平滑后的路径 (M,2)
    """
    if len(path) < 4:
        return path.copy()
    
    if num_points is None:
        num_points = min(100, len(path)*3)
    
    # 计算累积距离参数
    dist = np.zeros(len(path))
    for i in range(1, len(path)):
        dist[i] = dist[i-1] + np.linalg.norm(path[i] - path[i-1])
    
    if dist[-1] == 0:  # 所有点重合的情况
        return path.copy()
    
    t = dist / dist[-1]
    
    # 创建样条
    cs_x = CubicSpline(t, path[:,0])
    cs_y = CubicSpline(t, path[:,1])
    
    # 生成新点
    new_t = np.linspace(0, 1, num_points)
    new_x = cs_x(new_t)
    new_y = cs_y(new_t)
    
    return np.column_stack((new_x, new_y))

def preprocess_path(path):
    """组合平滑方法"""
    # 梯度下降平滑
    smoothed = gradient_smooth(path, alpha=0.15, beta=0.4, iterations=100)
    
    # 样条插值
    smoothed = spline_smooth(smoothed)
    
    return smoothed

class iLQR:
    """
    iLQR路径优化器
    """
    def __init__(self, initial_path, obstacles, dt=0.1, max_iter=100, tol=1e-3):
        self.path = initial_path.copy()
        self.obstacles = obstacles
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.N = len(initial_path)
        
        self.nx = 4  # [x, y, vx, vy]
        self.nu = 2  # [ax, ay]
        
        self.xs = np.zeros((self.N, self.nx))
        self.us = np.zeros((self.N-1, self.nu))
        
        for i in range(self.N):
            self.xs[i, 0] = initial_path[i, 0]
            self.xs[i, 1] = initial_path[i, 1]
        
        self.mass = 1.0
        self.damping = 0.1
        self.obstacle_safety_margin = 1
    
    def dynamics(self, x, u):
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + x[2] * self.dt
        x_next[1] = x[1] + x[3] * self.dt
        x_next[2] = x[2] + (u[0] - self.damping * x[2]) / self.mass * self.dt
        x_next[3] = x[3] + (u[1] - self.damping * x[3]) / self.mass * self.dt
        return x_next
    
    def obstacle_cost(self, x):
        cost = 0.0
        for obs in self.obstacles:
            dist = np.sqrt((x[0] - obs[0])**2 + (x[1] - obs[1])**2)
            safe_dist = obs[2] + self.obstacle_safety_margin
            if dist < safe_dist:
                cost += 1000 * np.exp(-2.0 * (dist - obs[2]))
        return cost
    
    def cost(self, x, u, t):
        ref_x = self.path[t, 0]
        ref_y = self.path[t, 1]
        state_cost = 0.5 * (10 * (x[0] - ref_x)**2 + 10 * (x[1] - ref_y)**2 + 0.1 * x[2]**2 + 0.1 * x[3]**2)
        control_cost = 0.5 * (0.1 * u[0]**2 + 0.1 * u[1]**2)
        obstacle_cost = self.obstacle_cost(x)
        return state_cost + control_cost + obstacle_cost
    
    def compute_derivatives(self, x, u, t):
        lx = np.zeros(self.nx)
        lx[0] = 10 * (x[0] - self.path[t, 0])
        lx[1] = 10 * (x[1] - self.path[t, 1])
        lx[2] = 0.1 * x[2]
        lx[3] = 0.1 * x[3]
        
        for obs in self.obstacles:
            dist = np.sqrt((x[0] - obs[0])**2 + (x[1] - obs[1])**2)
            safe_dist = obs[2] + self.obstacle_safety_margin
            if dist < safe_dist:
                dir_x = (x[0] - obs[0]) / (dist + 1e-6)
                dir_y = (x[1] - obs[1]) / (dist + 1e-6)
                grad_magnitude = 1000 * (-2.0) * np.exp(-2.0 * (dist - obs[2]))
                lx[0] += grad_magnitude * dir_x
                lx[1] += grad_magnitude * dir_y
        
        lu = np.array([0.1 * u[0], 0.1 * u[1]])
        lxx = np.diag([10, 10, 0.1, 0.1])
        luu = np.diag([0.1, 0.1])
        lux = np.zeros((self.nu, self.nx))
        lxu = np.zeros((self.nx, self.nu))
        
        fx = np.eye(self.nx)
        fx[0, 2] = self.dt
        fx[1, 3] = self.dt
        fx[2, 2] = 1 - self.damping * self.dt / self.mass
        fx[3, 3] = 1 - self.damping * self.dt / self.mass
        
        fu = np.zeros((self.nx, self.nu))
        fu[2, 0] = self.dt / self.mass
        fu[3, 1] = self.dt / self.mass
        
        return lx, lu, lxx, luu, lux, lxu, fx, fu
    
    def backward_pass(self):
        Vx = np.zeros((self.N, self.nx))
        Vxx = np.zeros((self.N, self.nx, self.nx))
        
        x = self.xs[-1]
        lx, lu, lxx, luu, lux, lxu, fx, fu = self.compute_derivatives(x, np.zeros(self.nu), self.N-1)
        Vx[-1] = lx
        Vxx[-1] = lxx
        
        k = np.zeros((self.N-1, self.nu))
        K = np.zeros((self.N-1, self.nu, self.nx))
        
        for t in range(self.N-2, -1, -1):
            x = self.xs[t]
            u = self.us[t]
            
            lx, lu, lxx, luu, lux, lxu, fx, fu = self.compute_derivatives(x, u, t)
            
            Qx = lx + fx.T @ Vx[t+1]
            Qu = lu + fu.T @ Vx[t+1]
            Qxx = lxx + fx.T @ Vxx[t+1] @ fx
            Quu = luu + fu.T @ Vxx[t+1] @ fu
            Qux = lux + fu.T @ Vxx[t+1] @ fx
            
            Quu_inv = np.linalg.inv(Quu)
            k[t] = -Quu_inv @ Qu
            K[t] = -Quu_inv @ Qux
            
            Vx[t] = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx[t] = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
        
        return k, K
    
    def forward_pass(self, k, K, alpha=1.0):
        xs_new = np.zeros_like(self.xs)
        us_new = np.zeros_like(self.us)
        xs_new[0] = self.xs[0]
        
        total_cost = 0.0
        
        for t in range(self.N-1):
            us_new[t] = self.us[t] + alpha * k[t] + K[t] @ (xs_new[t] - self.xs[t])
            xs_new[t+1] = self.dynamics(xs_new[t], us_new[t])
            total_cost += self.cost(xs_new[t], us_new[t], t)
        
        total_cost += self.cost(xs_new[-1], np.zeros(self.nu), self.N-1)
        
        return xs_new, us_new, total_cost
    
    def optimize(self):
        prev_cost = np.inf
        
        for i in range(self.max_iter):
            k, K = self.backward_pass()
            
            alpha = 1.0
            while alpha > 1e-4:
                xs_new, us_new, total_cost = self.forward_pass(k, K, alpha)
                
                if total_cost < prev_cost:
                    self.xs = xs_new
                    self.us = us_new
                    prev_cost = total_cost
                    break
                else:
                    alpha *= 0.5
            else:
                break
            
            if i > 0 and np.abs(prev_cost - total_cost) < self.tol: #检查收敛
                break
        
        return self.xs[:, :2]

def main():
    # 创建地图
    grid = np.zeros((100, 100))
    
    # 添加障碍物
    grid[3:10, 8] = 1
    grid[13, 20:23] = 1
    
    # 起点和终点
    start = (5, 4)
    goal = (25, 15)
    
    # A*路径规划
    astar = AStarPlanner(grid)
    raw_path = astar.planning(start[0], start[1], goal[0], goal[1])
    
    # 路径平滑预处理
    smoothed_path = preprocess_path(raw_path)
    
    # 创建障碍物列表
    obstacles = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                obstacles.append([j, i, 0.5])  # (x, y, radius)
    
    # iLQR路径优化
    ilqr = iLQR(smoothed_path, obstacles)
    optimized_path = ilqr.optimize()
    
    # 可视化
    plt.figure(figsize=(12, 10))
    
    # 绘制障碍物
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='black', alpha=1)
        plt.gca().add_patch(circle)
    
    # 绘制路径
    plt.plot(raw_path[:,0], raw_path[:,1], 'b-o', alpha=0.5, label='Raw A* Path', markersize=4)
    plt.plot(smoothed_path[:,0], smoothed_path[:,1], 'g--', linewidth=2, label='Smoothed Path')
    plt.plot(optimized_path[:,0], optimized_path[:,1], 'm-', linewidth=3, label='iLQR Optimized')
    
    # 起点和终点
    plt.plot(start[0], start[1], 'ro', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')
    
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title('Path Planning with Smoothing and Optimization')
    plt.show()

if __name__ == '__main__':
    main()