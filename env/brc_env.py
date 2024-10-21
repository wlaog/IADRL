import sys
import os

from matplotlib import animation, pyplot as plt
# 获取当前脚本文件夹的上一级目录并将项目根目录添加到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from env.Parameters_Generator.GMM import myData,GMMGenerator
from env.Pedmodel.ped_res import DriverPedestrianModel
from env.Belief.pre_train import ModelComparison

class Queue:
    def __init__(self, size):
        # 初始化队列长度为 size，全为0
        self.state = np.zeros(size)

    def update(self, new_elements):
        # 判断新元素数量是否为6
        if len(new_elements) != 6:
            raise ValueError("每次更新必须包含6个元素")

        # 维护先进先出队列：移除前面6个元素，并在尾部添加新的6个元素
        self.state = np.concatenate((self.state[6:], new_elements))

class BRCEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super(BRCEnv, self).__init__()
        
        # 动作空间：假设为-1到1之间的连续值
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_scaler = 3
        # 状态空间：假设有6*3*5维状态 6:x、v、a*2+P; 3:3s; 5:dt=0.2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((6)*3*5,), dtype=np.float32)
        
        # 初始状态和其他变量
        self.state = []
        self.state_Q = [] #state是当前时刻状态，state_Queue 是时序状态
        self.reward = 0.0
        self.terminated = False
        self.truncated = False

        self.dt=0.2
        self.pdecrdt=0.02
        
        self.Para = None
        self.init_state = None
        self.pedmodel = None

        # 初始化初始状态列表
        file_path = 'env\Parameters_Generator\initprio1.csv'
        # 使用 pandas 读取 CSV 文件
        data = pd.read_csv(file_path)
        # 将 DataFrame 转换为 NumPy 数组
        data_array = data.to_numpy()
        # # 显示导入的数据
        # print(data)
        self.init_array = data_array

        # 初始化Belief模型
        file_path = 'env/Belief/exp_data.csv'  # 数据文件路径
        model_comp = ModelComparison(file_path)   # 实例化 ModelComparison 类
        model_comp.train_model('Random Forest')
        self.rf_model = model_comp.trained_models['Random Forest']
        self.rf_scaler = model_comp.scaler  # 使用之前训练的 scaler
        print('初始化完毕')


    def reset(self, seed=None, options=None):
        """
            更新的逻辑是行人初始条件遍历，重置Para
        """
        super().reset(seed=seed)
        # 重置状态、奖励和终止标志
        # 初始化状态
        self.num = 0
        self.reward = 0.0 
        self.terminated = False
        self.truncated = False

        # 初始化行人Para
        file_path = 'env\\Parameters_Generator\\para_all.csv'
        data = pd.read_csv(file_path)
        data_array = myData(data.to_numpy())
        generator = GMMGenerator(n_components=12, covariance_type='diag')
        generator.fit(data_array.data_scaled)
        generated_samples_scaled = generator.generate_samples()
        self.Para = data_array.scaler.inverse_transform(generated_samples_scaled)
        self.Para = self.Para.flatten().tolist()

        # 更新case（初始状态)
        obs = self.case_up()


        return obs, {}

    def case_up(self):
        """
            用于遍历case，下一个case
        """

        # 初始化状态
        self.init_state = self.state_rand()
        self.pedmodel = DriverPedestrianModel(self.Para, self.init_state, self.dt, self.pdecrdt)
        self.state = np.concatenate((self.pedmodel.car_state[-2], self.pedmodel.ped_state[-2]))
        # self.reward = 0.0 除了reward其他的全都重置
        self.terminated = False
        self.truncated = False

        # 初始化意图值
        self.reward0 = self.belief_cal(self.state)

        # 记录车辆和行人位置
        # TODO 要记录到num维度下
        self.x = [self.pedmodel.car_state[-1][0]]  # 初始车辆位置
        self.y = [self.pedmodel.ped_state[-1][0]]  # 初始行人位置

        # 记录时序状态
        self.state_Q = Queue(6*3*5)
        self.state_Q.update(self.state)

        # 将状态转换为 float32 类型
        obs = self.state_Q.state.astype(np.float32)

        # 确保状态在 observation_space 内
        assert self.observation_space.contains(obs), f"状态 {obs} 不在定义的状态空间内"        

        return obs

    def vehicle_limit(self,action):
        """
            input:
                velocity: 当前速度
                acceleration: 当前加速度
                action: 动作空间原始输入
            return: 
                car_acc满足车辆状态约束的修正加速度
        """
        dt = 0.2
        # 当前状态
        velocity,acceleration = self.pedmodel.car_state[-1][1],self.pedmodel.car_state[-1][2]
        # 约束参数
        acmax, acmin = self.pedmodel.acmax, self.pedmodel.acmin
        jcmax, jcmin = self.pedmodel.jcmax, self.pedmodel.jcmin
        vcmax, vcmin = self.pedmodel.vcmax, self.pedmodel.vcmin
        # 加速度范围
        acc_max = min(acmax, 
                     (vcmax - velocity) / dt,
                      jcmax * dt + acceleration)
        acc_min = max(acmin,
                      (vcmin - velocity) / dt,
                      jcmin * dt + acceleration)

        car_acc_input = action * self.action_scaler #从动作空间转换到车辆实际空间
        car_acc = np.clip(car_acc_input, acc_min, acc_max)

        return car_acc


    def step(self, action):
        car_acc = self.vehicle_limit(action)
        state = self.State_update(car_acc)  # 更新状态
        reward = self.reward_cal(action, state)
        case_done, is_done = self.is_Done(state)
        
        # 更新当前状态、奖励和终止标志
        self.state, self.reward, self.terminated = state, reward, is_done
        self.state_Q.update(self.state)

        # 将状态转换为 float32 类型
        obs = self.state_Q.state.astype(np.float32)

        # 确保状态在 observation_space 内
        assert self.observation_space.contains(obs), f"状态 {obs} 不在定义的状态空间内"

        # 确保奖励是 float 类型
        reward = float(reward)

        # # 记录车辆和行人位置
        # self.x.append(self.pedmodel.car_state[-1][0])  # 车辆位置
        # self.y.append(self.pedmodel.ped_state[-1][0])  # 行人位置
        
        # 更新初始状态
        if case_done:
            self.num+=1
            self.case_up()

        return obs, reward, is_done, self.truncated, {}

    def render(self, mode='human'):
        if mode == 'human':
            x = np.array(self.x)
            y = np.array(self.y)

            # 设置坐标轴范围，固定不变
            x_min, x_max = min(0,np.min(x)) - 1, np.max(x) + 1
            y_min, y_max = min(0,np.min(y)) - 1, np.max(y) + 1

            fig, ax = plt.subplots()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_title("Pedestrian-Vehicle Interaction")

            # 初始状态的散点
            scat = ax.plot([], [], 'ro')[0]  # 行人
            car = ax.plot([], [], 'bo')[0]   # 车辆

            # 更新函数：根据当前帧更新位置
            def update(frame):
                scat.set_data(0, y[frame])  # 行人位置
                car.set_data(x[frame], 0)           # 车辆位置始终在初始位置
                return scat, car

            # 创建动图
            ani = animation.FuncAnimation(fig, update, frames=len(x), blit=True, repeat=False)

            # 显示动图
            plt.show()
        elif mode == 'rgb_array':
            return np.zeros((400, 600, 3), dtype=np.uint8)  # 返回一个示例图像
        # if mode == 'Ped veh interaction':
            
    
    def state_rand(self):
        # index = random.sample(range(0, len(self.init_array) ),1)
        index = [self.num]
        state = self.init_array[index,:]
        init_state = {
            'car': state[0,0:3],
            'ped': state[0,3:6]
        }
        return init_state

    def State_update(self, car_acc):
        self.pedmodel.run_model(car_acc)
        state = np.concatenate((self.pedmodel.car_state[-2], self.pedmodel.ped_state[-2])) # 注意state-1车的加速度是借用了-2的，因此-2才是对齐的
        return state

    def reward_cal(self,action,state):
        # 意图奖励: 
        # reward = -(self.belief_cal(state1) - self.belief_cal(state2)) # 增量形式不好 
        reward = -(self.belief_cal(state)-self.reward0) 
        # reward = -(state[1]**2 - self.init_state['car'][1])
        
        # # 动作惩罚（惩罚不在加速度范围内的动作）
        # # 加速度惩罚
        # action =action * self.action_scaler
        # if action < self.pedmodel.acmin or action > self.pedmodel.acmax:
        #     reward -= 10  # 惩罚超出加速度限制的行为
        # # 速度惩罚
        # if action * self.dt + state2[1] < self.pedmodel.vcmin or action * self.dt + state2[1] > self.pedmodel.vcmax:
        #     reward -= 10  # 惩罚不符合速度限制的行为

   
        return reward
    
    def belief_cal(self,state):
        # TODO计算意图
        input_data = pd.DataFrame({
            'xcar': state[0],
            'vcar': state[1],
            'acar': state[2],
            'ttccar': state[0]/max(1e-5,state[1]),
            'xped': state[3],
            'vped': state[4],
            'aped': state[5],
            'ttcped': state[3]/max(1e-5,state[4]),
        }, index=[0])
        input_data_scaled = self.rf_scaler.transform(input_data)
        output_data = np.clip(self.rf_model.predict(input_data_scaled), 0, 1)
        return output_data


    def is_Done(self,state):
        thound = 0.1
        if state[0] <thound or state[3] <thound or self.belief_cal(state) >= 0.95:
            case_done = True
        else:
            case_done = False

        if case_done == True and self.num == 290: 
            is_done = True
        else:
            is_done =False

        return case_done, is_done

    def close(self):
        pass

if __name__ == "__main__":
    env = BRCEnv()

    state, info = env.reset(seed=42)

    # 进行若干步的仿真
    for huihe in range(10):
        state, info = env.reset()
        print(f'round{huihe}')
        for t in range(1000):
            action = env.action_space.sample()  # 采取随机动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t+1))
                observation, info = env.reset()
                break
            # print(f"Step: Reward={reward}, Terminated={terminated}")
            # if terminated:
            #     break
    # # 关闭环境
    env.close()
    
