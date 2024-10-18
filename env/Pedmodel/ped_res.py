import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import pandas as pd

class DriverPedestrianModel:
    def __init__(self, Para, init_state, dt=0.2, pdecrdt=0.02, beliefmodel=None):
        # Parameters and constants
        self.Para = Para
        self.dt = dt
        self.pdecrdt = pdecrdt
        self.acmax, self.acmin = 3, -3
        self.jcmax, self.jcmin = 20, -20
        self.apmax, self.apmin = 1, -1
        self.jpmax, self.jpmin = 20, -20
        self.vpmin, self.vpmax = 0, 2.5
        self.vcmin, self.vcmax = 0, 20
        self.Inf0 = 1e-3
        self.car_state = []
        self.ped_state = []
        self.car_state.append(init_state['car'])
        self.ped_state.append(init_state['ped'])
        self.table_Sp = [
            {'acc': [self.ped_state[0][2]], 'sigma': [1]}
        ]
        self.table_Sc = [
            {'acc': [self.car_state[0][2]], 'sigma': [1]}
        ]
        self.run_model(self.car_state[0][2])
        self.beliefmodel = beliefmodel

    def compute_intention(self, prev_TTCc, prev_TTCp):
        Tper = self.Para[8]
        Pp_Y = 1 / (1 + np.exp(np.clip(prev_TTCc - prev_TTCp + Tper, -200, 200)))
        # if self.beliefmodel is not None:
        #     input_data = pd.DataFrame({
        #         'xcar': state[0],
        #         'vcar': state[1],
        #         'acar': state[2],
        #         'ttccar': state[0]/max(1e-5,state[1]),
        #         'xped': state[3],
        #         'vped': state[4],
        #         'aped': state[5],
        #         'ttcped': state[3]/max(1e-5,state[4]),
        #     })
        #     input_data_scaled = self.beliefmodel.scaler.transform(input_data)
        #     output_data = np.clip(self.beliefmodel.predict(input_data_scaled), 0, 1)
        #     return output_data, Pp_Y
        # else:
        return 1 - Pp_Y, Pp_Y

    def compute_next_state(self, x, v, a, A, B):
        next_states = A @ np.array([x, v]).reshape(-1, 1) + B * a
        return next_states[0, :], next_states[1, :], next_states[0, :] / np.clip(next_states[1, :], self.Inf0, None)

    def decision_making(self, A_b2b, A_b2v, sigma_p2c, sigma_p2p, Lambda_v, Lambda_b, Up, Uv, Tgame):
        p_b2v = np.zeros((A_b2v.shape[0], Tgame+1))
        p_b2b = np.zeros((A_b2b.shape[0], Tgame+1))
        
        for k in range(1, Tgame + 1):
            p_b2v[:, k] = np.exp(Lambda_v * (A_b2v[:, k - 1] + sigma_p2c))
            p_b2v[:, k] /= p_b2v[:, k].sum()

            A_b2b[:, k] = A_b2b[:, k - 1] + Up @ p_b2v[:, k]
            p_b2b[:, k] = np.exp(Lambda_b * (A_b2b[:, k] + sigma_p2p))
            p_b2b[:, k] /= p_b2b[:, k].sum()
            A_b2v[:, k] = A_b2v[:, k - 1] + Uv @ p_b2b[:, k]

        return A_b2b, A_b2v, p_b2v, p_b2b

    def run_model(self, car_action):
        """
        当前时刻人x,v,a,车x,v，函数输入车a后行人根据现状计算a'
        """
        self.car_state[-1][2] = car_action
        A = np.array([[1, -self.dt], [0, 1]])
        B = np.array([[-self.dt**2 / 2], [self.dt]])
        a = self.car_state[-1][ :2]
        b = np.array([car_action]).reshape(1)
        prev_car = np.concatenate([a, b])
        prev_ped = self.ped_state[-1][:3]
        
        prev_xc, prev_vc, prev_ac = prev_car
        prev_xp, prev_vp, prev_ap = prev_ped
        prev_TTCc = prev_car[0] / np.clip(prev_car[1], self.Inf0, None)
        prev_TTCp = prev_ped[0] / np.clip(prev_ped[1], self.Inf0, None)

        # 更新状态和意图
        pres_xc, pres_vc, pres_TTCc = self.compute_next_state(prev_xc, prev_vc, prev_ac, A, B)
        pres_xp, pres_vp, pres_TTCp = self.compute_next_state(prev_xp, prev_vp, prev_ap, A, B)
        Pc_Y, Pp_Y = self.compute_intention(prev_TTCc, prev_TTCp)

        # 根据约束进行策略集生成
        table_Sc_acc = np.array([0, 1.5, -1.5, 3, -3])
        # 根据速度限制和时间步长计算允许的最大/最小加速度
        max_dec_ap = max([min(self.apmin,prev_ap), (min(self.vpmin,prev_vp) - prev_vp) / self.dt, self.dt * self.jpmin + prev_ap])  # 限制减速到不会超过速度下限
        max_acc_ap = min([max(self.apmax,prev_ap), (max(self.vpmax,prev_vp) - prev_vp) / self.dt, self.dt * self.jpmax + prev_ap])  # 限制加速到不会超过速度上限
        # 生成减速和加速加速度表
        table_Sp_dec1 = np.arange(prev_ap - self.pdecrdt, max_dec_ap, -self.pdecrdt)
        table_Sp_acc1 = np.arange(prev_ap + self.pdecrdt, max_acc_ap, self.pdecrdt)
        # table_Sp_dec1 = np.arange(prev_ap, max(self.apmin, self.dt * self.jpmin + prev_ap), -self.pdecrdt)
        # table_Sp_acc1 = np.arange(prev_ap+self.pdecrdt, min(self.apmax, self.dt * self.jpmax + prev_ap), self.pdecrdt)
        # 拼接两个数组
        table_Sp = np.concatenate((np.concatenate((table_Sp_dec1, table_Sp_acc1)),[prev_ap]))
        # 去掉范围外值
        filtered_Sp = table_Sp[(table_Sp > max_dec_ap) & (table_Sp < max_acc_ap)]
        # 按照保持程度排序
        table_Sp_acc = filtered_Sp[np.argsort(np.abs(filtered_Sp - prev_ap))]
        Nv, Np = len(table_Sc_acc), len(table_Sp_acc)

        # Initialize reward matrices Up and Uv
        Up, Up1, Up2, Up3, Up4 = np.zeros((Np, Nv)), np.zeros((Np, Nv)), np.zeros((Np, Nv)), np.zeros((Np, Nv)), np.zeros((Np, Nv))
        Uv, Uv1, Uv2, Uv3, Uv4 = np.zeros((Nv, Np)), np.zeros((Nv, Np)), np.zeros((Nv, Np)), np.zeros((Nv, Np)), np.zeros((Nv, Np))

        # 生成未来时刻虚拟状态表（可能性）------------------------------------------------------------
        # Initialize matrices for car and pedestrian next states
        car_next = np.zeros((4, Nv))
        ped_next = np.zeros((4, Np))
        # Fill future potential acceleration arrays for pedestrians and cars
        ped_next[2, :] = table_Sp_acc  # Fill acceleration for pedestrian
        car_next[2, :] = table_Sc_acc  # Fill acceleration for car

        # Calculate the pedestrian's next state
        NexX_ped = np.dot(A, np.array([pres_xp, pres_vp]).reshape(-1, 1)) + B * ped_next[2, :]
        ped_next[0, :] = NexX_ped[0, :]
        ped_next[1, :] = NexX_ped[1, :]
        ped_next[3, :] = NexX_ped[0, :] / np.maximum(self.Inf0, NexX_ped[1, :])

        # Calculate the car's next state
        NexX_car = np.dot(A, np.array([pres_xc, pres_vc]).reshape(-1, 1)) + B * car_next[2, :]
        car_next[0, :] = NexX_car[0, :]
        car_next[1, :] = NexX_car[1, :]
        car_next[3, :] = NexX_car[0, :] / np.maximum(self.Inf0, NexX_car[1, :])

        # Extract next state variables for easier use
        next_xp, next_xc = ped_next[0, :], car_next[0, :]
        next_vp, next_vc = ped_next[1, :], car_next[1, :]
        next_ap, next_ac = ped_next[2, :], car_next[2, :]

        # Calculate time difference between car and pedestrian for different accelerations
        detaT = np.zeros((Nv, Np))
        for ki in range(Nv):
            for kj in range(Np):
                detaT[ki, kj] = car_next[3, ki] - ped_next[3, kj]  # Time difference
        # 生成未来时刻虚拟状态表（可能性）----------------------------------------------------------------------------------------
        


        # 计算显著性------------------------------------------------------------------------------------------
        prev_ped_str = np.array([self.table_Sp[-1]['acc'],self.table_Sp[-1]['sigma']])
        prev_car_str = np.array([self.table_Sc[-1]['acc'],self.table_Sc[-1]['sigma']])
        # Calculate strategy compatibility measures
        sigma_b2b = np.sum([np.exp(-(table_Sp_acc - prev_ped_str[0, i])**2) * prev_ped_str[1, i]
                            for i in range(prev_ped_str.shape[1])], axis=0)
        sigma_b2v = np.sum([np.exp(-(table_Sc_acc - prev_car_str[0, i])**2) * prev_car_str[1, i]
                            for i in range(prev_car_str.shape[1])], axis=0)

        # Normalize
   
        sigma_b2b = np.clip((sigma_b2b - sigma_b2b.min()) / max(sigma_b2b.ptp(), self.Inf0), 0, 1)
        # 计算显著性------------------------------------------------------------------------------------------

        # 根据Para更新参数----------------------------------------------------------------------------
        Lambda_b, Lambda_v = 10 ** -self.Para[0], 10 ** -self.Para[0]
        vp_exp = self.Para[1]
        shape_pi1 = self.Para[3]
        shape_pi2 = self.Para[4]
        shape_pi3 = self.Para[5]
        shape_pi4 = self.Para[6]
        Tgoal_y, Tgoal_g = self.Para[7], self.Para[2]
        # Update goals based on pedestrian speed and current car TTC
        Tgoal_g = max(Tgoal_g, pres_TTCc - pres_xp / vp_exp)
        Tgoal_y = min(Tgoal_y, pres_TTCc - pres_xp / vp_exp)
        Ap = [self.Para[-3] * self.Para[-2], self.Para[-3] * (1 - self.Para[-2]), 
              (1 - self.Para[-3]) * self.Para[-1], (1 - self.Para[-3]) * (1 - self.Para[-1])]
        Av = 1
        # Calculate Up1, Up2, Up3, Up4 based on current state and parameters
        Up1[:, :] = -np.tile(np.abs(next_ap - 0) ** shape_pi1, (Nv, 1)).T
        
        Up2[:, :] = -np.tile(np.abs((next_vp - vp_exp) / 4) ** shape_pi2, (Nv, 1)).T
        Up3[:, :] = (1 - Pc_Y) * (np.abs(detaT.T / 8) ** shape_pi3)
        Up4[:, :] = -Pp_Y * (np.abs((detaT.T - Tgoal_y) / 10) ** shape_pi4) - \
                    (1 - Pp_Y) * (np.abs((detaT.T - Tgoal_g) / 10) ** shape_pi4)

        Uv1[:, :] = (1 - Pc_Y) * np.abs(np.tile(next_vc, (Np, 1)).T - (np.sign(-detaT) + 1) * pres_vc) - \
                    Pc_Y * np.abs(np.tile(next_vc, (Np, 1)).T - (np.sign(detaT) + 1) * pres_vc)

        # Define normalization function
        def normalize(matrix, Inf0=1e-9):
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            return (matrix - min_val) / np.maximum(Inf0, max_val - min_val)

        # Normalize the utility matrices
        Up1 = normalize(Up1)
        Up2 = normalize(Up2)
        Up3 = normalize(Up3)
        Up4 = normalize(Up4)
        Uv1 = normalize(Uv1)

        # Calculate combined utility matrices Up and Uv
        Up = Ap[0] * Up1 + Ap[1] * Up2 + Ap[2] * Up3 + Ap[3] * Up4
        Uv = Av * Uv1  # Extend if more Av terms are needed




        # Calculate action matrices for decision-making
        if abs(Pc_Y-0.5)>=0.25 and abs(Pp_Y-0.5)>=0.25:
            Tgame = 1
        else:
            Tgame = 2
        # Sample strategies and initialize decision variables
        A_b2b, A_b2v = np.zeros((Np, Tgame+1)), np.zeros((Nv, Tgame+1))
        A_b2b, A_b2v, p_b2v, p_b2b = self.decision_making(
            A_b2b, A_b2v, sigma_b2v, sigma_b2b, Lambda_v, Lambda_b, Up, Uv, Tgame
        )

        # Determine the best strategy
        position_b = np.argmax(A_b2b[:, -1])
        selected_action = table_Sp_acc[position_b]
        # 更新显著性
        temp_Sc = {'acc':table_Sc_acc.tolist(),'sigma':A_b2v[:,-1].tolist()}
        temp_Sp = {'acc':table_Sp_acc.tolist(),'sigma':A_b2b[:,-1].tolist()}
        self.table_Sc.append(temp_Sc)
        self.table_Sp.append(temp_Sp)

        self.ped_state.append(np.array([pres_xp.item(), pres_vp.item(),selected_action] ))
        self.car_state.append(np.array([pres_xc.item(), pres_vc.item(),prev_ac] ))
        # Return selected action and intention probabilities
        return selected_action
    

if __name__ == "__main__":
    # state_car = np.array([
    # [80.8702344949027, 13.4448900000000, 0.0347700000000000],
    # [78.1870857327298, 13.4328100000000, -0.0226100000000000],
    # [75.5029210199695, 13.4186000000000, -0.0300800000000000],
    # [72.8201182938261, 13.4078900000000, -0.00324000000000000],
    # [70.1423034216286, 13.4070800000000, 0.0522800000000000],
    # [67.4617955638530, 13.4272500000000, 0.100520000000000],
    # [64.7720093174090, 13.4526700000000, 0.122110000000000],
    # [62.0771616706778, 13.4788900000000, 0.130460000000000],
    # [59.3777269341471, 13.5044400000000, 0.137430000000000],
    # [56.6760171888454, 13.5304500000000, 0.155310000000000],
    # [53.9688204406490, 13.5681700000000, 0.163630000000000],
    # [51.2488878470800, 13.6030400000000, 0.145430000000000],
    # [48.5232201441546, 13.6309000000000, 0.116140000000000],
    # [45.7934425589432, 13.6461400000000, 0.102360000000000],
    # [43.0679428648814, 13.6660300000000, 0.113690000000000],
    # [40.3301561938765, 13.6906700000000, 0.117990000000000],
    # [37.5860351990272, 13.7113200000000, 0.122290000000000],
    # [34.8434235891925, 13.7329900000000, 0.151010000000000],
    # [32.0960990461319, 13.7650500000000, 0.194890000000000],
    # [29.3425216847342, 13.8137200000000, 0.226560000000000],
    # [26.5732122461693, 13.8688600000000, 0.214150000000000],
    # [23.7921525500743, 13.9104900000000, 0.173260000000000],
    # [21.0052957526308, 13.9408500000000, 0.127580000000000],
    # [18.2119806233876, 13.9566800000000, 0.0992200000000000],
    # [15.4197229214110, 13.9695400000000, 0.109890000000000],
    # [12.6293585353014, 13.9927700000000, 0.149860000000000],
    # [9.82936795187508, 14.0335300000000, 0.170560000000000],
    # [7.01590666029974, 14.0721600000000, 0.153720000000000],
    # [4.19621472662992, 14.1001500000000, 0.125510000000000],
    # [1.37274197866579, 14.1206800000000, 0.104520000000000]])
    # state_ped = np.array([
    # [6.67005386024020, 1.32739000000000, -0.264310000000000],
    # [6.41260893130431, 1.27959000000000, -0.232430000000000],
    # [6.16267812147511, 1.24210000000000, -0.221600000000000],
    # [5.91662347831793, 1.20556000000000, -0.247600000000000],
    # [5.67608521075888, 1.15460000000000, -0.298970000000000],
    # [5.44954661251874, 1.08609000000000, -0.343940000000000],
    # [5.24098186648575, 1.01121000000000, -0.370090000000000],
    # [5.04759426415845, 0.934670000000000, -0.386790000000000],
    # [4.86889405338927, 0.856440000000000, -0.403080000000000],
    # [4.70475159432715, 0.773870000000000, -0.420820000000000],
    # [4.55680247136292, 0.685570000000000, -0.433150000000000],
    # [4.42768560193887, 0.594810000000000, -0.432310000000000],
    # [4.31734392508221, 0.506630000000000, -0.419870000000000],
    # [4.22521449743046, 0.426600000000000, -0.410610000000000],
    # [4.14588293267740, 0.349670000000000, -0.423870000000000],
    # [4.07732226118818, 0.262400000000000, -0.455010000000000],
    # [4.02555006152759, 0.154409999999999, -0.457470000000000],
    # [4.00499205951392, 0.0447300000000000, -0.376250000000000],
    # [3.99053395959346, -0.0327300000000000, -0.215420000000000],
    # [3.96219522583561, -0.0639600000000000, -0.0188499999999991],
    # [3.92971799130264, -0.0514499999999998, 0.185260000000001],
    # [3.89142181250927, 0.00510000000000037, 0.386540000000001],
    # [3.85322159008864, 0.103770000000000, 0.570560000000000],
    # [3.80739792589029, 0.241060000000000, 0.722410000000000],
    # [3.73884830995354, 0.409080000000001, 0.824590000000000],
    # [3.63745184080451, 0.596250000000001, 0.860090000000000],
    # [3.49746786949794, 0.785480000000001, 0.817890000000000],
    # [3.31735949005235, 0.952420000000000, 0.708820000000000],
    # [3.11010432763191, 1.08649000000000, 0.571690000000000],
    # [2.88009620174290, 1.18924000000000, 0.428189999999999]])
    # Para = np.array([1.86200060539367,	
    #         1.30114199958428,	
    #         3.07860658144764,	
    #         3.14106907164788,	
    #         1.57982305530239	,
    #         0.285227346748221	,
    #         2.74244419658705,	
    #         -2.21955389826418	,
    #         -2.39164800623771	,
    #         0.0935576299671364	,
    #         0.439338585279223	,
    #         0.459295012079155])
  # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()

    file_path = r"C:\Users\WJC\Desktop\20241008意图表达DRL\Project_Test\env\Parameters_Generator\Intdata.mat"
    
    eng.eval(f"load('{file_path}')", nargout=0)
    init_i = 200
    eng.workspace['init_i'] = init_i
    eng.eval("temp_motion_car = Intdata(init_i).motion(:,1:3)", nargout=0)
    eng.eval("temp_motion_ped = Intdata(init_i).motion(:,5:7)", nargout=0)
    eng.eval("temp_Para = Intdata(init_i).para", nargout=0)
    state_ped = np.array(eng.workspace['temp_motion_ped'])
    state_car = np.array(eng.workspace['temp_motion_car'])
    Para = np.array(eng.workspace['temp_Para']).flatten()
    init_state = {
        'car': state_car[0,:],
        'ped': state_ped[0,:]
    }
    init_state['car'][1] = 0
    ped_gen = np.zeros_like(state_ped)
    pedmodel = DriverPedestrianModel(Para, init_state, dt=0.2, pdecrdt=0.02)
    for i in range(1,100):
        # car_acc = state_car[np.min([i, state_car.shape[0] - 1]), 2]
        car_acc = 0
        pedmodel.run_model(car_acc)

    genall = np.array(pedmodel.ped_state)
    ped_gen[:] = genall[:ped_gen.shape[0], :]
    # Extracting the second and third columns from the 'state_ped' data
    true_column = [row[1] for row in state_ped]
    gen_column = [row[1] for row in ped_gen]

    # Generating the time points with a 0.2 interval
    time_intervals = np.arange(0, len(state_ped) * 0.2, 0.2)

    # Plotting the second and third columns against the time intervals on the same plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, true_column, label="true_column", color="blue", marker="o")
    plt.plot(time_intervals, gen_column, label="gen_column", color="red", marker="x")

    # Adding labels, title, and legend
    plt.xlabel("Time (seconds)")
    plt.ylabel("Values")
    plt.title("Plot of True&Gen Columns of state_ped Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()


