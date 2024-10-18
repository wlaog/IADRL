from scipy.io import loadmat
import numpy as np
import matlab.engine

eng = matlab.engine.start_matlab()

file_path = r"C:\Users\WJC\Desktop\20241008意图表达DRL\Project_Test\env\Parameters_Generator\intdata.mat"
eng.eval(f"load('{file_path}')", nargout=0) 
init_state_array = []
for init_i in range(1,300):
    eng.workspace['init_i'] = init_i
    eng.eval("temp_motion_car = Intdata(init_i).motion(:,1:3)", nargout=0)
    eng.eval("temp_motion_ped = Intdata(init_i).motion(:,5:7)", nargout=0)
    eng.eval("temp_Para = Intdata(init_i).para", nargout=0)
    init_state_array.append(np.concatenate((np.array(eng.workspace['temp_motion_ped'],np.array(eng.workspace['temp_motion_car'])))))

    # Para = np.array(eng.workspace['temp_Para'])

a= 1
