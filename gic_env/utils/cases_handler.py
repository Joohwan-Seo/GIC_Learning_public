import numpy as np
def get_cases(case):
    if case == 'case1':
        xd = np.array([0.65, 0.1, 0.08])
        Rt = np.array([[1, 0, 0],
                        [0, 0.8660, -0.50],
                        [0,0.50,0.8660]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case1.xml"

    elif case == 'case2':
        xd = np.array([0.75, 0.00, 0.15])
        Rt = np.array([[0.8660, 0, -0.5],
                        [0, 1, 0],
                        [0.5, 0, 0.8660]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case2.xml"

    elif case == 'case3':
        xd = np.array([1.05, 0.00, 0.35])
        Rt = np.array([[0, 0, -1],
                        [0, 1, 0],
                        [1, 0, 0]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case3.xml"

    if case == 'case_p30':
        xd = np.array([0.65, 0.1, 0.08])
        Rt = np.array([[1, 0, 0],
                        [0, 0.8660, -0.50],
                        [0,0.50,0.8660]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case_p30.xml"

    elif case == 'case_m30':
        xd = np.array([0.65,-0.1,0.08])
        Rt = np.array([[1, 0, 0],
                        [0, 0.8660, 0.50],
                        [0, -0.50,0.8660]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case_m30.xml"
            
    elif case == 'case_p60':
        xd = np.array([0.65,0.2,0.2])
        Rt = np.array([[1, 0, 0],
                        [0, 0.50, -0.8660],
                        [0, 0.8660,0.50]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case_p60.xml"
        
    elif case == 'case_m60':
        xd = np.array([0.65,-0.2,0.2])
        Rt = np.array([[1, 0, 0],
                        [0, 0.50, 0.8660],
                        [0, -0.8660,0.50]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case_m60.xml"
    
    elif case == 'case_p90':
        xd = np.array([0.65,0.4,0.4])
        Rt = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case_p90.xml"

    elif case == 'case_m90':
        xd = np.array([0.65,-0.4,0.4])
        Rt = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case_m90.xml"
    
    elif case == 'case_p120':
        xd = np.array([0.65, 0.4,0.8])
        Rt = np.array([[1, 0, 0],
                        [0, -0.5, -0.8660],
                        [0, 0.8660, -0.5]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case_p120.xml"

    elif case == 'case_m120':
        xd = np.array([0.65, -0.4,0.8])
        Rt = np.array([[1, 0, 0],
                        [0, -0.5, 0.8660],
                        [0, -0.8660, -0.5]])
        
        name = "gic_env/mujoco_models/pih/square_pih_fanuc_case_m120.xml"

    return xd, Rt, name

def get_file_names(cases):
    pass
    return 