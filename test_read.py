import numpy as np

npz_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB323\_experiment_data\initial_model_data\initial_real_model_data_CB323.npz"
data = np.load(npz_path)

print("Depth range:", data['depth'][0], '~', data['depth'][-1])
print("TWT init range:", data['twt_init'][0], '~', data['twt_init'][-1])
print("Checkshot depth range:", data['checkshot_depth'][0], '~', data['checkshot_depth'][-1])
print("Checkshot TWT range:", data['checkshot_twt'][0], '~', data['checkshot_twt'][-1])
print("t_work range:", data['t_work'][0], '~', data['t_work'][-1])
print("t_well_top:", data['t_well_top'])
print("t_well_bot:", data['t_well_bot'])
