import pandas as pd

file_path = 'C:/Users/user/Desktop/drone_vision_local/coordinate.csv'
data = pd.read_csv(file_path)

delta_t = 1/30

def calculate_velocity_acceleration(group):
    group = group.copy()
    group['v_x'] = (group['center_x'].diff()) / delta_t
    group['v_y'] = (group['center_y'].diff()) / delta_t
    group['a_x'] = group['v_x'].diff() / delta_t
    group['a_y'] = group['v_y'].diff() / delta_t

    group['v_x'].iloc[0] = 0
    group['v_y'].iloc[0] = 0
    group['a_x'].iloc[0:2] = 0
    group['a_y'].iloc[0:2] = 0

    return group

data_calculated = data.groupby('ID').apply(calculate_velocity_acceleration)

data_to_save = data_calculated.drop(columns=['x1', 'x2', 'y1', 'y2'])

output_file_path = 'C:/Users/user/Desktop/drone_vision_local/velocity.csv'
data_to_save.to_csv(output_file_path, index=False)

output_file_path