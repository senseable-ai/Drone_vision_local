import pandas as pd

delta_t = 0.03

def calculate_velocity_acceleration(group):
    group = group.copy()
    group['v_x'] = round((group['center_x'].diff()) / 3.6 / delta_t, 2)
    group['v_y'] = round((group['center_y'].diff()) / 3.6 / delta_t, 2)
    group['a_x'] = round(group['v_x'].diff() / delta_t, 2)
    group['a_y'] = round(group['v_y'].diff() / delta_t, 2)

    group['v_x'].iloc[0] = 0
    group['v_y'].iloc[0] = 0
    group['a_x'].iloc[0:2] = 0
    group['a_y'].iloc[0:2] = 0

    return group

for clip_number in range(1, 121):
    try:
        file_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{clip_number}/clip{clip_number}_coordinate.csv'
        data = pd.read_csv(file_path)

        data_calculated = data.groupby('ID').apply(calculate_velocity_acceleration)
        data_to_save = data_calculated.drop(columns=['x1', 'x2', 'y1', 'y2'])

        output_file_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{clip_number}/clip{clip_number}_speed.csv'
        data_to_save.to_csv(output_file_path, index=False)
    except Exception as e:
        print(f"Error in processing clip{clip_number}: {e}")
