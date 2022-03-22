import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
import os.path, os
import numpy as np
from tqdm import tqdm
from glob import glob
import yaml
import subprocess

class football():
    def __init__(self, yaml_path, 
                 force_parsing = False, 
                 dataset4 = 'training',
                 system4 = 'windows',
                 draw_distribution = False,
                 pdf_report = False):
        self.yaml_path = yaml_path
        self.yaml_dict = self.yaml2dict()
        self.yaml_dict['info_folder'] = './info/'
        self.yaml_dict['live_folder'] = './live/'
        self.force_parsing = force_parsing
        self.dataset4 = dataset4
        self.system4 = system4
        self.draw_distribution = draw_distribution
        self.pdf_report = pdf_report
    def yaml2dict(self):
        with open(self.yaml_path) as f:
            arguments_dict = yaml.safe_load(f)
        print('Arguments loaded')
        return arguments_dict

    @staticmethod
    def download(info_link, live_link, info_folder = './info/',
                 live_folder = './live/'):
        print('Download data.........')
        if not os.path.isfile('./live.rar'):
            bashCommand = f'wget -q -O live.rar https://getfile.dokpub.com/yandex/get/{live_link}'
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            _, _exit_code = process.communicate()
        if not os.path.isfile('./info.rar'):
            bashCommand = f'wget -q -O info.rar https://getfile.dokpub.com/yandex/get/{info_link}'
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            _, _ = process.communicate()
        print('Unpack data.........')

        if not os.path.isfile(info_folder + 'info.csv'):
            bashCommand = f'unrar e -y ./info.rar {info_folder}'
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            _, _ = process.communicate() 
        if not os.path.isdir(live_folder):
            bashCommand = f'unrar e -y ./live.rar {live_folder}'
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            _, _ = process.communicate()
        for file_name in glob(info_folder + '*'):
            if 'info' in file_name.split('/')[-1]:
                os.rename(
                    file_name, 
                    info_folder + 'info.' +  file_name.split('.')[-1]      
                        )
        print('Data downloaded & unpacked')

    def get_info_dict(self):
        # забираем только нужные переменные
        # из матчей для которых есть файлы
        print('Search for matches by id')
        info_df = pd.read_csv(self.yaml_dict['info_folder'] + 'info.csv', 
                              sep = ';', 
                              header = None, 
                              usecols=[0, 8, 9, 10, 11, 14, 15, 16], 
                              index_col = 0)
        info_df['labels'] = list(np.sign(info_df[10] - info_df[11]).astype(str))
        #проверяем наличие файлов
        file_exist_list = []
        for matches_count, row in tqdm(enumerate(info_df.index), 
                                        total = len(info_df)):
            if os.path.isfile(f"{self.yaml_dict['live_folder']}{row}.csv"):
                file_exist_list.append(matches_count)
        return info_df.iloc[file_exist_list, -8:].astype('float32').to_dict('index')

    def parse_matches(self):
        from tqdm import tqdm
        from glob import glob
        print('Parsing matches by files in live directory.')
        id_live_set = frozenset([int(name.split('/')[-1].split('.')[0]) 
                                for name in glob(
                                    f"{self.yaml_dict['live_folder']}*.csv"
                                                )])
        final_list = []
        not_found = 0
        nan_blocks = 0
        parsing_start_minute = 35
        last_minute = 75
        minutes_control = [45, 60, 75]
        info_dict = self.get_info_dict()
        for row in tqdm(id_live_set, total = len(id_live_set)):
            if row in info_dict:
                match_list = []
                with open(f"{self.yaml_dict['live_folder']}{row}.csv", 'r') as ut:
                    match_list += [row]
                    #сначала записываем в датасет премаркет, результаты 1 тайма и матча
                    match_list += list(info_dict[row].values())
                    tmp_list =[]
                    minute_info_list = []
                    for line in ut:
                        current_minute = int(line.split(';')[0]) #текущая минута
                        tmp_list = line.strip().split(';')[
                                                self.yaml_dict['koeffs_start']:\
                                                self.yaml_dict['koeffs_start'] + 3
                                                            ] #текущие кэфы
                        tmp_list += line.strip().split(';')[
                                                self.yaml_dict['koeffs_start'] + 3:\
                                                self.yaml_dict['koeffs_start'] + 5
                                                            ] #текущий результат
                        if current_minute >= parsing_start_minute:
                            ################### parsing conditions ####################
                            if '' not in tmp_list:
                                if minute_info_list: 
                                    if (minute_info_list[3] <= list(np.float32(tmp_list))[3]) & \
                                    (minute_info_list[4] <= list(np.float32(tmp_list))[4]):
                                        minute_info_list = list(np.float32(tmp_list))
                                else:
                                    minute_info_list = list(np.float32(tmp_list))
                            if current_minute in minutes_control:
                                if minute_info_list:
                                    match_list += minute_info_list
                                else:
                                    match_list += [np.nan, np.nan, np.nan, np.nan, np.nan]
                                    nan_blocks += 1
                            ################### parsing conditions ####################
                            if current_minute > last_minute -1:
                                break
                final_list.append(match_list)
            else:
                not_found +=1
        print('\n')
        print(f'files not in info: {not_found}')
        print(f'nan lines: {nan_blocks}')    

        return final_list

    def clean_dataset(self, live_df:pd.DataFrame):
        del_list = [3600669]
        if self.yaml_dict['clean_with_file']:
            clean_set = frozenset(pd.read_csv(yaml_dict['clean_file_name'], 
                        compression = 'gzip')['idx'].to_list() + del_list)
        else:
            clean_set = frozenset(del_list)
        live_len = len(live_df)
        if self.yaml_dict['dropna']:
            live_df = live_df.dropna() # Удаляем nan
        print(f'Удалено nanов: {live_len - len(live_df)}')
        live_len = len(live_df)
        if self.yaml_dict['clean_with_file']:
            live_df = live_df.loc[~(live_df.idx.isin(clean_set))] # Удаляем oшибочные  и шаблон (если валидация) по номеру
        print(f'Cleaned with file: {live_len - len(live_df)}')
        live_len = len(live_df)
        live_df = live_df.loc[~((live_df.home_half1 != live_df.parsed_home_45min) | \
                                (live_df.away_half1 != live_df.parsed_away_45min))]
        print(f'1 half parsing trust clean: {live_len - len(live_df)}')
        live_len = len(live_df)
        k_type_list = ['K1', 'KX', 'K2']
        time_point_list = ['pre', 'min45', 'min60', 'min75']
        up_lo_list = ['_upper', '_lower']
        for time_point in time_point_list:
            for k_type in k_type_list:
                for up_lo in up_lo_list:
                    live_df_column = k_type + time_point
                    yaml_dict_index = k_type + time_point + up_lo
                    if yaml_dict_index in self.yaml_dict:
                        if 'upper' in up_lo:
                            live_df = live_df.loc[live_df[live_df_column] <= self.yaml_dict[yaml_dict_index]]
                        elif 'lower' in up_lo:
                            live_df = live_df.loc[live_df[live_df_column] >= self.yaml_dict[yaml_dict_index]]
        print(f'Clean with apply upper lower threshold for K columns: {live_len - len(live_df)}')
        print(f'Matches quantity: {len(live_df)}')
        return live_df.reset_index(drop = True)

    def add_k_by_time(self, live_df:pd.DataFrame):
        k_type_list = ['K1', 'KX', 'K2', 'KX2']
        min_list = ['45', '60', '75']
        res_dict = {'K1':1, 'KX':0, 'K2':-1, 'KX2':1}
        for minx in min_list:
            for k_coeff in k_type_list: 
                if k_coeff == 'KX2':
                    live_df[f'min{minx}KX2'] = live_df[f'min{minx}KX'] * live_df[f'min{minx}K2'] / (live_df[f'min{minx}KX'] + live_df[f'min{minx}K2'])
                    live_df[f'min{minx}KX2'] = [row if row > 1 else 1 for row in live_df[f'min{minx}KX2'].values]        
                    live_df[f'{k_coeff}_ret{minx}'] = [-row[1] + 1 if row[0] < res_dict[k_coeff] else 1 for row in live_df[['Result', f'min{minx}KX2']].values]
                    live_df.loc[0,[f'{k_coeff}_ret{minx}']] = live_df.loc[0,[f'{k_coeff}_ret{minx}']] + 100000
                else:
                    live_df[f'{k_coeff}_ret{minx}'] = [-row[1] + 1 if row[0] == res_dict[k_coeff] else 1 for row in live_df[['Result', f'min{minx}{k_coeff}']].values]
                    live_df.loc[0,[f'{k_coeff}_ret{minx}']] = live_df.loc[0,[f'{k_coeff}_ret{minx}']] + 100000
                live_df[f'{k_coeff}_ret_sh1_{minx}'] = live_df[f'{k_coeff}_ret{minx}'].cumsum().shift(1)
                live_df[f'{k_coeff}_ret_sh2_{minx}'] = live_df[f'{k_coeff}_ret{minx}'].cumsum().shift(2)
                live_df.loc[0,[f'{k_coeff}_ret_sh1_{minx}']] = 100000
                live_df.loc[0,[f'{k_coeff}_ret_sh2_{minx}']] = 100000
                live_df.loc[1,[f'{k_coeff}_ret_sh2_{minx}']] = 100000
        return live_df
    
    def normalization(self, live_df:pd.DataFrame):
        print('Normalazing data...............')
        if ('home_half_threshold' in self.yaml_dict) & ('away_half_threshold' in self.yaml_dict):
            home_half_threshold = self.yaml_dict['home_half_threshold']
            live_df['home_half1_norm'] = [x/home_half_threshold if x <= home_half_threshold else 1 for x in live_df['home_half1']]
            away_half_threshold = self.yaml_dict['away_half_threshold']
            live_df['away_half1_norm'] = [x/away_half_threshold if x <= away_half_threshold else 1 for x in live_df['away_half1']]
            for minute in ['45', '60', '75']:
                home_half_threshold = 4
                live_df[f'parsed_home_{minute}min_norm'] = [x/home_half_threshold if x <= home_half_threshold else 1 for x in live_df[f'parsed_home_{minute}min']]
                away_half_threshold = 4
                live_df[f'parsed_away_{minute}min_norm'] = [x/away_half_threshold if x <= away_half_threshold else 1 for x in live_df[f'parsed_away_{minute}min']]
        else:
            print('<<_No for goals normalization threshold present in yaml config file_>>')

        time_points = ['pre', 'min45', 'min60', 'min75']
        coeff_list = ['K1', 'KX', 'K2']
        for time_point in time_points:

            for coeff in coeff_list:
                if (time_point + coeff + '_pos_threshold' in self.yaml_dict) & \
                (time_point + coeff + '_scale' in self.yaml_dict) &\
                (time_point + coeff + '_posneg_threshold' in self.yaml_dict):
                    preK_pos_threshold = self.yaml_dict[time_point + coeff + '_pos_threshold']
                    preK_scale = self.yaml_dict[time_point + coeff + '_scale']
                    live_df[time_point + coeff + '_norm_pos'] = [x / preK_scale 
                                                if x <= preK_pos_threshold 
                                                else preK_pos_threshold / preK_scale
                                                for x in live_df[time_point + coeff]]

                    preK_posneg_threshold = self.yaml_dict[time_point + coeff + '_posneg_threshold']
                    live_df[time_point + coeff + '_norm_posneg'] = [x - preK_posneg_threshold 
                                                    if x < preK_posneg_threshold 
                                                    else (x - preK_posneg_threshold) / preK_scale 
                                                    for x in live_df[time_point + coeff]]
                    live_df[time_point + coeff + '_norm_posneg'] = [x 
                                                    if x <= preK_pos_threshold / preK_scale 
                                                    else preK_pos_threshold / preK_scale 
                                                    for x in live_df[time_point + coeff + '_norm_posneg']]
                else:
                    print(f'<<_{time_point + coeff} Not enough arguments in yaml config file for normalization_>>')
        return live_df
    
    def csv4win(self, live_df:pd.DataFrame):
        line_terminator = '\n'
        if self.system4 == 'windows':
            line_terminator = '\r\n'
        live_df.loc[:, ['idx', 'Result', 'home_half1', 'away_half1',
                'preK1', 'preKX', 'preK2',  
                'preK1_norm_pos', 'preKX_norm_pos', 'preK2_norm_pos',
                'preK1_norm_posneg', 'preKX_norm_posneg', 'preK2_norm_posneg',                 
                'parsed_home_45min_norm', 'parsed_away_45min_norm',
                'min45K1', 'min45KX', 'min45K2',
                'min45K1_norm_pos', 'min45KX_norm_pos', 'min45K2_norm_pos',
                'min45K1_norm_posneg', 'min45KX_norm_posneg', 'min45K2_norm_posneg',
                'parsed_home_60min_norm', 'parsed_away_60min_norm',
                'min60K1', 'min60KX', 'min60K2',
                'min60K1_norm_pos', 'min60KX_norm_pos', 'min60K2_norm_pos',
                'min60K1_norm_posneg', 'min60KX_norm_posneg', 'min60K2_norm_posneg',
                'parsed_home_75min_norm', 'parsed_away_75min_norm',
                'min75K1', 'min75KX', 'min75K2',
                'min75K1_norm_pos', 'min75KX_norm_pos', 'min75K2_norm_pos',
                'min75K1_norm_posneg', 'min75KX_norm_posneg', 'min75K2_norm_posneg',
                'K1_ret_sh2_45', 'KX_ret_sh2_45','K2_ret_sh2_45',
                'K1_ret_sh2_60', 'KX_ret_sh2_60','K2_ret_sh2_60',
                'K1_ret_sh2_75', 'KX_ret_sh2_75','K2_ret_sh2_75',
                ]].to_csv('base_line_data.csv', index = False, line_terminator = line_terminator)

    def download_parse_clean_normalize(self):

        self.download(self.yaml_dict['info_link'], self.yaml_dict['live_link'])
        col_names = ['idx', 'home_half1', 'away_half1', 'home_score', 'away_score',
        'preK1', 'preKX', 'preK2', 'Result', 'min45K1', 'min45KX', 'min45K2', 
        'parsed_home_45min', 'parsed_away_45min', 'min60K1', 'min60KX', 'min60K2',
        'parsed_home_60min', 'parsed_away_60min', 'min75K1', 'min75KX', 'min75K2',
        'parsed_home_75min', 'parsed_away_75min']
        if self.force_parsing:
            live_df = pd.DataFrame(self.parse_matches(), columns = col_names)
            live_df.to_csv('./_live_df,csv.gz', index = False, compression='gzip')
        else:
            if not os.path.isfile('./_live_df,csv.gz'):
                live_df = pd.DataFrame(self.parse_matches(), columns = col_names)
                live_df.to_csv('./_live_df,csv.gz', index = False, compression='gzip')
            else:
                live_df = pd.read_csv('./_live_df,csv.gz', compression='gzip')
        live_df = self.clean_dataset(live_df)
        live_df = self.add_k_by_time(live_df)
        live_df = self.normalization(live_df)
        self.csv4win(live_df)
        return live_df
