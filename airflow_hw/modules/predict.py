# <YOUR_IMPORTS>
import dill
import json
import pandas as pd
import logging
import os

from datetime import datetime

def predict() -> None:
    # <YOUR_CODE>
    path = os.environ.get('PROJECT_PATH', os.path.dirname(os.path.dirname(__file__)))
    dir_list = [os.path.join(f'{path}/data/models/', x) for x in os.listdir(f'{path}/data/models/')]
    if dir_list:
        # Создадим список из путей к файлам и дат их создания.
        date_list = [[x, os.path.getctime(x)] for x in dir_list]
        # Отсортируем список по дате создания в обратном порядке
        sort_date_list = sorted(date_list, key=lambda x: x[1], reverse=True)
        # Выведем первый элемент списка. Он и будет самым последним по дате
        logging.info(sort_date_list[0][0])
        with open(sort_date_list[0][0], 'rb') as file:
            model = dill.load(file)
        df_csv = pd.DataFrame(columns=['car_id', 'pred'])
        test_list = [os.path.join(f'{path}/data/test/', x) for x in os.listdir(f'{path}/data/test/')]
        for x in test_list:
            with open(x, 'rb') as f:
                json_data = json.loads(f.read())
            df = pd.DataFrame([json_data])
            y = model.predict(df)
            df_csv = df_csv._append({'car_id': df['id'][0], 'pred': y[0]}, ignore_index=True)
        csv_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
        df_csv.to_csv(csv_filename, index=False)

    else:
        logging.info('No pkl files')



if __name__ == '__main__':
    predict()
