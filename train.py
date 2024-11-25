#!/usr/bin/env python
# coding: utf-8

import pickle 

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score



#data preparation

df = pd.read_csv('Obesity_Dataset.csv')
df.columns = df.columns.str.lower()
df = df.rename(columns = {'class' : 'weight_class', 'physical_excercise' : 'physical_exercise'})

#sex
sex_val = {1: 'male',
           2: 'female'}

df.sex = df.sex.map(sex_val)

#overweight_obese_family
overweight_obese_family_val = {1: 'yes',
                               2: 'no'}
df.overweight_obese_family = df.overweight_obese_family.map(overweight_obese_family_val)

#consumption_of_fast_food
consumption_of_fast_food_val = {1: 'yes',
                                2: 'no'}
df.consumption_of_fast_food = df.consumption_of_fast_food.map(consumption_of_fast_food_val)

#frequency_of_consuming_vegetables
frequency_of_consuming_vegetables_val = {1: 'rarely',
                                         2: 'sometimes',
                                         3: 'always'}
df.frequency_of_consuming_vegetables = df.frequency_of_consuming_vegetables.map(frequency_of_consuming_vegetables_val)

#number_of_main_meals_daily
number_of_main_meals_daily_val =  {1: '1-2',
                                   2: '2',
                                   3: '3+'}
df.number_of_main_meals_daily = df.number_of_main_meals_daily.map(number_of_main_meals_daily_val)

#food_intake_between_meals
food_intake_between_meals_val = {1: 'rarely',
                                 2: 'sometimes',
                                 3: 'usually',
                                 4: 'always'}
df.food_intake_between_meals = df.food_intake_between_meals.map(food_intake_between_meals_val)

#smoking
smoking_val = {1: 'yes',
               2: 'no'}
df.smoking =df.smoking.map(smoking_val)

#liquid_intake_daily
liquid_intake_daily_val = {1: '<1L',
                           2: '1-2L',
                           3: '2L+'}
df.liquid_intake_daily = df.liquid_intake_daily.map(liquid_intake_daily_val)

#calculation_of_calorie_intake
calculation_of_calorie_intake_val = {1: 'yes',
                                     2: 'no'}
df.calculation_of_calorie_intake = df.calculation_of_calorie_intake.map(calculation_of_calorie_intake_val)

#physical_exercise
physical_exercise_val = {1: 'no',
                          2: '1-2days',
                          3: '3-4days',
                          4: '5-6days',
                          5: '6days+'}
df.physical_exercise = df.physical_exercise.map(physical_exercise_val)

#schedule_dedicated_to_technology
schedule_dedicated_to_technology_val = {1: '0-2hrs',
                                        2: '3-5hrs',
                                        3: '5hrs+'}
df.schedule_dedicated_to_technology = df.schedule_dedicated_to_technology.map(schedule_dedicated_to_technology_val)

#type_of_transportation_used
type_of_transportation_used_val = {1: 'automobile',
                                   2: 'motorbike',
                                   3: 'bike',
                                   4: 'public_transport',
                                   5: 'walking'}
df.type_of_transportation_used = df.type_of_transportation_used.map(type_of_transportation_used_val)



# training


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 3)

df_full_train = df_full_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 3)

df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)

y_train = df_train['weight_class'].astype(int).values
y_val = df_val['weight_class'].astype(int).values

del df_train['weight_class']
del df_val['weight_class']


def train(df, y):
    
    cat = df.to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)
    
    model = RandomForestClassifier(random_state=3)
    model.fit(X,y)
    
    return dv, model

def predict(df, dv, model):
    
    cat = df.to_dict(orient = 'records')
    
    X = dv.transform(cat)
    
    y_pred = model.predict(X)
    
    return y_pred


# training final model and testing
y_test = df_test.weight_class.values

dv, model = train(df_train, y_train)
y_pred = predict(df_test, dv, model)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print('Accuracy score = %.3f' % accuracy)
print(f'F1 score = {f1:.3f}')






# ## Saving the model

output_file = 'weight-class-model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')





