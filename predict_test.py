#!/usr/bin/env python
# coding: utf-8



# get_ipython().run_line_magic('autosave', '0')



import json
import requests

url = 'http://localhost:8080/predict'


person = {
    "sex": "male",
    "age": 37,
    "height": 184,
    "overweight_obese_family": "yes",
    "consumption_of_fast_food": "yes",
    "frequency_of_consuming_vegetables": "rarely",
    "number_of_main_meals_daily": 2,
    "food_intake_between_meals": "usually",
    "smoking": "yes",
    "liquid_intake_daily": "1-2L",
    "calculation_of_calorie_intake": "yes",
    "physical_exercise": "5-6days",
    "schedule_dedicated_to_technology": "0-2hrs",
    "type_of_transportation_used": "automobile"
}

result = requests.post(url, json=person).json()

print(json.dumps(result, indent=2))





