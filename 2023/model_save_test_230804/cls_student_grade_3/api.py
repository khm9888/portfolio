
from tensorflow import keras
import data_reader
from fastapi import FastAPI
# import main
app = FastAPI()


values_sample = [170,50,1]
def data_preprocess(values):
    
    height = float(values[0]) / 194.2
    weight = float(values[1]) / 130.7
    
    return [height, weight,values[2]]
    

def return_school_label(label):
    shcools = ["초등학교","중학교","고등학교"]
    return shcools[label]

#모델 load

def load_model():
    model_load_path = "save_model/my_model.h5"
    new_model = keras.models.load_model(model_load_path)

    return new_model
new_model=load_model()


#http://127.0.0.1:8000/?height=150&weight=50&gender=1

@app.get("/")
async def root(height: int = 0, weight: int = 0 ,gender: int =0):
    
    values_sample = [height, weight, gender]
    # print("apitesting")
    processed_data=data_preprocess(values_sample)
    y_predict  = new_model.predict([processed_data])
    
    for predict_row in y_predict:
        # print(predict_row)
        predict_row_label = predict_row.argmax()
        print(predict_row)
        print(predict_row_label, end=" ") #0
        label = return_school_label(predict_row_label)
        confidence = predict_row[predict_row_label]
        print(label, confidence) #
    return {"label": str(label), "confidence":str(confidence)}