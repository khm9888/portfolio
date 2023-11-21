
from tensorflow import keras
import data_reader
import os
# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 2  # 예제 기본값은 20입니다.

use_checkpoint = True
use_early_stop = True

# checkpoint 저장 위치
checkpoint_path="training_1/cp.ckpt"
checkpoint_dir=os.path.dirname(checkpoint_path)


## checkpoint를 epoch를 반영할 수가 있음
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(3, activation='softmax')
    ])
    # 인공신경망을 컴파일합니다.
    model.compile(optimizer="adam", metrics=["accuracy"],loss="sparse_categorical_crossentropy")

    return model

model = create_model()

# # Display the model's architecture
# input_shape = (None, 3)
# model.build(input_shape)
# model.summary()


# # 인공신경망을 학습시킵니다.
# print("************ TRAINING START ************")
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# #checkpoint 사용
# cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# callback_list = []
# if use_checkpoint:
#     callback_list.append(cp_callback)
# if use_early_stop:
#     callback_list.append(early_stop)

# history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
#                     validation_data=(dr.test_X, dr.test_Y),
#                     callbacks=callback_list)

# ## 모델 hdf5 파일로 전제 저장 코드
# def model_whole_save_fn(trained_model):
#     model_save_path = "save_model/my_model.h5"
#     model_save_dir=os.path.dirname(model_save_path)
#     trained_model.save(model_save_path)

# model_whole_save_fn(model)

# os.listdir(checkpoint_dir)

# 모델 평가
untrained_model = create_model()
loss, acc = untrained_model.evaluate(dr.test_X, dr.test_Y, verbose=2)
# print("trained model, accuracy: {:5.2f}%".format(100 * acc))
print(f"Untrained model, accuracy: {100 * acc:5.2f}%")

trained_model = create_model()
trained_model.load_weights(checkpoint_path)
loss, acc = trained_model.evaluate(dr.test_X, dr.test_Y, verbose=2)
print(f"trained model, accuracy: {100 * acc:5.2f}%")

print(dr.test_X[:10])
print(type(dr.test_X[:10]))

# y_predict = trained_model.predict(dr.test_X)

# print(y_predict) #[[5.441674e-01 5.372047e-04 4.552954e-01]]

# labels = ["가위", "바위", "보"]

# for predict_row in y_predict[:10]:
#     print(predict_row)
#     print(predict_row.argmax(), end=" ") #0
#     label = labels[predict_row.argmax()]
#     confidence = predict_row[predict_row.argmax()]
#     print(label, confidence) #

# # 학습 결과를 그래프로 출력합니다.
# data_reader.draw_graph(history)
