import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="GRU_Quantized_Model.tflite")
interpreter.allocate_tensors()

input_tensor_index = interpreter.get_input_details()[0]['index']
output_tensor_index = interpreter.get_output_details()[0]['index']

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([rh])

colors = [(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(245,117,16),(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(245,117,16),(245,117,16)]

sequence = []
sentence = []
predictions = []
threshold = 0.8
count=[]

count=[]

actions = np.array(["nothing", "a", "aa", "aw", "e", "ee", "u" ,"uw" , "o", "oo", "ow", "sac", "hoi", "huyen", "nang", "nga"])
actions_test = np.array(["nothing", "a", "aa", "aw", "e", "ee", "u" ,"uw" , "o", "oo", "ow", "sac", "hoi", "huyen", "nang", "nga"])
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        t = time.time()
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        count.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            sequence_array = np.expand_dims(sequence, axis=0).astype(np.float32)
            interpreter.set_tensor(input_tensor_index, sequence_array)
            interpreter.invoke()
            res = interpreter.get_tensor(output_tensor_index)

            print(actions_test[np.argmax(res)])
            predictions.append(np.argmax(res))
 
            if len(sentence) > 6: 
                sentence = sentence[-6:]
            
            cv2.putText(image, 'Action: {}'.format(actions[np.argmax(res)]), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)
        print('fps', 1/(time.time()-t))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()