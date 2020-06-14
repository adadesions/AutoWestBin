from imageai.Prediction import ImagePrediction
import cv2
import os
import time
import RPi.GPIO as gpio

def my_prediction(img_path, prob):
    result = {}
    execution_path = os.getcwd()
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(execution_path, "./data/resnet50.h5")) # I rename a model to simple name
    prediction.loadModel()

    predictions, probabilities = prediction.predictImage(os.path.join(execution_path, img_path), result_count=5)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        if( eachProbability >= prob ):
            result[eachPrediction] = eachProbability
            # print(eachPrediction , " : " , eachProbability)

    return result


def image_process():
    # The device number is cahangable
    cap = cv2.VideoCapture(0)

    while True:
        # Read Video capture in realtime
        ret, frame = cap.read()
        cv2.imshow("RPI CAM", frame)

        # Get Signal from pin11
        # You can change the pin number here!
        pir = gpio.input(11)

        # If PIR sensor detected movement
        if pir == 1:
            cv2.imwrite('output.png', frame)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


def verify_keyword(sample_word):
    code = '100'
    keywords = ['bottle', 'can', 'water', 'plastic', 'wine']
    gate1 = ['bottle', 'water', 'wine']
    gate2 = ['can']

    for sample in sample_word.keys():
        for keyword in keywords:
            if keyword in sample:
                if keyword in gate1:
                    # Send signal to MOTOR here!
                    # You can add code overhere
                    code = '110'
                elif keyword in gate2:
                    # Optional
                    # For sending signal to another MOTOR
                    code = '101'
                print('FOUND:', keyword, 'in', sample)

    return code


if __name__ == '__main__':
    gpio.setwarnings(False)
    gpio.setmode(gpio.BOARD)
    gpio.setup(11, gpio.IN)
    print("AI is processing...")

    # Capture an image and save to disk
    image_process()

    # Send sample frame from cam to my_prediction function
    result = my_prediction('./output.png', 1)

    # Display result from my_prediction function
    print(result)

    # Find keyword in result
    code = verify_keyword(result)
    print("code to sensors:", code)
