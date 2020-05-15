from imageai.Prediction import ImagePrediction
import cv2
import os

def my_prediction(img_path, prob):
    result = {}
    execution_path = os.getcwd()
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(execution_path, "./data/resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
    prediction.loadModel()

    predictions, probabilities = prediction.predictImage(os.path.join(execution_path, img_path), result_count=5)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        if( eachProbability >= prob ):
            result[eachPrediction] = eachProbability
            # print(eachPrediction , " : " , eachProbability)

    return result


def image_process():
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    cv2.imwrite('output.png', frame)


def verify_keyword(sample_word):
    code = '100'
    keywords = ['bottle', 'can', 'water', 'plastic', 'wine']
    gate1 = ['bottle', 'water', 'wine']
    gate2 = ['can']

    for sample in sample_word.keys():
        for keyword in keywords:
            if keyword in sample:
                if keyword in gate1:
                    code = '110'
                elif keyword in gate2:
                    code = '101'
                print('FOUND:', keyword, 'in', sample)

    return code


if __name__ == '__main__':
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
