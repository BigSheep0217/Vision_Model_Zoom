import numpy as np
import cv2
import onnxruntime


def init_onnx_session(model_path):
    providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
    })
    ]
    session = onnxruntime.InferenceSession(model_path, providers=providers) 
    # providers = [
    # 'CPUExecutionProvider'
    # ]
    # session = onnxruntime.InferenceSession(model_path, providers=providers)
    session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def session_run_plate(one_session, output_name, input_name, img):
    data = one_session.run([output_name], input_feed={input_name: [img]})[0][0]
    return data

def get_numpy_img_from_image(image, shape):
    img = image
    ori_frame = img.copy()
    img = cv2.resize(img, shape)
    return ori_frame, img

if __name__ == "__main__":
    session_plate, input_name_plate, output_name_plate = init_onnx_session("checkpoints/mask.onnx")
    plate = cv2.imread("test_images/1.jpg")
    ori_image, plate = get_numpy_img_from_image(plate, shape=(224, 56))
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
    plate = np.array(plate)
    plate = plate.transpose(2, 0, 1) / 255.0
    number_str = session_run_plate(session_plate, output_name_plate, input_name_plate, plate)
    print(number_str)


