import degirum as dg
import sys
import time

device_type = "HAILORT/HAILO8L"

inference_host_address = "@local"
zoo_url = "degirum/hailo"
token = ""
model_name = "yolov8n_coco--640x640_quant_hailort_multidevice_1"
image_source = "../hailo_examples/assets/ThreePersons.jpg"

try:
    model = dg.load_model(
        model_name=model_name,
        inference_host_address=inference_host_address,
        zoo_url=zoo_url,
        token=token,
        device_type=device_type,)
except Exception as e:
    print(f"Error loading model '{model_name}': {e}")
    sys.exit(1)

try:
    inference_result = model(image_source)
    print(inference_result)
except Exception as e:
    print(f"Error during inference: {e}")

count = 0
target = 100
start_time = time.time()
while count < target:
    inference_result = model(image_source)
    count = count + 1
end_time = time.time()
delta_time = end_time - start_time
print(f"Did {target} inferences in {delta_time} seconds")
