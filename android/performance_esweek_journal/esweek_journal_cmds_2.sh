# adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/MobileNet_for_mobile0.5.tflite --num_threads=1
# adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/MobileNet_for_mobile0.625.tflite --num_threads=1
# adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/MobileNet_for_mobile0.75.tflite --num_threads=1
# adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/MobileNet_for_mobile0.875.tflite --num_threads=1
# adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/MobileNet_for_mobile1.0.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG19_for_mobile0.5.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG19_for_mobile0.625.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG19_for_mobile0.75.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG19_for_mobile0.875.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG19_for_mobile1.0.tflite --num_threads=1
