adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.5.tflite /data/local/tmp
adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.625.tflite /data/local/tmp
adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.75.tflite /data/local/tmp
adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.875.tflite /data/local/tmp
adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG191.0.tflite /data/local/tmp
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG190.5.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG190.625.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG190.75.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG190.875.tflite --num_threads=1
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/VGG191.0.tflite --num_threads=1


adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile0.5.tflite /data/local/tmp
adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile0.625.tflite /data/local/tmp
adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile0.75.tflite /data/local/tmp
adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile0.875.tflite /data/local/tmp
adb push /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile1.0.tflite /data/local/tmp


adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/MobileNet_for_mobile0.5.tflite --num_threads=1
