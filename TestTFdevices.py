import tensorflow as tf
from tensorflow.python.client import device_lib
import time

# Number of power iterations. Increase for more powerful GPUs :-)
N = 5

# List devices and count them
devices = device_lib.list_local_devices()
n_devices = len(devices)
print(f"Found {n_devices} device(s).")


# Support function to really use tensors
def matpow(M, n):
    if n < 1:  # Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))


#
# Test all devices
results = []

for dev_num, dev in enumerate(devices):
    print(f"Running for device {dev.name} ({dev_num+1}/{n_devices})...")
    with tf.device(dev.name):
        before = time.time()
        c1 = []
        a = tf.Variable(tf.random.uniform(shape=(10000, 10000)), name="a")
        b = tf.Variable(tf.random.uniform(shape=(10000, 10000)), name="b")
        c1.append(matpow(a, N))
        c1.append(matpow(b, N))
        after = time.time()
        elapsed = after - before
        print(f"done in {elapsed:.2f} secs.")
        results.append(f"device name='{dev.name}', type={dev.device_type}, running time={elapsed:.2f} secs.")

print("Results:")
for res in results:
    print(res)

print("All done.")
