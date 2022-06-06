import torch as tf


def CudaIsAvailable():
    print(f'1.version of cuda: {tf.__version__}'
          f'\n2.Is cuda available；{tf.cuda.is_available()}')
