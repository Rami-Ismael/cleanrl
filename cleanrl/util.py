## A method to check if my code is runnning on google colab
import sys
def on_colab() -> bool:
    if 'google.colab' in sys.modules:
        return True
    else:
        return False
    