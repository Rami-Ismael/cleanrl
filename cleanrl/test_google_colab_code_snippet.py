## A method to check if my code is runnning on google colab
import sys
def on_colab() -> bool:
    if 'google.colab' in sys.modules:
        return True
    else:
        return False
print( "This code is running on google colab" if on_colab() else "This code is not running on google colab")