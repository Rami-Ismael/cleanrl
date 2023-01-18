import logging
## A method to check if my code is runnning on google colab
logging.basicConfig(filename="tests.log", level=logging.NOTSET,
                    filemode='w',
                    format='%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s')
import sys
def on_colab() -> bool:
    if 'google.colab' in sys.modules:
        return True
    else:
        return False
print(sys.modules)
logging.info(sys.modules)
print( "This code is running on google colab" if on_colab() else "This code is not running on google colab")
logging.info("This code is running on google colab" if on_colab() else "This code is not running on google colab")