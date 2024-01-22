pip install -i https://test.pypi.org/simple/ kypackage==5.0.1

from kypackage.first_script import first_function
from kypackage.second_script import second_function
from kypackage.math.add_script import add_function
from kypackage.math.multiply_script import multiply_function
from kypackage.quantity.quantity_script import quantity_function

first_function()

second_function()

add_function(1,2)

multiply_function(1,2)

quantity_function(1)
  
  
