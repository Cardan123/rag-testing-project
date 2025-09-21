# Python Programming Guide

## Introduction to Python

Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, Python emphasizes code readability and allows programmers to express concepts in fewer lines of code.

## Basic Syntax

### Variables and Data Types

```python
# Numbers
integer_var = 42
float_var = 3.14
complex_var = 1 + 2j

# Strings
string_var = "Hello, World!"
multiline_string = """This is a
multiline string"""

# Boolean
boolean_var = True

# None type
none_var = None
```

### Basic Operations

```python
# Arithmetic operators
a + b    # Addition
a - b    # Subtraction
a * b    # Multiplication
a / b    # Division
a // b   # Floor division
a % b    # Modulus
a ** b   # Exponentiation

# Comparison operators
a == b   # Equal
a != b   # Not equal
a < b    # Less than
a > b    # Greater than
a <= b   # Less than or equal
a >= b   # Greater than or equal

# Logical operators
and, or, not
```

## Data Structures

### Lists

```python
# Creating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# List operations
fruits.append("grape")        # Add element
fruits.insert(1, "kiwi")     # Insert at index
fruits.remove("banana")      # Remove element
fruits.pop()                 # Remove last element
fruits.sort()                # Sort in place
len(fruits)                  # Get length
```

### Tuples

```python
# Creating tuples (immutable)
coordinates = (10, 20)
colors = ("red", "green", "blue")

# Tuple unpacking
x, y = coordinates
```

### Dictionaries

```python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Dictionary operations
person["age"]              # Access value
person["job"] = "Engineer" # Add new key-value
del person["city"]         # Delete key
person.keys()              # Get all keys
person.values()            # Get all values
person.items()             # Get key-value pairs
```

### Sets

```python
# Creating sets (unique elements)
unique_numbers = {1, 2, 3, 4, 5}
fruits_set = {"apple", "banana", "orange"}

# Set operations
fruits_set.add("grape")         # Add element
fruits_set.remove("banana")     # Remove element
set1.union(set2)               # Union
set1.intersection(set2)        # Intersection
set1.difference(set2)          # Difference
```

## Control Flow

### Conditional Statements

```python
if condition:
    # Execute if condition is True
    pass
elif another_condition:
    # Execute if another_condition is True
    pass
else:
    # Execute if all conditions are False
    pass
```

### Loops

```python
# For loop
for item in iterable:
    print(item)

# While loop
while condition:
    # Execute while condition is True
    pass

# Loop control
break     # Exit loop
continue  # Skip to next iteration
```

### List Comprehensions

```python
# Basic syntax
new_list = [expression for item in iterable]

# Examples
squares = [x**2 for x in range(10)]
even_numbers = [x for x in range(20) if x % 2 == 0]
```

## Functions

### Defining Functions

```python
def function_name(parameters):
    """Optional docstring"""
    # Function body
    return value

# Example
def greet(name, greeting="Hello"):
    """Greet someone with a custom message"""
    return f"{greeting}, {name}!"

# Calling functions
message = greet("Alice")
custom_message = greet("Bob", "Hi")
```

### Advanced Function Features

```python
# *args and **kwargs
def flexible_function(*args, **kwargs):
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

# Lambda functions
square = lambda x: x**2
add = lambda x, y: x + y

# Higher-order functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

## Object-Oriented Programming

### Classes and Objects

```python
class Person:
    # Class variable
    species = "Homo sapiens"

    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age

    def introduce(self):
        """Instance method"""
        return f"Hi, I'm {self.name} and I'm {self.age} years old"

    @classmethod
    def from_birth_year(cls, name, birth_year):
        """Class method"""
        current_year = 2024
        age = current_year - birth_year
        return cls(name, age)

    @staticmethod
    def is_adult(age):
        """Static method"""
        return age >= 18

# Creating objects
person1 = Person("Alice", 25)
person2 = Person.from_birth_year("Bob", 1995)
```

### Inheritance

```python
class Employee(Person):
    def __init__(self, name, age, job_title):
        super().__init__(name, age)
        self.job_title = job_title

    def introduce(self):
        # Method overriding
        base_intro = super().introduce()
        return f"{base_intro} and I work as a {self.job_title}"

# Multiple inheritance
class Mixins:
    def some_method(self):
        pass

class AdvancedEmployee(Employee, Mixins):
    pass
```

## Error Handling

### Try-Except Blocks

```python
try:
    # Code that might raise an exception
    result = 10 / 0
except ZeroDivisionError:
    # Handle specific exception
    print("Cannot divide by zero!")
except Exception as e:
    # Handle any other exception
    print(f"An error occurred: {e}")
else:
    # Execute if no exception occurred
    print("Success!")
finally:
    # Always execute
    print("Cleanup code here")
```

### Custom Exceptions

```python
class CustomError(Exception):
    """Custom exception class"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise CustomError("Age cannot be negative")
    return age
```

## File I/O

### Reading and Writing Files

```python
# Reading files
with open("filename.txt", "r") as file:
    content = file.read()           # Read entire file
    lines = file.readlines()       # Read all lines
    line = file.readline()         # Read one line

# Writing files
with open("output.txt", "w") as file:
    file.write("Hello, World!")
    file.writelines(["Line 1\n", "Line 2\n"])

# Working with JSON
import json

# Read JSON
with open("data.json", "r") as file:
    data = json.load(file)

# Write JSON
with open("output.json", "w") as file:
    json.dump(data, file, indent=2)
```

## Modules and Packages

### Importing Modules

```python
# Different import styles
import math
from math import sqrt, pi
from math import *
import numpy as np

# Using imported modules
result = math.sqrt(16)
result = sqrt(16)
result = np.array([1, 2, 3])
```

### Creating Modules

```python
# mymodule.py
def my_function():
    return "Hello from my module!"

PI = 3.14159

# Using the module
# main.py
import mymodule
print(mymodule.my_function())
print(mymodule.PI)
```

## Popular Libraries

### NumPy (Numerical Computing)

```python
import numpy as np

# Arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Operations
arr * 2              # Element-wise multiplication
np.mean(arr)         # Calculate mean
np.sum(arr)          # Calculate sum
arr.reshape(5, 1)    # Reshape array
```

### Pandas (Data Analysis)

```python
import pandas as pd

# DataFrames
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'Chicago']
})

# Operations
df.head()            # First 5 rows
df.describe()        # Statistical summary
df['age'].mean()     # Column operations
df[df['age'] > 30]   # Filtering
```

### Matplotlib (Plotting)

```python
import matplotlib.pyplot as plt

# Basic plotting
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Plot')
plt.show()
```

## Best Practices

### Code Style (PEP 8)

```python
# Good naming conventions
user_name = "alice"          # Snake case for variables
MAX_SIZE = 100              # Constants in uppercase
class UserProfile:          # CamelCase for classes
    pass

def calculate_average():    # Snake case for functions
    pass

# Good spacing
result = x + y              # Spaces around operators
my_list = [1, 2, 3]        # Spaces after commas
```

### Documentation

```python
def calculate_area(length, width):
    """
    Calculate the area of a rectangle.

    Args:
        length (float): The length of the rectangle
        width (float): The width of the rectangle

    Returns:
        float: The area of the rectangle

    Raises:
        ValueError: If length or width is negative
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width must be positive")
    return length * width
```

### Testing

```python
import unittest

class TestMathFunctions(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 2, 4)

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            10 / 0

if __name__ == '__main__':
    unittest.main()
```

## Advanced Topics

### Decorators

```python
def timer_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer_decorator
def slow_function():
    time.sleep(1)
    return "Done"
```

### Context Managers

```python
class MyContext:
    def __enter__(self):
        print("Entering context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context")

# Using context manager
with MyContext() as ctx:
    print("Inside context")
```

### Generators

```python
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Using generator
fib = fibonacci_generator()
for _ in range(10):
    print(next(fib))
```

This guide covers the fundamental and advanced concepts of Python programming, providing a solid foundation for developing applications and understanding more complex topics in data science and machine learning.