# Definition of the Calculator class
class Calculator:
    def __init__(self):
        # Initialization method: Automatically called when an object is created to set its initial state.
        self.result = 0  # Initialize the instance variable result to 0.

    def add(self, num):
        # add method: Adds the given number (num) to the current result and returns the result.
        self.result += num  # Add num to the current result.
        return self.result  # Return the current result.

# Creating two instances using the Calculator class
cal1 = Calculator()  # Create cal1 object
cal2 = Calculator()  # Create cal2 object

# Printing the type of each object
print(type(cal1))  # <class '__main__.Calculator'>: Type of cal1 is the Calculator class.
print(type(cal2))  # <class '__main__.Calculator'>: Type of cal2 is the Calculator class.
print(type(Calculator()))  # <class '__main__.Calculator'>: Printing the type of a new Calculator object
print(type(Calculator))  # <class 'type'>: Printing the type of the Calculator class itself

# Calling the add method on each object and printing the results
print(cal1.add(3))  # 3: Print the result of adding 3 to cal1's result
print(cal1.add(4))  # 7: Print the result of adding 4 to cal1's current result
print(cal2.add(3))  # 3: Print the result of adding 3 to cal2's result
print(cal2.add(7))  # 10: Print the result of adding 7 to cal2's current result

'''
## Class Definition (class Calculator):
A class is a fundamental unit for managing related objects.
A class is a blueprint for creating objects.
Python is an object-oriented programming language.
Object: A bundled unit containing data and functions (methods).
Object Attribute: Data that stores the state of an object.
Object Behavior: Objects have methods that allow them to process their data and perform specific actions.

# Example: Defining a Person class representing a person
class Person:
    def __init__(self, name, age):
        self.name = name  # Name attribute
        self.age = age    # Age attribute

    def greet(self):
        return f"Hello, I'm {self.name} and I'm {self.age} years old."  # Method

# Creating objects (instances) based on the Person class
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

# Using object attributes and methods
print(person1.name)  # Prints "Alice"
print(person2.greet())  # Prints "Hello, I'm Bob and I'm 25 years old."


## Initialization Method (__init__):
The __init__ method is automatically called when a class is instantiated to initialize the object's initial state.
When an instance is created, memory is allocated for it.
During instantiation, the class's constructor responsible for initializing the object is called.
Once an instance is created, it occupies memory space where its data (instance variables) and methods are stored.
Here, self.result is initialized to 0 to set up an instance variable for storing calculation results.

## Object vs Instance:
Calculator() creates a new instance (object) of the Calculator class.
cal1 and cal2 are instances of the Calculator class, each having its own independent result value.

## When to Use Classes:
Classes are useful when you need to create multiple objects with similar characteristics.
For example, you can create a new object every time you need a calculator-like functionality.
'''