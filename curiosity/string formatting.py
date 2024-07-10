# Examples of string formatting methods

# Using % operator for string formatting
a = "나의 이름은 %s입니다. 나이는 %d입니다." % ("이요셉", 20)
# Performs string formatting using the % operator.
# %s replaces a string and %d replaces an integer.

# Using format() method for string formatting
b = "나의 이름은 {0}입니다. 나이는 {1}입니다.".format("이요셉", 20)
# {0} and {1} are replaced by the arguments of the format() method in order.

# Using f-string formatting with a dictionary
dict = {'name': '이요셉', 'age': 20}
c = f"나의 이름은 {dict['name']}입니다. 나이는 {dict['age']}입니다."

# Simple f-string example for defining a string
d = f'{"끝"}'

# Print each result
print(a)  # "나의 이름은 이요셉입니다. 나이는 20입니다."
print(b)  # "나의 이름은 이요셉입니다. 나이는 20입니다."
print(c)  # "나의 이름은 이요셉입니다. 나이는 20입니다."
print(d)  # "끝"