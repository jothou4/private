# Example of integer format specifier %10d
# %10d prints an integer in a width of 10 characters, right-aligned.
# Example value: 123
print('%10d' % 123)
# Output: '       123'
# There are 7 spaces before the number '123' to make a total of 10 characters.

# Example of floating point format specifier %.4f
# %.4f prints a floating point number with up to 4 decimal places.
# Example value: 123.123
print('%.4f' % 123.123)
# Output: '123.1230'
# The original value 123.123 is shown with 4 decimal places as 123.1230.

# Example of floating point format specifier %10.4f
# %10.4f prints a floating point number in a width of 10 characters, with up to 4 decimal places.
# Example value: 123.123
print('%10.4f' % 123.123)
# Output: '  123.1230'
# There are 2 spaces before the number '123.1230' to make a total of 10 characters.