import math


def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and return the result."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers and return the result. Handle division by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def factorial(n: int) -> int:
    """Calculate the factorial of a number using an iterative approach for better performance."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def main() -> None:
    """Main function to demonstrate basic operations."""
    x = 10
    y = 5

    print("Addition:", add(x, y))
    print("Subtraction:", subtract(x, y))
    print("Multiplication:", multiply(x, y))
    print("Division:", divide(x, y))
    print("Factorial of 5:", factorial(5))

    result = add(x, y)
    print("Result of addition:", result)

    for i in range(10):
        if i >= 0:
            print(f"Square root of {i} is {math.sqrt(i)}")