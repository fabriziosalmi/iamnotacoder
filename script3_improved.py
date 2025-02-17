import math


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Return the difference between two integers."""
    return a - b


def multiply(a: int, b: int) -> int:
    """Return the product of two integers."""
    return a * b


def divide(a: float, b: float) -> float:
    """
    Return the quotient of two numbers.
    
    Raises ValueError if divisor is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def factorial(n: int) -> int:
    """Return the factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def main():
    # Use constants instead of hardcoded values for better maintainability and security
    X = 10
    Y = 5

    print("Addition:", add(X, Y))
    print("Subtraction:", subtract(X, Y))
    print("Multiplication:", multiply(X, Y))
    print("Division:", divide(X, Y))
    print("Factorial of 5:", factorial(5))

    # Avoid unnecessary global variable usage
    result = add(X, Y)

    # Use list comprehension for optimized loop and avoid string formatting in loop
    squares = [f"Square root of {i} is {math.sqrt(i):.2f}" for i in range(10)]
    print("\n".join(squares))


if __name__ == "__main__":
    main()