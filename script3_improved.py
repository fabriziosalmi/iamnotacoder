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
    
    Raises ValueError if the divisor is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def factorial(n: int) -> int:
    """Return the factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result


def main():
    """Main function to demonstrate various operations."""
    x = 10
    y = 5

    print("Addition:", add(x, y))
    print("Subtraction:", subtract(x, y))
    print("Multiplication:", multiply(x, y))
    print("Division:", divide(x, y))
    print("Factorial of 5:", factorial(5))

    # Use list comprehension for optimized loop
    results = [math.sqrt(i) for i in range(10)]
    for i, sqrt_value in enumerate(results):
        print(f"Square root of {i} is {sqrt_value}")


if __name__ == "__main__":
    main()