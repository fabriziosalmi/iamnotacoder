import math


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The difference between the two numbers.
    """
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The product of the two numbers.
    """
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers. Does not handle division by zero.

    Args:
        a (float): The numerator.
        b (float): The denominator.

    Returns:
        float: The quotient of the two numbers.

    Raises:
        ValueError: If the denominator is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def factorial(n: int) -> int:
    """Calculate the factorial of a number using recursion.

    Args:
        n (int): The number to calculate the factorial for.

    Returns:
        int: The factorial of the number.

    Raises:
        ValueError: If the number is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def main():
    """Main function to demonstrate basic operations."""
    # Some hardcoded values for testing
    x = 10
    y = 5

    print("Addition:", add(x, y))
    print("Subtraction:", subtract(x, y))
    print("Multiplication:", multiply(x, y))
    print("Division:", divide(x, y))
    print("Factorial of 5:", factorial(5))

    # Avoid global variables
    result = add(x, y)

    # Optimize loop
    for i in range(10):
        if i >= len(math.sqrt.__code__.co_varnames):
            break
        print("Square root of", i, "is", math.sqrt(i))


if __name__ == "__main__":
    main()