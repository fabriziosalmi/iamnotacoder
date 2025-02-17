import math
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers."""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers. Raises ValueError if divisor is zero.
    
    Args:
        a (float): The dividend.
        b (float): The divisor.
        
    Returns:
        float: The quotient.
        
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        logging.error("Cannot divide by zero.")
        raise ValueError("Cannot divide by zero.")
    return a / b


def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer using recursion.
    
    Args:
        n (int): The number to calculate the factorial for.
        
    Returns:
        int: The factorial of n.
    """
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

    # Avoid unnecessary global variable usage
    result = add(x, y)

    # Optimize the loop to avoid repeated calculations
    sqrt_results = [math.sqrt(i) for i in range(10)]
    for i, sqrt_val in enumerate(sqrt_results):
        print(f"Square root of {i} is {sqrt_val:.2f}")


if __name__ == "__main__":
    main()