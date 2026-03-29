# Contributing to FLUXION

Thank you for your interest in contributing to FLUXION! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be considerate of others and follow standard open-source community guidelines.

---

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fluxion.git
   cd fluxion
   ```
3. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Prerequisites

- Python 3.8+
- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.15+
- Verilator 5.0+ (for Verilator extension)

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run Python tests
pytest tests/python/

# Run C++ tests (after build)
./build/fluxion_test
```

### Code Formatting

```bash
# Format Python code
black src/python/
isort src/python/

# Type checking
mypy src/python/
```

---

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title** describing the problem
2. **Steps to reproduce** the bug
3. **Expected behavior** vs **actual behavior**
4. **Environment details** (OS, Python version, etc.)
5. **Code snippets** or error messages if relevant

### Suggesting Features

Open an issue with:

1. **Clear title** for the feature
2. **Description** of what you want to achieve
3. **Use cases** and examples
4. **Possible implementation** ideas (optional)

### Submitting Code

1. **Fork** the repository
2. **Create a branch** for your changes
3. **Make your changes** following coding standards
4. **Add tests** for new functionality
5. **Submit a pull request**

---

## Pull Request Process

1. **Ensure all tests pass** before submitting
2. **Update documentation** if needed
3. **Add entry to CHANGELOG** (if applicable)
4. **Request review** from maintainers
5. **Address review feedback** promptly

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New code has appropriate tests
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow conventions

---

## Coding Standards

### Python

- Follow **PEP 8** style guide
- Use **type hints** for function signatures
- Write **docstrings** for classes and public functions
- Maximum line length: **100 characters**

```python
def calculate_force(particle: FluxionParticle, distance: float) -> float:
    """
    Calculate force on a particle.

    Args:
        particle: The particle to calculate force on
        distance: Distance to connected particle

    Returns:
        The calculated force magnitude
    """
    return self.weight * particle.mass / (distance ** 2)
```

### C++

- Follow **C++17** standard
- Use **snake_case** for functions and variables
- Use **PascalCase** for classes
- Add comments for complex logic

```cpp
double calculateForce(const FluxionNode& node, double distance) {
    // Calculate spring force using Hooke's law
    return weight_ * node.mass / (distance * distance);
}
```

---

## Commit Messages

Follow this format:

```
type(scope): brief description

Longer description if needed.

Fixes #issue_number
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

### Examples

```
feat(annealing): add adaptive temperature schedule

Implement adaptive temperature schedule that adjusts cooling
rate based on acceptance rate during annealing.

Fixes #42
```

```
fix(force-fields): correct wire tension calculation

The wire tension force was incorrectly using squared distance
instead of linear distance for Hooke's law calculation.
```

---

## Questions?

Feel free to open an issue for questions or discussions, or reach out to the maintainers.

Thank you for contributing to FLUXION!