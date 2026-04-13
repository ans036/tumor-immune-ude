# Contributing to Tumor-UDE

Thank you for your interest in contributing! This project welcomes contributions from the community.

## How to contribute

1. **Fork** the repository and create a new branch from `main`.
2. Make your changes, ensuring they follow the existing code style.
3. Add or update tests in `test/` if your changes affect functionality.
4. Run the test suite to verify nothing is broken:
   ```bash
   julia --color=yes test/runtests.jl
   julia --color=yes test/pure_mechanistic_tests.jl
   ```
5. Open a **Pull Request** against `main` with a clear description of your changes.

## Reporting issues

If you find a bug or have a feature request, please [open an issue](https://github.com/ans036/tumor-immune-ude/issues/new) with:

- A clear title and description
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Julia version and OS

## Seeking support

For questions about usage or the methodology, please [open a discussion](https://github.com/ans036/tumor-immune-ude/issues) or contact the author via the repository's issue tracker.

## Code style

- Follow standard Julia conventions (4-space indentation, descriptive function names).
- Keep functions focused and modular.
- Add comments for non-obvious logic.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
