# Publishing SLiCE to PyPI

This document provides comprehensive instructions for publishing the SLiCE package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org/account/register/)
2. **Test PyPI Account**: Create an account at [test.pypi.org](https://test.pypi.org/account/register/) for testing
3. **GitHub Repository**: Ensure your code is pushed to GitHub
4. **Version Control**: Use semantic versioning (e.g., 0.1.0, 0.2.0, 1.0.0)

## Method 1: Automated Publishing (Recommended)

### Setup Trusted Publishing (Secure, No API Keys Required)

1. **Configure PyPI Trusted Publishing**:
   - Go to [PyPI Publishing Settings](https://pypi.org/manage/account/publishing/)
   - Add a new trusted publisher with:
     - PyPI project name: `slice-score`
     - Owner: `yourusername`
     - Repository name: `SLiCE`
     - Workflow filename: `publish.yml`
     - Environment name: `release`

2. **Create a Release**:
   ```bash
   # Update version in pyproject.toml first
   git add .
   git commit -m "Bump version to v0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```

3. **GitHub will automatically**:
   - Run tests via CI workflow
   - Build the package
   - Publish to PyPI when a release is created

### Manual Release Creation

Alternatively, create releases through GitHub UI:
1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Choose your tag (e.g., `v0.1.0`)
4. Add release notes
5. Publish release

## Method 2: Manual Publishing

### Initial Setup

```bash
# Using pip
pip install build twine
pip install -e ".[dev]"

# Using uv (recommended - faster)
uv sync --extra dev
```

### Testing Before Publishing

```bash
# Using pip
pytest tests/ -v
black --check slice tests examples
isort --check-only slice tests examples
flake8 slice
slice-eval --help

# Using uv (recommended)
uv run pytest tests/ -v
uv run black --check slice tests examples
uv run isort --check-only slice tests examples  
uv run flake8 slice
uv run slice-eval --help
```

### Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package
python -m build

# Verify the build
twine check dist/*
```

### Test on Test PyPI (Recommended)

```bash
# Upload to Test PyPI first
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ slice-score

# Test the installed package
python -c "import slice; print('Import successful')"
slice-eval --version
```

### Publish to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Verify on PyPI
pip install slice-score
```

## Version Management

### Semantic Versioning

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Update Version

1. **Update pyproject.toml**:
   ```toml
   [project]
   version = "0.2.0"
   ```

2. **Update slice/__init__.py**:
   ```python
   __version__ = "0.2.0"
   ```

3. **Commit changes**:
   ```bash
   git add pyproject.toml slice/__init__.py
   git commit -m "Bump version to v0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```

## API Token Setup (Alternative to Trusted Publishing)

If you prefer using API tokens instead of trusted publishing:

### Create API Token

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/token/)
2. Create a new API token with scope limited to your project
3. Copy the token (starts with `pypi-`)

### Configure GitHub Secrets

1. Go to your GitHub repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Add repository secret:
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token

### Update GitHub Actions

Modify `.github/workflows/publish.yml`:

```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}
```

## Troubleshooting

### Common Issues

1. **Version already exists**:
   - Update version number in `pyproject.toml`
   - Create new git tag
   - Rebuild and republish

2. **Import errors after installation**:
   - Check package structure
   - Verify `__init__.py` exports
   - Test local installation: `pip install -e .`

3. **Missing files in package**:
   - Check `MANIFEST.in`
   - Verify `pyproject.toml` includes
   - Test with `python -m build` and inspect `dist/`

4. **CLI not working**:
   - Verify entry point in `pyproject.toml`
   - Check `slice/cli.py` exists and has `main()` function
   - Test: `python -m slice.cli --help`

### Testing Checklist

Before each release:

- [ ] All tests pass (`pytest tests/`)
- [ ] Code quality checks pass (black, isort, flake8)
- [ ] CLI works (`slice-eval --help`)
- [ ] Examples run without errors
- [ ] Version numbers updated
- [ ] CHANGELOG.md updated (if you create one)
- [ ] Documentation is current

## Monitoring

### After Publishing

1. **Verify Installation**:
   ```bash
   pip install slice-score
   python -c "import slice; print('Success')"
   ```

2. **Check PyPI Page**: Visit [pypi.org/project/slice-score/](https://pypi.org/project/slice-score/)

3. **Monitor Downloads**: Use tools like [pypistats](https://pypistats.org/)

### Package Statistics

```bash
# Install pypistats
pip install pypistats

# View download statistics
pypistats recent slice-score
pypistats overall slice-score
```

## Security Best Practices

1. **Use Trusted Publishing**: Preferred over API tokens
2. **Limit Token Scope**: If using tokens, limit to specific project
3. **Regular Token Rotation**: Update tokens periodically
4. **Never Commit Secrets**: Use GitHub Secrets for tokens
5. **Review Dependencies**: Regularly audit package dependencies

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/guides/building-and-testing-python)
- [Trusted Publishing Guide](https://blog.pypi.org/posts/2023-04-20-introducing-trusted-publishers/)

## Support

If you encounter issues:

1. Check GitHub Issues
2. Review PyPI documentation
3. Test on Test PyPI first
4. Verify all configuration files

---

Remember to test thoroughly on Test PyPI before publishing to the main PyPI repository!