# Contributing to Resonant Models

Thank you for your interest in contributing to the Resonant Models project! This document provides guidelines for contributing to this groundbreaking research.

## Project Vision

Resonant Models aims to establish **phase-based neural architectures** as a viable alternative to attention mechanisms. Our goal is to create models that:

1. Scale to infinite context (O(d) memory complexity)
2. Ground computation in physical principles (phase dynamics, Kuramoto synchronization)
3. Achieve practical performance (currently 10.5% needle-in-haystack retrieval)

## Ways to Contribute

### 1. Reproducing Results
- Run the benchmark commands from USAGE.md
- Report your results (success or failure)
- Document any hardware/software differences

### 2. Improving Performance
- Experiment with hyperparameters
- Try novel configurations (new omega ranges, cascade depths, etc.)
- Share results that exceed 10.5% benchmark

### 3. Architectural Extensions
- Implement multi-resolution PVM
- Add hybrid resonance + sparse attention
- Create new phase-based mechanisms

### 4. Documentation
- Clarify confusing sections
- Add tutorials or examples
- Improve mathematical explanations

### 5. Bug Fixes
- Report issues with reproducibility
- Fix edge cases
- Improve numerical stability

## Contribution Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings to all public methods
- Keep commits focused and atomic

### Testing
- Ensure reproducibility (fix random seeds)
- Provide baseline comparisons
- Document hardware requirements
- Include example commands

### Documentation
- Explain WHY, not just WHAT
- Provide mathematical intuition where relevant
- Use clear, professional English
- Cite relevant research

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Write clear, documented code
   - Add tests if applicable
   - Update documentation
4. **Test thoroughly**:
   - Run existing tests: `pytest tests/`
   - Verify reproduction commands still work
5. **Commit with clear messages**:
   ```
   Add multi-resolution PVM for hierarchical memory

   - Implements separate memory states for different time scales
   - Adds omega_pyramid parameter for frequency hierarchy
   - Achieves 12.3% needle retrieval (+1.8% over baseline)
   ```
6. **Push and create PR**:
   - Provide clear description of changes
   - Include performance numbers (before/after)
   - Reference related issues

### Review Criteria

PRs will be evaluated on:
- **Scientific rigor**: Are claims backed by experiments?
- **Reproducibility**: Can others replicate your results?
- **Code quality**: Is it clean, documented, tested?
- **Performance**: Does it improve over baselines?

## Research Ethics

### Transparency
- Report negative results (what didn't work)
- Avoid cherry-picking best runs
- Acknowledge limitations honestly

### Attribution
- Cite prior work that inspired your contribution
- Credit collaborators appropriately
- Respect intellectual property

### Open Science
- Share experimental results openly
- Provide code for reproducibility
- Make data/models available when possible

## Communication

### GitHub Issues
- Use for bug reports, feature requests, questions
- Provide minimal reproducible examples
- Tag appropriately (bug, enhancement, question, etc.)

### Discussions
- Use for open-ended topics
- Share experimental results
- Discuss theoretical questions

### Code of Conduct
- Be respectful and professional
- Focus on ideas, not people
- Welcome diverse perspectives
- Help newcomers learn

## Development Setup

```bash
# Clone repository
git clone https://github.com/Freeky7819/attention-free-phase-blocks.git
cd attention-free-phase-blocks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/

# Check style
flake8 src/
```

## Experimental Protocol

When proposing new features, follow this protocol:

1. **Hypothesis**: What improvement do you expect and why?
2. **Baseline**: Run control experiment (existing method)
3. **Experimental**: Run with your modification
4. **Comparison**: Report metrics side-by-side
5. **Analysis**: Explain results (success or failure)

Example:
```markdown
## Experiment: Multi-Resolution PVM

**Hypothesis**: Separate memories for different time scales will improve
long-range retrieval by 2-5%.

**Baseline**: 10.5% needle retrieval (single PVM)

**Experimental**: 12.3% needle retrieval (3-level pyramid)

**Configuration**:
- Level 1: omega=6.0 (coarse, 100-token window)
- Level 2: omega=8.0 (medium, 50-token window)
- Level 3: omega=10.0 (fine, 25-token window)

**Analysis**: Improvement concentrated in long-range needles (>256 tokens away).
Short-range performance unchanged. Cost: +15% memory, +8% compute.
```

## Roadmap

Current priorities:
1. **Breaking 10.5% ceiling**: Reaching 15-20% needle retrieval
2. **Scaling to 1M+ context**: Testing extreme long-range scenarios
3. **Hybrid architectures**: Combining resonance with sparse attention
4. **Real-world tasks**: Beyond synthetic benchmarks

See [ROADMAP.md](ROADMAP.md) for detailed plans.

## Questions?

- Check [FAQ](docs/FAQ.md) first
- Search existing issues/discussions
- Ask in [Discussions](https://github.com/Freeky7819/attention-free-phase-blocks/discussions)
- Email: [contact email if provided]

---

**Thank you for helping build the future of neural architectures!** 🌊

Your contributions advance not just this project, but the entire field's understanding
of alternatives to attention mechanisms. Every experiment, bug fix, and insight matters.
