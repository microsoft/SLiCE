# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-01-XX

### 🎉 Initial Release
This is the first public release of SLiCE (Schema Lineage Composite Evaluation).

### Added
- **Core Evaluation Engine**: Schema lineage evaluation with component-wise scoring
- **Multiple Similarity Metrics**: BLEU, weighted BLEU, fuzzy matching, F1 scores, and AST-based similarity
- **Multi-language Support**: Python, SQL, and C# code analysis
- **Flexible Configuration**: Customizable weights for components and metrics
- **Sample Data Module** (`slice.data`):
  - Access to curated sample datasets
  - Pipeline script retrieval
  - Gold and generated lineage data loading
  - Batch evaluation preparation with schema matching
- **Command Line Interface**: Easy-to-use CLI for quick evaluations
- **Batch Processing**: Parallel evaluation of multiple lineage pairs
- **Comprehensive Examples**:
  - Basic usage demonstration
  - Custom weights configuration
  - Sample data usage and batch evaluation
- **Robust Testing**: 94 comprehensive tests covering all functionality

### Features
- Component-wise evaluation (source schema, tables, transformations, aggregations)
- Support for metadata evaluation (optional)
- Fuzzy string matching for robustness
- AST-based code similarity analysis
- Format validation and error handling
- Parallel batch processing for performance

### Documentation
- Complete API documentation
- Usage examples and tutorials
- Installation and setup guides
- Contributing guidelines

### Dependencies
- Python 3.9+
- Core: fuzzywuzzy, numpy, pandas
- Development: pytest, coverage tools
- Optional: multiprocessing for parallel evaluation