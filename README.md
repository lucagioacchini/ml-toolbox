# ML-Toolbox
___

This repo contains `mltoolbox`, a comprehensive Python library for machine learning tasks, focusing on representation learning, clustering, and classification with support for preprocessing and evaluation metrics.

## 🔧 Features and Components

### Representation Learning
- **iWord2Vec**: Incremental Word2Vec implementation supporting vocabulary updates and embedding management
- **MultimodalAutoencoder**: Neural network for learning joint representations from multiple data modalities

### Clustering
- **kGMA**: K-nearest neighbors graph-based modularity analysis
- **kMeans**: Enhanced k-means implementation with data standardization and model persistence

### Classification
- **KnnClassifier**: K-nearest neighbors classifier with probability estimation
- **DeepClassifier**: Customizable deep learning classifier with preprocessing and model saving

### Preprocessing
- **StratifiedKFold**: Cross-validation maintaining class distribution
- **OneHotLabelEncoder**: Label encoding with class weight computation

### Metrics
- **knn_probability**: Probability estimation based on k-nearest neighbors
- **silhouette**: Implementation of silhouette score for cluster evaluation

## ⚙️ Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the package:
```bash
pip3 install -e .
```

## 📚 Documentation

Each module includes comprehensive docstrings with:
- Detailed descriptions of functionality
- Parameter specifications
- Usage examples
- Implementation notes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Luca Gioacchini

## 📧 Contact

For questions and feedback, please open an issue on GitHub.