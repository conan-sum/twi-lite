# twi-lite

## Description
**twi-lite** is a Python package that aims to make analyzing Twitter user data more convenient. It provides a template to create customizable data pipelines that allows full control from feature selection to model evaluation. It also provides a simple database interface for storing and extracting embeddings and clustering.

## Getting Started

### Prerequisites
1. Clone/download the repo
2. Create the file ```twi-lite/creds.json``` for your MySQL credentials with the following format:
```json
{
    "host" : "host.com",
    "user" : "username",
    "passwd" : "password"
}
```

### Installation
1. Copy ```/twilite``` and ```creds.json``` to your project directory

## Dependencies
* ```NumPy```
* ```Pandas```
* ```SciPy```
* ```Scikit-learn```
* ```UMAP learn```

## Documentation
[See wiki for documentation.](https://github.com/conan-sum/twi-lite/wiki)