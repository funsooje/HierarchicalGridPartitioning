# HierGP: Hierarchical Grid Partitioning for Spatiotemporal Data

HierGP is a novel dynamic global grid system that offers a flexible and fast approach for creating a customized global grid system. The system enables efficient processing of large spatiotemporal data points, such as phone location data, and is particularly beneficial for environmental analyses.

## Features

- **Flexible Grid System**: Customize the grid system to suit various levels of granularity.
- **Efficient Processing**: Handles large datasets with high efficiency.
- **Environmental Analysis**: Ideal for analyzing spatiotemporal data in environmental studies.

## Installation

To use HierGP, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/HierGP.git
cd HierGP
```

## Usage

### HierGP Class

The main class for HierGP is defined in hiergp.py. It provides methods for setting up and managing the hierarchical grid system.

### Example Usage

You can find usage examples in driver.py. Below is a brief example to get you started:

```python
from hiergp import hierGP
import pandas as pd

# Create some sample data
data = pd.DataFrame({
    'latitude': [34.05, 36.16, 40.71],
    'longitude': [-118.24, -115.15, -74.00]
})

# Initialize HierGP
hgp = HierGP(base_size = 25)

# Add samples to the grid
hgp.generateGrids(data, resolution=7)
```

### Utilities

The utils.py file contains various utility functions used in both hiergp.py and driver.py. These utilities include functions for data manipulation and plotting.

## Project Structure

- hiergp.py: Contains the main HierGP class and dataManager class.
- driver.py: Provides example usage of the HierGP system with sample datasets and plots.
- utils.py: Contains utility functions for data manipulation and visualization.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
