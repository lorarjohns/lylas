# LYLAS Lab on Pay Parity Hackathon



## Setup

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install -r requirements.txt
```
set environment variables for your API key, username, and password to use the [Socrata API](https://dev.socrata.com). You'll need an [NYC Open Data](https://opendata.cityofnewyork.us) API key.

One way to set environment variables is with an `.envrc` file in your project directory. You can activate it with [direnv](https://direnv.net).

## Features

 - Topic modeling using Latent Dirichlet Allocation, SVD, t-SNE
 - Visualizations with pyLDAvis
 - Fuzzy matching with Cythonized implementation of sparse top-n similarity
 - NLP with gensim, sklearn, and spaCy

 ** Highly unfinished! ** 
 TODO:
 - implement search and retrieval interface
 - expand dataset
 - tests
 - really, everything!
 
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.