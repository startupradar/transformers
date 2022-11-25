# StartupRadar Transformers

This python package allows you to integrate the [StartupRadar API](https://api.startupradar.co/docs) 
directly into your own Data or Machine Learning pipelines.
With only a list of domains, you can create huge Pandas DataFrames 
filled with all the data available on [StartupRadar](https://startupradar.co).


## Implemented transformers

### startupradar.transformers.core
All transformers in this module require API access.

- `LinkTransformer`: Create columns for all the domains a given domain links to.
- `BacklinkTransformer`: Create columns for all the domains that link to the given domain.
- `DomainTextTransformer`: Create a text column with the homepage text of the given domain.

### startupradar.transformers.pandas
Transformers that also output Pandas DataFrames, can be used by anyone, no API key necessary.

- `FeatureUnionDF`: Create a FeatureUnion with pd.DataFrames as input and output
- `PipelineDF`: Creates a pipeline that retains DataFrames and their column names
- `TfidfVectorizerDF`: Adaption of the sklearn transformer
- `CountVectorizerDF`: Adaption of the sklearn transformer

### startupradar.transformers.util
The transformers in this module also don't require the API and can be used by anyone.

- `ColumnPrefixTransformer`: Create a DataFrame with the same column names, but prefixed with e.g. `prefix_`
- `DomainNameTransformer`: Extract features from a domain name, currently only top level domain, e.g. `com` or `io`
- `CommonStringTransformer`: Application of a `CountVectorizer` to find common strings among passed inputs

### Upcoming
Transformers we're thinking about that may be coming soon:
 
- something to leverage the similar domains endpoint
- tfidf of all backlinks or (forward) links combined (domain- or url-level)

## How it works
For most transformers, you can simply pass a series of domain names as input.
In the case of the DomainNameTransformer, it could look like this:

```shell
> import pandas as pd
> from startupradar.transformers.util import DomainNameTransformer
>
> domains = ["loreyventures.com", "startupradar.co", "karllorey.com"]
> domains_series = pd.Series(domains)
> t = DomainNameTransformer()
> t.fit_transform(domains_series)
                   tld
loreyventures.com  com
startupradar.co     co
karllorey.com      com

```