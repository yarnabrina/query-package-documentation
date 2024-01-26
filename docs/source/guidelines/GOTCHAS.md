# Known Issues

## Dataset Generation

1. It is only possible for packages which are installed in current environment.
2. It expects the package to be thoroughly documented in `numpydoc` style.

## Database Generation

1. It does not capture parent-child information between source documents and smaller chunks that are embedded to match smaller chunk for relevance and use full document as context.

## Information Retrieval

1. It is sometimes non-deterministic.
2. It is often quite slow, especially when `MMR` is used.

## Response Generation

1. It often gets stuck and takes long time to complete.
2. It sometimes fails completely with out of memory, but this is likely to vary based on device specifications.
