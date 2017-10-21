# Documentation

## Jazzy
Documentation generated with [Jazzy](https://github.com/realm/jazzy) application.
To prepare or update documenation

### Generate source structure

```
sourcekitten doc --spm-module CAPI > Documentation/CAPI.json
sourcekitten doc --spm-module CCAPI > Documentation/CCAPI.json
sourcekitten doc --spm-module Proto > Documentation/Proto.json
sourcekitten doc --spm-module OpProducer > Documentation/OpProducer.json
sourcekitten doc --spm-module TensorFlowKit > Documentation/TensorFlowKit.json
```

### Generate documentation
```
jazzy --config Documentation/CAPI.yaml
jazzy --config Documentation/CCAPI.yaml
jazzy --config Documentation/Proto.yaml
jazzy --config Documentation/OpProducer.yaml
jazzy --config Documentation/TensorFlowKit.yaml

```
### Script
If you want to generate all documentation call script:
```
Documentation/generate.sh
```
