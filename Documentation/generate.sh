#!/bin/sh

sourcekitten doc --spm-module CAPI > Documentation/CAPI.json
#sourcekitten doc --spm-module CCAPI > Documentation/CCAPI.json
sourcekitten doc --spm-module Proto > Documentation/Proto.json
sourcekitten doc --spm-module OpProducer > Documentation/OpProducer.json
sourcekitten doc --spm-module TensorFlowKit > Documentation/TensorFlowKit.json

jazzy --config Documentation/CAPI.yaml
#jazzy --config Documentation/CCAPI.yaml
jazzy --config Documentation/Proto.yaml
jazzy --config Documentation/OpProducer.yaml
jazzy --config Documentation/TensorFlowKit.yaml

rm Documentation/CAPI.json
rm Documentation/Proto.json
rm Documentation/OpProducer.json
rm  Documentation/TensorFlowKit.json
