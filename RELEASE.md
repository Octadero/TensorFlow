# Release Notes

## Release 0.0.8
### CAPI
* Fixed issue with `Graph` operations iteration.
* Fixed issue with EXC_BAD_ACCESS at `Session` loading operation.

### TensorFlowKit
* Reorganised SaveModel class API.
* Added posibility to pass `Tensorflow_RunOptions` option to `Session`.
* Added posibility to pass `Tensorflow_ConfigProto` option to `SessionOptions`.
* Improved tests.

### Proto
* Updated `proto` classes.

### OpProducer
* Nothing new.

## Release 0.0.7
### CAPI
* Improved String encoding and decoding operations.
* Improved OpList extractor.

### TensorFlowKit
* Added SavedModel class. Save and Restore operations now available from the box.
* Improved `Tensor` behaviour as container of strings.
* Added new `Seesion` run function calling outputs and inputs by names.
* Added new functions and variables to `Operation` class.
* Added control dependency feature.
* Added new way to add `Tensor` constants to `Graph`.

### Proto
* Updated `proto` classes.

### OpProducer
* Fixed issue with `typeListAttr` values;

## Release 0.0.6
### TensorFlowKit
* Added more flaxible way to import and export TensorFlow graph;


## Release 0.0.5
### CAPI
* Added simple way to list all operations in `Graph`.

### TensorFlowKit
* Added FileWriter;
* Added Summary;
* Added simple way to list all operations in `Graph`.

### Proto
* Updated `proto` classes.

### OpProducer
* Improved producer;

## Release 0.0.4
### CAPI
* Nothing changed.

### CCAPI
* Removed from project. After TensorFlow r1.4 release I hope, that module will not be not necessary.

### TensorFlowKit
* Added EventWriter;
* Improved Operations wrapper;

### OpProducer
* Improved producer;


## Release 0.0.3
### CAPI
* Nothing changed.

### CCAPI
* Removed from project. After TensorFlow r1.4 release I hope, that module will not be not necessary.

### TensorFlowKit
* Nothing changed.

### OpProducer
* Nothing changed.


## Release 0.0.2
### CAPI
* New Status error representation.

### CCAPI
* Nothing.

### TensorFlowKit
* Added Gradient functions.

### OpProducer
* Moved all operations to Scope extension.


## Release 0.0.1
### CAPI
* Initial release of CAPI library.
* TensorFlow C API was bridged to swift code through CCTensorFlow sub package.

### CCAPI
* Initial release of CCAPI library.
* TensorFlow C++ API was bridged to swift code through CCTensorFlow sub package.
* Added first wrapper to use Event writer API.

### TensorFlowKit
* Initial release of TensorFlowKit.
* Created main features of:
	* Graph
	* Scope
	* Session
	* Operation
	
### OpProducer
* Initial release of code-generator.

### Tests
* First implementation of tests.
