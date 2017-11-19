# Release Notes

## Release 0.0.4
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
