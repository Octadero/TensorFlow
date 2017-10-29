# TensorFlow Swift high-level API.


## Structure of API
![architecture](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/TensorFlowProject@2x.png)

API based on [TensorFlow](https://www.tensorflow.org) library.
* CTensorFlow is C API [system module](https://github.com/apple/swift-package-manager/blob/master/Documentation/Usage.md#require-system-libraries);
* _Removed from project_ CCTensorFlow is C++ API [system module](https://github.com/apple/swift-package-manager/blob/master/Documentation/Usage.md#require-system-libraries);
* _Removed from project_ CProtobuf is protobuf library [system module](https://github.com/apple/swift-package-manager/blob/master/Documentation/Usage.md#require-system-libraries);

* CAPI - Swift writen low-level API to C library;
* _Removed from project_ CCAPI - Swift writen low-level API to C+ library;
* Proto - Swift auto - generated classes for TensorFlow structures and models;
* OpPruducer - Swift writen command line tool to produce new [TensorFlow Operations](https://www.tensorflow.org/extend/architecture)
* TensorFlowKit - Swift writen high-level API;

## Developing and extending API

### Create new proto files

* Install swift protobuf generator from [Protobuf Swift library](https://github.com/apple/swift-protobuf) 

* Execute commands

```
// Create temperory folder
mkdir /tmp/swift
cd %path-to-tensorflow-reposytory%

// Find all proto files and generate swift classes.
find. -name '*.proto' -print -exec protoc --swift_opt=Visibility=Public --swift_out=/tmp/swift {} \;

// All files will be removed after restart.
open /tmp/swift
```

### List of operations
There are a few ways to get a list of the OpDefs for the registered ops:

```
TF_GetAllOpList in the C API retrieves all registered OpDef protocol messages. This can be used to write the generator in the client language. This requires that the client language have protocol buffer support in order to interpret the OpDef messages.

The C++ function OpRegistry::Global()->GetRegisteredOps() returns the same list of all registered OpDefs (defined in [tensorflow/core/framework/op.h]). This can be used to write the generator in C++ (particularly useful for languages that do not have protocol buffer support).

The ASCII-serialized version of that list is periodically checked in to [tensorflow/core/ops/ops.pbtxt] by an automated process.

```
OpProducer using C API to extract and prepare all available operation as Swift source code.

## Using library.

First of all you should install tensorflow_c library. You can do that using brew on mac os or apt on ubuntu.
Also, you can install it from sources, [how to install TensorFlow from sources you can find here](https://www.octadero.com/2017/08/27/tensorflow-c-environment/).

### Xcode
*Make shure that you read README of submodule CTensorFlow*

To generate xcode project file you can call:

```
swift package -Xlinker -rpath -Xlinker /server/repository/tensorflow/bazel-bin/tensorflow generate-xcodeproj
```
_Where */server/repository/tensorflow/bazel-bin/tensorflow* path to your TensorFlow C library_

Also you can use config:
```
swift package generate-xcodeproj --xcconfig-overrides TensorFlow.xcconfig
```

* At `TensorFlow.xcconfig` file set `TENSORFLOW_PATH` property with correct path.
* It is important to set 'TensorFlow.xcconfig' name the same with projectfile.
* *There is [issus SR-6073](https://bugs.swift.org/browse/SR-6073)* with LD_RUNPATH_SEARCH_PATHS property. So, currently you have to set `$(inherited)` value manualy at your build variable.

Build with RPATH setting:
```
swift build -Xcxx -std=c++11 -Xlinker -rpath -Xlinker /server/repository/tensorflow/bazel-bin/tensorflow -Xlinker -L/server/repository/tensorflow/bazel-bin/tensorflow -Xlinker -ltensorflow
swift test -Xcxx -std=c++11 -Xlinker -rpath -Xlinker /server/repository/tensorflow/bazel-bin/tensorflow -Xlinker -L/server/repository/tensorflow/bazel-bin/tensorflow -Xlinker -ltensorflow
```
### Features
Swift API provides accae to all available C features in TensorFlow library.

* Create / read grapht;
* Save statistic on file system;
* Cunstruct and run session;
* Include available operations;

![graph](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/grapht@2x.png)

### Troubleshooting
* In case app can't load dynamic library on macOS:

```
otool -L libtensorflow_cc.so
sudo install_name_tool -id /%repository%/tensorflow/bazel-out/darwin_x86_64-opt/bin/tensorflow/libtensorflow_cc.so libtensorflow_cc.so
sudo install_name_tool -id /%repository%/tensorflow/bazel-out/darwin_x86_64-opt/bin/tensorflow/libtensorflow_framework.so libtensorflow_framework.so
sudo install_name_tool -id /%repository%/tensorflow/bazel-out/darwin_x86_64-opt/bin/tensorflow/libtensorflow.so libtensorflow.so
```

* In case if app can't load dynamic library on linux or mac os with error:
```
error while loading shared libraries: *libtensorflow_cc.so*: cannot open shared object file: No such file or directory
```
Add your library path linux:
```
export LD_LIBRARY_PATH=/server/repository/tensorflow/bazel-bin/tensorflow:/usr/local/lib/:$LD_LIBRARY_PATH
```
mac os:
```
export DYLD_LIBRARY_PATH=/server/repository/tensorflow/bazel-bin/tensorflow/:$DYLD_LIBRARY_PATH
```

---
C++ old related issues.

* At build phase error:
```
warning: error while trying to use pkgConfig flags for CProtobuf: nonWhitelistedFlags("Non whitelisted flags found: [\"-D_THREAD_SAFE\", \"-D_THREAD_SAFE\"] in pc file protobuf")
```
Remove `-D_THREAD_SAFE` keys from file /usr/local/lib/pkgconfig/protobuf.pc

Error at build phase:
```
#include "tensorflow/core/framework/graph.pb.h"                                                                                                                                         ^                                                                                                                                                                     1 error generated.                                                                                                                                                             error: terminated(1): /usr/bin/swift-build-tool -f /server/repository/TensorFlow/.build/debug.yaml main
```

Download dependencies for C++ library at tensorflow repository.
```
tensorflow/contrib/makefile/download_dependencies.sh
```

