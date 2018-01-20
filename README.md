# TensorFlow Swift high-level API. [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=TensorFlowKit%20is%20Swift%20API%20for%20tensorflow,%20easiest%20way%20to%20do%20your%20app%20more%20intelligent&url=https://github.com/Octadero/TensorFlow&via=octadero&hashtags=swift,ml,tensorflow,NeuralNetworks,AI)



<p align="center">
<a href="https://developer.apple.com/swift/" target="_blank">
<img src="https://img.shields.io/badge/Swift-4.0-orange.svg?style=flat" alt="Swift 4.0">
</a>
<a href="https://developer.apple.com/swift/" target="_blank">
<img src="https://img.shields.io/badge/Platforms-%20Linux%20%26%20OS%20X%20-brightgreen.svg?style=flat" alt="Platforms Linux & OS X">
</a>
<a href="https://github.com/Octadero/TensorFlow/blob/master/LICENSE" target="_blank">
<img src="https://img.shields.io/aur/license/yaourt.svg?style=flat" alt="GPL">
</a>
<a href="http://twitter.com/Octadero" target="_blank">
<img src="https://img.shields.io/badge/Twitter-@Octadero-0084b4.svg?style=flat" alt="Octadero Twitter">
</a>
</p>

## Structure of API
![architecture](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/TensorFlowProject@2x.png)

API based on [TensorFlow](https://www.tensorflow.org) library.
* CTensorFlow is C API [system module](https://github.com/apple/swift-package-manager/blob/master/Documentation/Usage.md#require-system-libraries);
* CAPI - Swift writen low-level API to C library;
* Proto - Swift auto - generated classes for TensorFlow structures and models;
* OpPruducer - Swift writen command line tool to produce new [TensorFlow Operations](https://www.tensorflow.org/extend/architecture)
* TensorFlowKit - Swift writen high-level API;

## Using library.

First of all you should install tensorflow_c library. You can do that using brew on mac os.
Version 1.4 currantly  available for [mac os](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.4.1.tar.gz) and [linux](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.4.1.tar.gz) on Google cloud, so you can use it:
Also, you can install it from sources, [how to install TensorFlow from sources you can find here](https://www.octadero.com/2017/08/27/tensorflow-c-environment/).

### Tutorials
* There is [review of technology stack](https://www.octadero.com/2017/11/14/swift-and-tensorflow/)
* [Short tutorial](https://www.octadero.com/2017/11/16/mnist-by-tensorflowkit/), please review them.

### Xcode
*Make shure that you read README of submodule CTensorFlow*

To generate xcode project file you can call:
If you install TensorFlow library (1.4.1 version) as brew package:
```
swift package -Xlinker -rpath -Xlinker /usr/local/Cellar/libtensorflow/1.4.1/lib/ generate-xcodeproj
```
```
swift package -Xlinker -rpath -Xlinker /server/repository/tensorflow/bazel-bin/tensorflow generate-xcodeproj
```
_Where */server/repository/tensorflow/bazel-bin/tensorflow* path to your TensorFlow C library */user/local/lib, usualy.*_

Also you can use config:
```
swift package generate-xcodeproj --xcconfig-overrides TensorFlow.xcconfig
```

* At `TensorFlow.xcconfig` file set `TENSORFLOW_PATH` property with correct path.
* It is important to set 'TensorFlow.xcconfig' name the same with projectfile.
* *There is [issus SR-6073](https://bugs.swift.org/browse/SR-6073)* with LD_RUNPATH_SEARCH_PATHS property. So, currently you have to set `$(inherited)` value manualy at your build variable.

Build and set RPATH setting:

```
#MacOS
swift build -Xlinker -rpath -Xlinker /usr/local/Cellar/libtensorflow/1.4.1/lib/
swift test -Xlinker -rpath -Xlinker /usr/local/Cellar/libtensorflow/1.4.1/lib/
```

```
#Linux
swift build -Xlinker -rpath -Xlinker /server/repository/tensorflow/bazel-bin/tensorflow
swift test -Xlinker -rpath -Xlinker /server/repository/tensorflow/bazel-bin/tensorflow
```

### Features
Swift API provides accae to all available C features in TensorFlow library.

* Create / read grapht;
* Save statistic on file system;
* Cunstruct and run session;
* Include available operations;
* Store and restore checkpoints by SavedModel class;

![graph](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/grapht@2x.png)

## Summary
![Summary](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/tensorboard-1.gif)

Starting from version 0.0.5 you have posibility to track any metrics using TensorFlowKit.
That is easy way to visualize your model in Swift application.
### You can visualize weights and biases:
![summary-distribution](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/summary-distribution@2x.png)

### Draw your graph:
![summary-graph](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/summary-graph@2x.png)

### Track changes in 3D:
![summary-histogram](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/summary-histogram@2x.png)

### Extract components as png images:
![summary-image](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/summary-image@2x.png)

### Watch on dynamics of changes:
![summary-scalar](https://raw.githubusercontent.com/Octadero/TensorFlow/master/Documentation/resources/summary-scalar@2x.png)

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

### Debuging
Sets the threshold for what messages will be logged.
Add `TF_CPP_MIN_LOG_LEVEL=0` and `TF_CPP_MIN_VLOG_LEVEL=0`


### List of operations
There are a few ways to get a list of the OpDefs for the registered ops:

```
TF_GetAllOpList in the C API retrieves all registered OpDef protocol messages. This can be used to write the generator in the client language. This requires that the client language have protocol buffer support in order to interpret the OpDef messages.

The C++ function OpRegistry::Global()->GetRegisteredOps() returns the same list of all registered OpDefs (defined in [tensorflow/core/framework/op.h]). This can be used to write the generator in C++ (particularly useful for languages that do not have protocol buffer support).

The ASCII-serialized version of that list is periodically checked in to [tensorflow/core/ops/ops.pbtxt] by an automated process.

```
OpProducer using C API to extract and prepare all available operation as Swift source code.



