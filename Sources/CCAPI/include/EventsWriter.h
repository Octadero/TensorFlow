/* Copyright 2017 The Octadero Authors. All Rights Reserved.
 Created by Volodymyr Pavliukevych on 2017.
 
 Licensed under the GPL License, Version 3.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.gnu.org/licenses/gpl-3.0.txt
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef EventsWriter_h
#define EventsWriter_h

#ifdef __cplusplus
extern "C" {
	
#endif
    /// Function for writing serialized grapht on file system
    /// Using C++ TensorFlow library for that.
	void createEventWriter(const void* serializedGraph, unsigned long  serializedGraphSize, char * filePath, double wall_time, long long step);
#ifdef __cplusplus
}
#endif

#endif
