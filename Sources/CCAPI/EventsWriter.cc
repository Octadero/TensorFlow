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

#include "include/EventsWriter.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/util/events_writer.h"

using namespace std;
using namespace tensorflow;

void write(const void* serializedGraph, size_t serializedGraphSize, EventsWriter* writer, double wall_time, int64 step) {
	Event event;
	event.set_wall_time(wall_time);
	event.set_step(step);
	event.set_graph_def(serializedGraph, serializedGraphSize);
	writer->WriteEvent(event);
	
}

void createEventWriter(const void* serializedGraph, size_t serializedGraphSize, char * filePath, double wall_time, long long step) {
	EventsWriter eventsWriter(filePath);
	write(serializedGraph, serializedGraphSize, &eventsWriter, wall_time, step);
}
