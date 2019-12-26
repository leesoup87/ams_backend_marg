cd $(dirname $0)
python3 -m grpc_tools.protoc -I./src/pb/SenseClient --descriptor_set_out=./src/pb/SenseClient/api_descriptor.pb --python_out=./src/pb/SenseClient --grpc_python_out=./src/pb/SenseClient src/pb/SenseClient/SenseClient.proto
