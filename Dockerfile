FROM ubuntu:22.04

RUN apt-get update && apt-get install -y g++ make

WORKDIR /app
COPY . .

RUN g++ main.cpp -o main -std=c++17 -O2 -w

EXPOSE 8080
CMD ["./main", "--serve", "--port", "8080"]
