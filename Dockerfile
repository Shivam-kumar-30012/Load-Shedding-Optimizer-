Copy

FROM ubuntu:22.04
 
RUN apt-get update && apt-get install -y g++
 
WORKDIR /app
COPY . .
 
RUN g++ main.cpp -o main -std=c++17 -O2 -w
 
EXPOSE 10000
CMD ["./main", "--serve", "--port", "10000"]
 
