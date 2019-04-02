#include "stack.cuh"
#include <stdio.h>

__device__
Stack::Stack(int max_size){
    // this->stack_data = new int[ max_size ];
    // memset(this->stack_data, 0, max_size);
    this->top = 0;
    this->size = max_size;
}

__device__
Stack::~Stack(){
    // delete [] stack_data;
}

__device__
bool Stack::push(int x){
    if (this->top < this->size - 1){
        this->stack_data[this->top] = x;
        this->top++;
        return true;
    }
    else{
        return false;
    }
}

__device__
bool Stack::pop(int& value){
    if (this->top > 0){
        this->top--;
        value = this->stack_data[this->top];
        return true;
    }
    else{
        return false;
    }
}

__device__
bool Stack::peek(int& value){
    if (this->top > 0){
        value = this->stack_data[this->top-1];
        return true;
    }
    else{
        return false;
    }
}