#ifndef STACK_H
#define STACK_H

class Stack{

    private:
        int top;
        int size;
        int stack_data[30];
    
    public:
        // Initialise the stack
        __host__ __device__
        Stack(int max_size);

        // Destroy the stack
        __host__ __device__
        ~Stack();

        // Push an element on to the stack
        __host__ __device__
        bool push(int x);

        // Pop the top element off of the stack and return it
        __host__ __device__
        bool pop(int& value);

        // Peek at the top of the stack and return the top element
        __host__ __device__
        bool peek(int& value);
};

#endif