#ifndef DEEP_LEARNING_H
#define DEEP_LEARNING_H

// Epsilon greedy constants
#define EPSILON_START 0.05f
#define EPSILON_MIN 0.05f
#define EPSILON_DECAY 0.01f

// Train with only 3D coordiante rather then world converted vertices?
#define TRAIN_ON_POSITION false

// Saving and loading model for deep learning 
#define LOAD_MODEL false
#define SAVE_MODEL true

// Saving the training statistics
#define SAVE_TRAINING_STATS true

// Time parts of the codes execution time
#define TIMING false

#endif 