#ifndef NEURON_H
#define NEURON_H

#include "include.h"

class Neuron{
public:
    Neuron(int data_size, int neuron_number, int neuron_layer, bool new_weights);
    ~Neuron();

    static double *input[4];
    //static double result_rbg[3];
    static double *delta[3];
    static double **delta_x_weight[2]; //[i][j][k] - i - layer i + 2; j - layer j + 2; k = neuron_number in layer j + 1
    static int layer_size[3];
    static double learning_rate;

    int neuron_number;
    int neuron_layer;
    int data_size;
    double *weights;
    double error;

    void initialize_weights();
    void save_weights_to_file();
    void read_weights_from_file();
    static void initialize_input_delta(int layer1_size, int layer2_size, int layer3_size);
    static void delte_input_delta(int layer1_size, int layer2_size, int layer3_size);
    void save_delta(double correct_answer[]);
    void save_result();
};

#endif // NEURON_H
