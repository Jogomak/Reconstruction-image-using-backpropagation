#include "neuron.h"

double *Neuron::input[4] = {nullptr, nullptr, nullptr, nullptr};
//double *Neuron::input1 = nullptr;
//double *Neuron::input2 = nullptr;
//double *Neuron::input3 = nullptr;
//double Neuron::result_rbg[3] = {0, 0, 0};
double *Neuron::delta[3] = {nullptr, nullptr, nullptr};
//double *Neuron::delta1 = nullptr;
//double *Neuron::delta2 = nullptr;
//double *Neuron::delta3 = nullptr;
double **Neuron::delta_x_weight[2] = {nullptr, nullptr};
double Neuron::learning_rate = 0.001;
int Neuron::layer_size[3] = {0, 0, 0};

Neuron::Neuron(int data_size, int neuron_number, int neuron_layer, bool new_weights){
    this->data_size = data_size;
    this->neuron_number = neuron_number;
    this->neuron_layer = neuron_layer;
    weights = new double[data_size + 1];

    if(new_weights){
        initialize_weights();
    }else{
        read_weights_from_file();
    }
}

Neuron::~Neuron(){
    delete weights;
}

void Neuron::initialize_weights(){
    srand(static_cast<unsigned int>(time(nullptr)));

    for(int i = 0; i < data_size + 1; i++){
        weights[i] = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
}

void Neuron::read_weights_from_file(){
    std::fstream file;
    long long int *binary_weight = new long long int();
    char path[20];
    //sprintf(path, "weights/%d_%d.txt", neuron_layer, neuron_number);
    sprintf(path, "%d_%d.txt", neuron_layer, neuron_number);

    file.open(path, std::ios::in);
    if(file.good()){
        for(int i = 0; i < data_size + 1; i++){
            file >> *binary_weight;
            weights[i] = *reinterpret_cast<double*>(binary_weight);
        }
    }
    file.close();
    delete binary_weight;
}

void Neuron::save_weights_to_file(){
    std::fstream file;
    long long int *binary_weight;
    char data_to_save[21];
    char path[20];
    //sprintf(path, "weights/%d_%d.txt", neuron_layer, neuron_number);
    sprintf(path, "%d_%d.txt", neuron_layer, neuron_number);

    file.open(path, std::ios::out | std::ios::trunc);
    if(file.good()){
        for(int i = 0; i < data_size + 1; i++){
            binary_weight = reinterpret_cast<long long int*>(weights + i);
            sprintf(data_to_save, "%lld\n", *binary_weight);
            file << data_to_save;
        }
    }
    file.close();
}

void Neuron::initialize_input_delta(int layer1_size, int layer2_size, int layer3_size){
    Neuron::layer_size[0] = layer1_size;
    Neuron::layer_size[1] = layer2_size;
    Neuron::layer_size[2] = layer3_size;

    Neuron::input[1] = new double[layer1_size];
    Neuron::input[2] = new double[layer2_size];
    Neuron::input[3] = new double[layer3_size];  //result {R, G, B}
    Neuron::delta[0] = new double[layer1_size];
    Neuron::delta[1] = new double[layer2_size];
    Neuron::delta[2] = new double[layer3_size];
    Neuron::delta_x_weight[0] = new double*[layer2_size];
    Neuron::delta_x_weight[1] = new double*[layer3_size];
    for(int i = 0; i < layer2_size; i++){
        Neuron::delta_x_weight[0][i] = new double[layer1_size];
    }
    for(int i = 0; i < layer3_size; i++){
        Neuron::delta_x_weight[1][i] = new double[layer2_size];
    }
}

void Neuron::delte_input_delta(int layer1_size, int layer2_size, int layer3_size){
    for(int i = 0; i < layer3_size; i++){
        delete []Neuron::delta_x_weight[1][i];
    }
    for(int i = 0; i < layer2_size; i++){
        delete []Neuron::delta_x_weight[0][i];
    }
    delete []Neuron::delta_x_weight[1];
    delete []Neuron::delta_x_weight[0];
    delete []Neuron::delta[2];
    delete []Neuron::delta[1];
    delete []Neuron::delta[0];
    delete []Neuron::input[3];
    delete []Neuron::input[2];
    delete []Neuron::input[1];
}

void Neuron::save_result(){
    input[neuron_layer][neuron_number] = 0;

    for(int i = 0; i < data_size; i++){
        input[neuron_layer][neuron_number] += weights[i] * input[neuron_layer - 1][i];
    }
    input[neuron_layer][neuron_number] += weights[data_size];

    input[neuron_layer][neuron_number] = 1 / (1 + exp(-input[neuron_layer][neuron_number]));
    //std::cout << neuron_layer << ' ' << neuron_number << ' ' << input[neuron_layer][neuron_number] << '\n';
}

void Neuron::save_delta(double correct_answer[]){
    double delta_x_weight_sum;
    delta[neuron_layer - 1][neuron_number] = 0;

    if(neuron_layer == 3){
        delta[neuron_layer - 1][neuron_number] = -(correct_answer[neuron_number] - input[neuron_layer][neuron_number]) * input[neuron_layer][neuron_number] * (1 - input[neuron_layer][neuron_number]);

        for(int i = 0; i < data_size; i++){
            delta_x_weight[neuron_layer - 2][neuron_number][i] = delta[neuron_layer - 1][neuron_number] * weights[i];

            weights[i] = weights[i] - learning_rate * delta[neuron_layer - 1][neuron_number] * input[neuron_layer - 1][i];
        }
        weights[data_size] = weights[data_size] - learning_rate * delta[neuron_layer - 1][neuron_number] * input[neuron_layer - 1][data_size];
    }else{
        delta_x_weight_sum = 0;
        for(int i = 0; i < layer_size[neuron_layer]; i++){
            delta_x_weight_sum += delta_x_weight[neuron_layer - 1][i][neuron_number];
        }

        delta[neuron_layer - 1][neuron_number] = delta_x_weight_sum * input[neuron_layer][neuron_number] * (1 - input[neuron_layer][neuron_number]);

        if(neuron_layer > 1){
            for(int i = 0; i < data_size; i++){
                delta_x_weight[neuron_layer - 2][neuron_number][i] = delta[neuron_layer - 1][neuron_number] * weights[i];
            }
        }

        for(int i = 0; i < data_size; i++){
            weights[i] = weights[i] - learning_rate * delta[neuron_layer - 1][neuron_number] * input[neuron_layer - 1][i];
        }
        weights[data_size] = weights[data_size] - learning_rate * delta[neuron_layer - 1][neuron_number] * input[neuron_layer - 1][data_size];
    }
}
