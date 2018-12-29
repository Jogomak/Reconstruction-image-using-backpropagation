#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent){
    image_width = 24;
    image_height = 29;
    image_pixels_count = image_width * image_height; //696

    this->setMinimumSize(image_width * 10, image_height * 10);

    picture_output = new QImage(image_width, image_height, QImage::Format_ARGB32);
    picture_output_bits = picture_output->bits();
    picture_original = new QImage(":/original_image.png", "PNG");
    picture_original_bits = picture_original->bits();

    picture_output->fill(0xFFFFFFFF);

    new_weights = 0;
    learning_rounds = 0;
    Neuron::learning_rate = 0.001;
    neurons_in_count = 90;
    neurons_hidden_layer_count = 18;
    neurons_out_count = 3;
    prepare_learning_data();
    create_network(neurons_in_count, neurons_hidden_layer_count, neurons_out_count);
    learn_network();
    reconstruct_picture();
}

MainWindow::~MainWindow(){
    Neuron::delte_input_delta(neurons_in_count, neurons_hidden_layer_count, neurons_out_count);
    for(int i = 0; i < neurons_out_count; i++){
        delete network[2][i];
    }
    for(int i = 0; i < neurons_hidden_layer_count; i++){
        delete network[1][i];
    }
    for(int i = 0; i < neurons_in_count; i++){
        delete network[0][i];
    }
    delete []network[2];
    delete []network[1];
    delete []network[0];

    for(int i = 0; i < image_pixels_count; i++){
        delete []correct_answers[i];
        delete []learning_data[i];
    }
    delete []learning_data;
    delete []correct_answers;

    delete picture_original;
    delete picture_output;
}


void MainWindow::paintEvent(QPaintEvent *){
    QPainter painter(this);

    painter.drawImage(0, 0, picture_output->scaled(240, 290));
}

void MainWindow::prepare_learning_data(){
    correct_answers = new double*[image_pixels_count];
    learning_data = new double*[image_pixels_count];
    learning_input_size = 42;

    for(int i = 0; i < image_pixels_count; i++){
        learning_data[i] = new double[learning_input_size];
        learning_data[i][0] = static_cast<double>(i % image_width) * 2 / image_width - 1;/// (0.8 * image_width) + 0.1;   //x
        learning_data[i][1] = static_cast<double>(i / image_width) * 2 / image_height - 1;/// (0.8 * image_height) + 0.1;  //y
        learning_data[i][2] = sin(learning_data[i][0] * 2 * M_PI);  //sin(x*2pi)
        learning_data[i][3] = sin(learning_data[i][1] * 2 * M_PI);  //sin(y*2pi)
        learning_data[i][4] = cos(learning_data[i][0] * 2 * M_PI);  //cos(x*2pi)
        learning_data[i][5] = cos(learning_data[i][1] * 2 * M_PI);  //cos(y*2pi)
        learning_data[i][6] = sin(2 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][7] = sin(2 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][8] = cos(2 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][9] = cos(2 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][10] = sin(3 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][11] = sin(3 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][12] = cos(3 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][13] = cos(3 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][14] = sin(4 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][15] = sin(4 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][16] = cos(4 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][17] = cos(4 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][18] = sin(5 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][19] = sin(5 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][20] = cos(5 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][21] = cos(5 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][22] = sin(6 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][23] = sin(6 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][24] = cos(6 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][25] = cos(6 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][26] = sin(7 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][27] = sin(7 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][28] = cos(7 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][29] = cos(7 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][30] = sin(8 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][31] = sin(8 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][32] = cos(8 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][33] = cos(8 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][34] = sin(9 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][35] = sin(9 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][36] = cos(9 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][37] = cos(9 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][38] = sin(10 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][39] = sin(10 * learning_data[i][1] * 2 * M_PI);
        learning_data[i][40] = cos(10 * learning_data[i][0] * 2 * M_PI);
        learning_data[i][41] = cos(10 * learning_data[i][1] * 2 * M_PI);

        //scaled to 0.1 - 0.9
        correct_answers[i] = new double[3];
        correct_answers[i][0] = picture_original_bits[i * 4 + 0] / 320. + 0.1;  //B
        correct_answers[i][1] = picture_original_bits[i * 4 + 1] / 320. + 0.1;  //G
        correct_answers[i][2] = picture_original_bits[i * 4 + 2] / 320. + 0.1;  //R
        //std::cout << correct_answers[i][0] << ' ' << correct_answers[i][1] << ' ' << correct_answers[i][2] << ' ' << correct_answers[i][3] << '\n';
    }
}

void MainWindow::create_network(int neurons_in_count, int neurons_hidden_layer_count, int neurons_out_count){
    network[0] = new Neuron*[neurons_in_count];
    network[1] = new Neuron*[neurons_hidden_layer_count];
    network[2] = new Neuron*[neurons_out_count];

    for(int i = 0; i < neurons_in_count; i++){
        network[0][i] = new Neuron(learning_input_size, i, 1, new_weights);
    }
    for(int i = 0; i < neurons_hidden_layer_count; i++){
        network[1][i] = new Neuron(neurons_in_count, i, 2, new_weights);
    }
    for(int i = 0; i < neurons_out_count; i++){
        network[2][i] = new Neuron(neurons_hidden_layer_count, i, 3, new_weights);
    }

    Neuron::initialize_input_delta(neurons_in_count, neurons_hidden_layer_count, neurons_out_count);
}

double MainWindow::get_error(){
    double error = 0;

    for(int i = 0; i < image_pixels_count; i++){
        Neuron::input[0] = learning_data[i];
        for(int j = 0; j < neurons_in_count; j++){
            network[0][j]->save_result();
        }
        for(int j = 0; j < neurons_hidden_layer_count; j++){
            network[1][j]->save_result();
        }
        for(int j = 0; j < neurons_out_count; j++){
            network[2][j]->save_result();
        }

        for(int j = 0; j < 3; j++){
            error += pow(Neuron::input[3][j] - correct_answers[i][j], 2.);
        }
    }

    error = error / 2.;
    return error;
}

void MainWindow::update_learning_rate(){
    static double error_old = get_error();
    static double p = 1.05;
    static double increase = 1.1;
    static double decrease = 0.9;
    double error;

    error = get_error();

    if(error > p * error_old){
        Neuron::learning_rate *= decrease;
    }else{
        Neuron::learning_rate *= increase;
    }

    std::cout << error << ' ' << error_old << ' ' << Neuron::learning_rate << '\n';
    error_old = error;
}

void MainWindow::learn_network(){
    for(int t = 0; t < learning_rounds; t++){
        if(t % 1000 == 0){
            //Neuron::learning_rate *= .5;
            update_learning_rate();
        }

        for(int i = 0; i < image_pixels_count; i++){
            Neuron::input[0] = learning_data[i];
            for(int j = 0; j < neurons_in_count; j++){
                network[0][j]->save_result();
            }
            for(int j = 0; j < neurons_hidden_layer_count; j++){
                network[1][j]->save_result();
            }
            for(int j = 0; j < neurons_out_count; j++){
                network[2][j]->save_result();
            }
            for(int j = 0; j < neurons_out_count; j++){
                network[2][j]->save_delta(correct_answers[i]);
            }
            for(int j = 0; j < neurons_hidden_layer_count; j++){
                network[1][j]->save_delta(correct_answers[i]);
            }
            for(int j = 0; j < neurons_in_count; j++){
                network[0][j]->save_delta(correct_answers[i]);
            }
        }
    }
    //save weights to files
    for(int j = 0; j < neurons_in_count; j++){
        network[0][j]->save_weights_to_file();
    }
    for(int j = 0; j < neurons_hidden_layer_count; j++){
        network[1][j]->save_weights_to_file();
    }
    for(int j = 0; j < neurons_out_count; j++){
        network[2][j]->save_weights_to_file();
    }
}


void MainWindow::reconstruct_picture(){
    for(int i = 0; i < image_pixels_count; i++){
        Neuron::input[0] = learning_data[i];
        for(int j = 0; j < neurons_in_count; j++){
            network[0][j]->save_result();
        }
        for(int j = 0; j < neurons_hidden_layer_count; j++){
            network[1][j]->save_result();
        }
        for(int j = 0; j < neurons_out_count; j++){
            network[2][j]->save_result();
        }

        picture_output_bits[i * 4 + 0] = static_cast<unsigned char>(round((Neuron::input[3][0] - 0.1) * 320));
        picture_output_bits[i * 4 + 1] = static_cast<unsigned char>(round((Neuron::input[3][1] - 0.1) * 320));
        picture_output_bits[i * 4 + 2] = static_cast<unsigned char>(round((Neuron::input[3][2] - 0.1) * 320));

        //std::cout << i << ' ' << (Neuron::input[3][0]) << ' ' << Neuron::input[3][1] << ' ' << Neuron::input[3][2] << '\n';
    }

    update();
}
