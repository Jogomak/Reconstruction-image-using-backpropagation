#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "include.h"
#include "neuron.h"

class MainWindow : public QMainWindow{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    QImage *picture_original;
    QImage *picture_output;

    unsigned char *picture_original_bits;
    unsigned char *picture_output_bits;

    int image_width;
    int image_height;
    int image_pixels_count;

    int learning_input_size;
    double **learning_data;
    double **correct_answers;

    int neurons_in_count;
    int neurons_hidden_layer_count;
    int neurons_out_count;
    Neuron **network[3];

    bool new_weights;
    int learning_rounds;

    double get_error();
    void update_learning_rate();
    void prepare_learning_data();
    void create_network(int neurons_in_count, int neurons_hidden_layer_count, int neurons_out_count);
    void learn_network();
    void reconstruct_picture();

private slots:
    void paintEvent(QPaintEvent *);
};

#endif // MAINWINDOW_H
