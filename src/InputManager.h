#ifndef PV021_PROJECT_INPUTREADER_H
#define PV021_PROJECT_INPUTREADER_H

#include <vector>
#include <fstream>
#include <cmath>
#include <sstream>

template<typename F = float> struct InputManager;

template<typename F = float>
struct Image {
    friend class InputManager<F>;
    const std::vector<F>& get_pixels() const {
        return pixels;
    }

    F get_label() const{
        return label;
    }

private:
    std::vector<F> pixels;
    //F value_predicted;
    F label; // label
};


template<typename F>
struct InputManager {
    explicit InputManager(const std::string& training) {
        init_images(training);
        compute_mean();
        compute_standard_deviation();
        preprocess_input();
    }

    InputManager(const std::string& training, const std::string& labels): InputManager(training) {
        read_labels(labels);
    }

    void read_labels(const std::string& filename) {
        std::ifstream labels(filename);
        size_t i = 0;
        for(std::string line; std::getline(labels, line);) {
            std::istringstream str(line);
            char colon;
            do {
                str >> training_data[i].label;
                ++i;
            }
            while (str >> colon);
        }
    }

    const std::vector<Image<F>>& get_images() {
        return training_data;
    }

    F get_mean() const {
        return mean;
    }

    F get_standard_deviation() const {
        return standard_deviation;
    }

    size_t get_training_input_count() const {
        return training_data.size();
    }

    size_t get_pixel_per_image_count() const {
        return training_data.front().get_pixels().size();
    }

private:
    void init_images(const std::string& filename) {
        std::ifstream images(filename);
        for(std::string line; std::getline(images, line);) {
            std::istringstream str(line);
            Image<F> image;
            F val;
            char colon;
            do { // read the whole image line
                str >> val;
                image.pixels.push_back(val);
            }
            while (str >> colon);
            training_data.push_back(image);
        }
        /*
        while (!images.eof()) {
            Image<F> image;
            F val;
            char colon;
            do { // read the whole image line
                images >> val >> colon;
                image.pixels.push_back(val);
            }
            while (colon == ',');
            training_data.push_back(image);
        }
         */
    }

    void preprocess_input() {
        for (Image<F>& i : training_data) {
            for (F& p : i.pixels) {
                p = F(p - mean) / F(standard_deviation);
            }
        }
    }


    void compute_mean() {
        for (Image<F>& i : training_data) {
            float current_image_sum = 0.0f;
            for (const F& p : i.pixels)
                current_image_sum += p;

            mean += current_image_sum / i.pixels.size();
        }
        mean /= F(training_data.size());
    }

    void compute_standard_deviation() {
        for (Image<F>& i : training_data) {
            for (const F& p : i.pixels) {
                standard_deviation += (p - mean) * (p - mean);
            }
        }
        standard_deviation /= F(training_data.size() - 1);
        standard_deviation = std::sqrt(standard_deviation);
    }

    //std::ifstream training_data_strm;
    //std::ifstream labels_strm;
    //std::vector<std::vector<F>> training_data;
    std::vector<Image<F>> training_data;
    F mean = F(0);
    F standard_deviation = F(0);
    //const size_t batch_size;
    //ImageInputIterator<F> image_input_iterator;
};

//template<typename T, template<typename> class Iterator>
//Iterator<T> get_first_input(const std::string& filename, size_t batch_size);
//
//template<typename F = float>
//ImageInputIterator<F> get_first_training_input(const std::string& filename, size_t batch_size);
//
//template<typename F = float>
//TestInputIterator<F> get_first_test_input(const std::string& filename, size_t batch_size);


#endif //PV021_PROJECT_INPUTREADER_H
