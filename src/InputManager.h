#ifndef PV021_PROJECT_INPUTREADER_H
#define PV021_PROJECT_INPUTREADER_H

#include <vector>
#include <fstream>

template<typename F = float>
struct Image {
    std::vector<F> pixels;
    //F value_predicted;
    F actual_value; // label
};

//
//template<typename T>
//struct InputIterator {
//    InputIterator(const std::string& filename, size_t batch_size);
//    ~InputIterator();
//    T& operator*();
//    InputIterator& operator++();
//    bool is_last();
//
//protected:
//    std::vector<T> current_batch;
//    std::vector<T>::iterator current_item_it;
//    std::ifstream input_file;
//    const size_t batch_size;
//};
//
//
//template<typename F = float>
//struct ImageInputIterator : public InputIterator<std::vector<F>> {
//    ImageInputIterator(const std::string& filename, size_t batch_size);
//    size_t get_input_count() const;
//    void rewind() {
//        input_file.clear();
//        input_file.seekg(0);
//        load_new_batch();
//    }
//
//private:
//    void load_new_batch();
//};
//
//
//template<typename F = float>
//struct TestInputIterator : public InputIterator<F> {
//    TestInputIterator(const std::string& filename, size_t batch_size);
//
//private:
//    void load_new_batch();
//};


template<typename F = float>
struct InputManager {
    InputManager(const std::string& training) {
        init_images(training);
        compute_mean();
        compute_standard_deviation();
        preprocess_input();
    }

    InputManager(const std::string& training, const std::string& labels): InputManager(trainig) {
        read_labels(labels);
    }

    void read_labels(const std::string& filename) {
        std::ifstream labels(filename);
        size_t i = 0;
        char new_line;

        while (!labels.eof() && i < training_data.size()) {
            input_file >> training_data[i].actual_value >> new_line;
            ++i;
        }
    }

    std::vector<Image<F>>& get_images() {
        return training_data;
    }

private:
    void init_images(const std::string& filename) {
        while (!input_file.eof()) {
            Image<F> image;
            F val;
            char colon;
            do { // read the whole image line
                input_file >> val >> colon;
                image.pixels.push_back(val);
            }
            while (colon == ',');
            training_data.push_back(image);
        }
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
            image_count++;
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
