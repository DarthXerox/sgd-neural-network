#ifndef PV021_PROJECT_INPUTREADER_H
#define PV021_PROJECT_INPUTREADER_H

#include <vector>
#include <fstream>

template<typename T>
struct InputIterator {
    InputIterator(const std::string& filename, size_t batch_size);
    ~InputIterator();
    T& operator*();
    InputIterator& operator++();
    bool is_last();

protected:
    std::vector<T> current_batch;
    std::vector<T>::iterator current_item_it;
    std::ifstream input_file;
    const size_t batch_size;
};


template<typename F = float>
struct ImageInputIterator : public InputIterator<std::vector<F>> {
    ImageInputIterator(const std::string& filename, size_t batch_size);
    size_t get_input_count() const;
    void rewind() {
        input_file.clear();
        input_file.seekg(0);
        load_new_batch();
    }

private:
    void load_new_batch();
};


template<typename F = float>
struct TestInputIterator : public InputIterator<F> {
    TestInputIterator(const std::string& filename, size_t batch_size);

private:
    void load_new_batch();
};


template<typename F = float>
struct InputManager {
    InputManager(const std::string& filename, size_t batch_size)
    : batch_size(batch_size), image_input_iterator(filename, batch_size) {
        compute_mean();
        image_input_iterator.rewind();
        compute_standard_deviation();
        image_input_iterator.rewind();
    }

    std::vector<F> get_next_image();
    F get_next_label();


    void load_test_labels() {

    }

private:
    void compute_mean() {
        size_t image_count = 0;
        while (!first_iterator.is_last()) {
            float current_image_sum = 0.0f;
            for (const F& n : *first_iterator)
                current_image_sum += n;

            mean += current_image_sum / (*first_iterator).size();
            image_count++;
            ++first_iterator;
        }
        mean /= F(image_count);
    }

    void compute_standard_deviation() {
        while (!first_iterator.is_last()) {
            for (const F& n : *first_iterator) {
                standard_deviation += (n - mean) * (n - mean);
            }
            ++first_iterator;
        }
        standard_deviation /= F(image_count - 1);
        standard_deviation = std::sqrt(standard_deviation);
    }

    F mean = F(0);
    F standard_deviation = F(0);
    const size_t batch_size;
    ImageInputIterator<F> image_input_iterator;
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
