#include "InputManager.h"
#include <vector>
#include <fstream>


template<typename T>
InputIterator<T>::InputIterator(std::string filename, size_t batch_size)
: input_file(filename), batch_size(batch_size) {}


TrainingInputIterator::TrainingInputIterator(std::string filename, size_t batch_size)
: InputIterator<std::vector<int>>(filename, batch_size) {
    load_new_batch();
}


TestInputIterator::TestInputIterator(std::string filename, size_t batch_size)
: InputIterator<int>(filename, batch_size) {
    load_new_batch();
}


template<typename T>
InputIterator<T>::~InputIterator() {
    input_file.close();
}


template<typename T>
InputIterator<T>& InputIterator<T>::operator++() {
    current_item_it++;
    if (current_item_it == current_batch.end()) {
        load_new_batch();
    }
    return this;
}


template<typename T>
T& InputIterator<T>::operator*() {
    return *current_item_it;
}


template<typename T>
bool InputIterator<T>::is_last() const {
    return input_file.eof() && current_item_it == --current_batch.end();
}


void TrainingInputIterator::load_new_batch() {
    current_batch.clear();
    for (size_t i = 0; i < this.batch_size && input_file.eof(); ++i) {
        std::vector<int> current_input;
        int val;
        char colon;
        do {
            input_file >> val >> colon;
            current_input.push_back(val);
        }
        while (colon == ',');
        current_batch.push_back(current_input);
    }
    current_item_it = current_batch.begin();
}


void TestInputIterator::load_new_batch() {
    current_batch.clear();
    for (size_t i = 0; i < this.batch_size && input_file.eof(); ++i) {
        int val;
        char colon;
        do {
            input_file >> val >> colon;
            current_batch.push_back(val);
        }
        while (colon == ',');
    }
    current_item_it = current_batch.begin();
}


size_t TrainingInputIterator::get_input_count() const {
    return current_batch.front().size();
}

template<typename T, template<typename> class Iterator>
Iterator<T> get_first_input(const std::string& filename, size_t batch_size) {
    return Iterator<T>(filename, batch_size);
}


TrainingInputIterator get_first_training_input(const std::string& filename, size_t batch_size) {
    return get_first_input<std::vector<int>, TrainingInputIterator>(filename, batch_size);
}


TestInputIterator get_first_test_input(const std::string& filename, size_t batch_size) {
    return get_first_input<TestInputIterator>(filename, batch_size);
}

