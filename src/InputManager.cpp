#include "InputManager.h"
#include <vector>
#include <fstream>


template<typename T>
InputIterator<T>::InputIterator(std::string filename, size_t batch_size)
: input_file(filename), batch_size(batch_size) {}


template<typename F>
ImageInputIterator<F>::ImageInputIterator(std::string filename, size_t batch_size)
: InputIterator<std::vector<F>>(filename, batch_size) {
    load_new_batch();
}


template<typename F>
TestInputIterator<F>::TestInputIterator(std::string filename, size_t batch_size)
: InputIterator<F>(filename, batch_size) {
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

//template<typename T>
//T* InputIterator<T>::operator->() {
//    return &*current_item_it;
//}


template<typename T>
bool InputIterator<T>::is_last() const {
    return input_file.eof() && current_item_it == --current_batch.end();
}


template<typename F>
void ImageInputIterator<F>::load_new_batch() {
    this->current_batch.clear();
    for (size_t i = 0; i < this->batch_size && !input_file.eof(); ++i) {
        std::vector<F> current_input;
        F val;
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


template<typename F>
void TestInputIterator<F>::load_new_batch() {
    this->current_batch.clear();
    for (size_t i = 0; i < this.batch_size && !input_file.eof(); ++i) {
        F val;
        char colon;
        input_file >> val >> colon;
        current_batch.push_back(val);
    }
    current_item_it = current_batch.begin();
}


template<typename F>
size_t ImageInputIterator<F>::get_input_count() const {
    return current_batch.front().size();
}

//template<typename T, template<typename> class Iterator>
//Iterator<T> get_first_input(const std::string& filename, size_t batch_size) {
//    return Iterator<T>::Iterator(filename, batch_size);
//}
//
//
//template<typename F>
//ImageInputIterator<F> get_first_training_input(const std::string& filename, size_t batch_size) {
//    return get_first_input<std::vector<F>, ImageInputIterator>(filename, batch_size);
//}
//
//
//template<typename F>
//TestInputIterator<F> get_first_test_input(const std::string& filename, size_t batch_size) {
//    return get_first_input<F, TestInputIterator>(filename, batch_size);
//}

