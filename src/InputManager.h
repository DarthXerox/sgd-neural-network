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


struct TrainingInputIterator : public InputIterator<std::vector<int>> {
    TrainingInputIterator(const std::string& filename, size_t batch_size);
    size_t get_input_count() const;

private:
    void load_new_batch();
};


struct TestInputIterator : public InputIterator<int> {
    TestInputIterator(const std::string& filename, size_t batch_size);

private:
    void load_new_batch();
};


template<typename T, template<typename> class Iterator>
Iterator<T> get_first_input(const std::string& filename, size_t batch_size);
TrainingInputIterator get_first_training_input(const std::string& filename, size_t batch_size);
TestInputIterator get_first_test_input(const std::string& filename, size_t batch_size);


#endif //PV021_PROJECT_INPUTREADER_H
