#ifndef MOL_PARA_H
#define MOL_PARA_H
#include "aligned_allocator.h"
#include <cmath>
#include <utility>

using namespace std;

class Multi_MOL_parameters {
public:
  AlignedDoubleVector charge;
  AlignedDoubleVector radius;
  AlignedDoubleVector sigma;
  AlignedDoubleVector epsilon;
  AlignedDoubleVector sqrt_epsilon;
  AlignedDoubleVector mass;
  AlignedIntVector type_id;

  Multi_MOL_parameters() = default;
  explicit Multi_MOL_parameters(size_t size) { resize(size); }

  struct Proxy {
    Multi_MOL_parameters *ptr;
    size_t idx;

    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    void set_data(T1 &&_charge, T2 &&_radius, T3 &&_sigma, T4 &&_epsilon,
                  T5 &&_mass) {
      ptr->charge[idx] = std::forward<T1>(_charge);
      ptr->radius[idx] = std::forward<T2>(_radius);
      ptr->sigma[idx] = std::forward<T3>(_sigma);
      ptr->epsilon[idx] = std::forward<T4>(_epsilon);
      ptr->mass[idx] = std::forward<T5>(_mass);
      ptr->sqrt_epsilon[idx] = std::sqrt(ptr->epsilon[idx]);
    }
    void set_charge(double c) { ptr->charge[idx] = c; }
    void set_mass(double m) { ptr->mass[idx] = m; }
    void set_type_id(int32_t id) { ptr->set_type_id(idx, id); }
    void set_epsilon_sigma_radius(double e, double s, double r) {
      ptr->epsilon[idx] = e;
      ptr->sigma[idx] = s;
      ptr->radius[idx] = r;
      ptr->sqrt_epsilon[idx] = std::sqrt(e);
    }

    double get_mass() const {
      return (ptr->mass[idx] > 0.01) ? ptr->mass[idx] : 0.01;
    }
    double get_charge() const { return ptr->charge[idx]; }
    double get_radius() const { return ptr->radius[idx]; }
    double get_epsilon() const { return ptr->epsilon[idx]; }
    double get_sqrt_epsilon() const { return ptr->sqrt_epsilon[idx]; }
    double get_sigma() const { return ptr->sigma[idx]; }
    int32_t get_type_id() const { return ptr->get_type_id(idx); }
  };

  struct ConstProxy {
    const Multi_MOL_parameters *ptr;
    size_t idx;

    double get_mass() const {
      return (ptr->mass[idx] > 0.01) ? ptr->mass[idx] : 0.01;
    }
    double get_charge() const { return ptr->charge[idx]; }
    double get_radius() const { return ptr->radius[idx]; }
    double get_epsilon() const { return ptr->epsilon[idx]; }
    double get_sqrt_epsilon() const { return ptr->sqrt_epsilon[idx]; }
    double get_sigma() const { return ptr->sigma[idx]; }
    int32_t get_type_id() const { return ptr->get_type_id(idx); }
  };

  Proxy operator[](size_t i) { return Proxy{this, i}; }
  ConstProxy operator[](size_t i) const { return ConstProxy{this, i}; }

  size_t size() const { return charge.size(); }
  bool empty() const { return charge.empty(); }
  bool has_type_ids() const {
    return !type_id.empty() && type_id.size() == charge.size();
  }
  void ensure_type_ids() {
    if (!has_type_ids())
      type_id.assign(charge.size(), -1);
  }
  void release_type_ids() {
    type_id.clear();
    type_id.shrink_to_fit();
  }
  int32_t get_type_id(size_t idx) const {
    return idx < type_id.size() ? type_id[idx] : -1;
  }
  void set_type_id(size_t idx, int32_t id) {
    ensure_type_ids();
    type_id[idx] = id;
  }
  const int32_t *type_id_ptr() const noexcept {
    return has_type_ids() ? type_id.data() : nullptr;
  }
  void clear() {
    charge.clear();
    radius.clear();
    sigma.clear();
    epsilon.clear();
    sqrt_epsilon.clear();
    mass.clear();
    type_id.clear();
  }
  void resize(size_t n) {
    charge.resize(n, 0.0);
    radius.resize(n, 0.0);
    sigma.resize(n, 0.0);
    epsilon.resize(n, 0.0);
    sqrt_epsilon.resize(n, 0.0);
    mass.resize(n, 1.0);
    if (!type_id.empty())
      type_id.resize(n, -1);
  }

  // Support range-based for loops
  struct Iterator {
    Multi_MOL_parameters *ptr;
    size_t idx;
    Proxy operator*() { return Proxy{ptr, idx}; }
    Iterator &operator++() {
      ++idx;
      return *this;
    }
    bool operator!=(const Iterator &other) const { return idx != other.idx; }
  };

  struct ConstIterator {
    const Multi_MOL_parameters *ptr;
    size_t idx;
    ConstProxy operator*() const { return ConstProxy{ptr, idx}; }
    ConstIterator &operator++() {
      ++idx;
      return *this;
    }
    bool operator!=(const ConstIterator &other) const {
      return idx != other.idx;
    }
  };

  Iterator begin() { return Iterator{this, 0}; }
  Iterator end() { return Iterator{this, charge.size()}; }
  ConstIterator begin() const { return ConstIterator{this, 0}; }
  ConstIterator end() const { return ConstIterator{this, charge.size()}; }
};
#endif
