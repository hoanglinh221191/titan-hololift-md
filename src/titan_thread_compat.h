#ifndef TITAN_THREAD_COMPAT_H
#define TITAN_THREAD_COMPAT_H

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace titan {

#if defined(__cpp_lib_jthread) && (__cpp_lib_jthread >= 201911L) &&          \
    defined(__cpp_lib_stop_token) && (__cpp_lib_stop_token >= 201907L)

using jthread = std::jthread;
using stop_token = std::stop_token;

#else

class stop_token {
public:
  stop_token() = default;
  explicit stop_token(std::shared_ptr<std::atomic<bool>> state) noexcept
      : state_(std::move(state)) {}

  [[nodiscard]] bool stop_requested() const noexcept {
    return state_ && state_->load(std::memory_order_acquire);
  }

  [[nodiscard]] bool stop_possible() const noexcept {
    return static_cast<bool>(state_);
  }

private:
  std::shared_ptr<std::atomic<bool>> state_;

  friend class jthread;
};

namespace detail {

template <class Fn, class... Args>
void invoke_jthread_callable(const std::shared_ptr<std::atomic<bool>> &state,
                             Fn &&fn, Args &&...args) {
  stop_token token(state);
  if constexpr (std::is_invocable_v<Fn, stop_token, Args...>) {
    std::invoke(std::forward<Fn>(fn), token, std::forward<Args>(args)...);
  } else {
    std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }
}

} // namespace detail

class jthread {
public:
  jthread() noexcept = default;

  template <class Fn, class... Args,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<Fn>, jthread>>>
  explicit jthread(Fn &&fn, Args &&...args) {
    start(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  ~jthread() {
    request_stop();
    join();
  }

  jthread(const jthread &) = delete;
  jthread &operator=(const jthread &) = delete;

  jthread(jthread &&other) noexcept = default;

  jthread &operator=(jthread &&other) noexcept {
    if (this != &other) {
      request_stop();
      join();
      thread_ = std::move(other.thread_);
      stop_state_ = std::move(other.stop_state_);
    }
    return *this;
  }

  template <class Fn, class... Args,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<Fn>, jthread>>>
  void start(Fn &&fn, Args &&...args) {
    request_stop();
    join();

    stop_state_ = std::make_shared<std::atomic<bool>>(false);
    using FnStorage = std::decay_t<Fn>;
    using ArgsStorage = std::tuple<std::decay_t<Args>...>;

    auto stop_state = stop_state_;
    thread_ = std::thread(
        [stop_state, fn_store = FnStorage(std::forward<Fn>(fn)),
         args_store = ArgsStorage(std::forward<Args>(args)...)]() mutable {
          std::apply(
              [&](auto &&...unpacked_args) {
                detail::invoke_jthread_callable(
                    stop_state, std::move(fn_store),
                    std::forward<decltype(unpacked_args)>(unpacked_args)...);
              },
              std::move(args_store));
        });
  }

  [[nodiscard]] bool joinable() const noexcept { return thread_.joinable(); }

  void join() {
    if (thread_.joinable())
      thread_.join();
  }

  void detach() { thread_.detach(); }

  [[nodiscard]] std::thread::id get_id() const noexcept { return thread_.get_id(); }

  [[nodiscard]] stop_token get_stop_token() const noexcept {
    return stop_token(stop_state_);
  }

  bool request_stop() noexcept {
    if (!stop_state_)
      return false;
    stop_state_->store(true, std::memory_order_release);
    return true;
  }

private:
  std::thread thread_;
  std::shared_ptr<std::atomic<bool>> stop_state_;
};

#endif

} // namespace titan

#endif // TITAN_THREAD_COMPAT_H
