set(TARGET llama-continuous-batch)
add_executable(${TARGET} main.cpp continuous_batch.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
