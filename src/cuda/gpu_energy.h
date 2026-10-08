#ifndef GPU_ENERGY_H
#define GPU_ENERGY_H

#include <nvml.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <stdio.h>
#include "gpu_energy.h"

// Diz ao compilador do Windows para ligar automaticamente a biblioteca NVML
#pragma comment(lib, "nvml.lib")

nvmlDevice_t nvml_device;
std::atomic<bool> nvml_keep_polling(false);
double nvml_total_joules = 0.0;
std::thread nvml_thread;

void power_polling_loop() {
    unsigned int power;
    auto last_time = std::chrono::high_resolution_clock::now();
    while (nvml_keep_polling) {
        // Retorna potência instantânea em miliwatts
        if (nvmlDeviceGetPowerUsage(nvml_device, &power) == NVML_SUCCESS) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - last_time;
            
            // Joules = Watts * Segundos
            nvml_total_joules += (power / 1000.0) * elapsed.count();
            last_time = current_time;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

int init_gpu_energy_meter() {
    if (nvmlInit() != NVML_SUCCESS) return -1;
    // Pega a primeira GPU do sistema (índice 0)
    if (nvmlDeviceGetHandleByIndex(0, &nvml_device) != NVML_SUCCESS) return -1;
    return 0;
}

void start_gpu_energy_measurement() {
    nvml_total_joules = 0.0;
    nvml_keep_polling = true;
    nvml_thread = std::thread(power_polling_loop);
}

double stop_gpu_energy_measurement() {
    nvml_keep_polling = false;
    if (nvml_thread.joinable()) {
        nvml_thread.join();
    }
    return nvml_total_joules;
}

void cleanup_gpu_energy_meter() {
    nvmlShutdown();
}

#endif