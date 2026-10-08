#include "energy_meter.h"
#include "cpucounters.h"
#include <iostream>

using namespace pcm;

static SystemCounterState before_state;
static PCM* m = nullptr;

extern "C" int init_energy_meter(void) {
    m = PCM::getInstance();
    
    // Tenta inicializar o PCM sem programar os contadores de performance customizados
    // (Passando PCM::DisableJKIAwareness ou ignorando o retorno restrito do program se necessário)
    PCM::ErrorCode status = m->program();
    if (status != PCM::Success) {
        // Se o erro for apenas de conflito de PMU, tentamos forçar ou prosseguir se o RAPL estiver disponível
        std::cout << "AVISO PCM: Status de inicializacao com codigo " << status << ". Tentando prosseguir..." << std::endl;
    }
    return 0;
}

extern "C" void start_energy_measurement(void) {
    before_state = getSystemCounterState();
}

extern "C" EnergyReading stop_energy_measurement(void) {
    SystemCounterState after_state = getSystemCounterState();
    EnergyReading reading;
    
    reading.cpu_joules = getConsumedJoules(before_state, after_state);
    reading.dram_joules = getDRAMConsumedJoules(before_state, after_state);
    
    return reading;
}

extern "C" void cleanup_energy_meter(void) {
    if (m) m->cleanup();
}