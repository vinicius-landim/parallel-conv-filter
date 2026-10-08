#ifndef ENERGY_METER_H
#define ENERGY_METER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double cpu_joules;
    double dram_joules;
} EnergyReading;

int init_energy_meter(void);
void start_energy_measurement(void);
EnergyReading stop_energy_measurement(void);
void cleanup_energy_meter(void);

#ifdef __cplusplus
}
#endif

#endif