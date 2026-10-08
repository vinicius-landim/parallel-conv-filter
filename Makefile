CC = cl.exe

CFLAGS = /O2 /EHsc /openmp /I"src" /I"lib" /I"C:\Users\Desktop\Downloads\Repositorios\PCD\intel PCM\pcm\src"
LDFLAGS = /link "C:\Users\Desktop\Downloads\Repositorios\PCD\intel PCM\pcm\build_limpo\src\pcm-lib.lib" Advapi32.lib

SERIAL_SRC = src\serial\conv_filter.c
OMP_SRC = src\openmp\conv_filter.c
PCM_SRC = src\energy_meter.cpp

SERIAL_BIN = src\serial\conv_filter.exe
OMP_BIN = src\openmp\conv_filter.exe

all: build bench

build: $(SERIAL_BIN) $(OMP_BIN)

$(SERIAL_BIN): $(SERIAL_SRC) $(PCM_SRC)
	$(CC) $(CFLAGS) $(SERIAL_SRC) $(PCM_SRC) /Fe"$(SERIAL_BIN)" $(LDFLAGS)
	del *.obj

$(OMP_BIN): $(OMP_SRC) $(PCM_SRC)
	$(CC) $(CFLAGS) $(OMP_SRC) $(PCM_SRC) /Fe"$(OMP_BIN)" $(LDFLAGS)
	del *.obj

bench: build
	python scripts\coleta_metricas.py

clean:
	del /Q src\serial\conv_filter.exe
	del /Q src\openmp\conv_filter.exe
	rmdir /S /Q tabelas graficos